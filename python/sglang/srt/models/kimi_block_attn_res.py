# Copyright (c) Anthropic and contributors.
# Licensed under the Apache License, Version 2.0 (see SGLang LICENSE).

"""Block AttnRes inference model for SGLang.

Wraps SGLang's stock ``kimi_linear`` model with Block Attention Residual
weaving (Kimi paper, Figure 2). Each layer's pre-attention and pre-FFN
norm-add is replaced by a learned aggregation over the block list
(prior committed block representations + the current partial_block).

Forward flow (per layer)::

    h = block_attn_res(blocks, partial_block, attn_res_proj, attn_res_norm)
    if is_block_start: blocks.append(partial_block); partial_block = None
    attn_out = self_attn(input_layernorm(h))
    partial_block = attn_out if partial_block is None else partial_block + attn_out
    h = block_attn_res(blocks, partial_block, mlp_res_proj, mlp_res_norm)
    ffn_out = mlp(post_attention_layernorm(h))
    partial_block = partial_block + ffn_out
    -> (blocks, partial_block)

This file's scope is intentionally narrow: model definition + weight
loading + inference forward. KV cache, RadixCache, scheduler, and other
serving-runtime concerns are inherited from upstream sglang and not
modified here. The fabric-trace research interest is the parallelism
implementation (TP/PP/EP under inference) and its NCCL traffic — not
the serving stack.

Phase 10 of `torchtitan_attention_residual`: see
`phase10/dcp_to_hf_kimi_attn_res.py` for ckpt conversion and
`phase10/run_*.sh` for runners.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.kimi_linear import (
    KimiDecoderLayer,
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from sglang.srt.utils.common import BumpAllocator, add_prefix, maybe_prefix


# ---------------------------------------------------------------------------
# Block AttnRes aggregation kernel
# ---------------------------------------------------------------------------


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Inter-block attention aggregator (Kimi paper, Fig. 2).

    Pseudo-query is ``proj.weight`` (shape ``[1, D]``); values are the
    stacked blocks plus the current ``partial_block``; keys are the
    same values RMSNorm'd. Softmax over the block axis produces mixing
    weights, which then gather the value vectors back into a single
    ``[B, T, D]`` aggregated state.

    Mirrors ``torchtitan.experiments.attn_res.attn_res.block_attn_res``.
    """
    V = torch.stack(blocks + [partial_block], dim=0)  # (N+1, B, T, D)
    K = norm(V)
    weight = proj.weight  # (1, D)
    if hasattr(weight, "to_local"):
        weight = weight.to_local()
    query = weight.squeeze(0)  # (D,)
    logits = torch.einsum("d,nbtd->nbt", query, K)
    weights = F.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, V)


def _zero_init(linear: nn.Linear) -> None:
    """Paper § 5: zero-init pseudo-queries so initial softmax is uniform."""
    with torch.no_grad():
        linear.weight.zero_()


# ---------------------------------------------------------------------------
# AttnRes-wrapped decoder layer
# ---------------------------------------------------------------------------


class KimiAttnResDecoderLayer(KimiDecoderLayer):
    """Decoder layer with Block AttnRes pre-attn and pre-FFN aggregations.

    Inherits all sub-modules (``self_attn``, ``mlp``, layernorms) from
    upstream :class:`KimiDecoderLayer`. Adds 4 small parameters:

    * ``attn_res_proj`` — pseudo-query for pre-attention aggregation
    * ``attn_res_norm`` — RMSNorm for keys in that aggregation
    * ``mlp_res_proj``  — pseudo-query for pre-FFN aggregation
    * ``mlp_res_norm``  — RMSNorm for keys in that aggregation

    The pseudo-query projections are ``Linear(D, 1, bias=False)``;
    their weight ``[1, D]`` IS the per-layer pseudo-query ``w_l``.
    Replicated across TP (no sharding — they're tiny).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        d = config.hidden_size
        # Replicate across TP — these are O(D) parameters, not worth sharding.
        self.attn_res_proj = ReplicatedLinear(
            d, 1, bias=False, prefix=add_prefix("attn_res_proj", prefix),
        )
        self.mlp_res_proj = ReplicatedLinear(
            d, 1, bias=False, prefix=add_prefix("mlp_res_proj", prefix),
        )
        self.attn_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
        _zero_init(self.attn_res_proj)
        _zero_init(self.mlp_res_proj)

    # NOTE: the upstream forward signature is
    #   (positions, hidden_states, forward_batch, residual, zero_allocator)
    # and returns (hidden_states, residual). We override to thread blocks
    # explicitly via the new signature below; the parent Model knows to
    # call this method instead of the upstream one.
    def forward_attn_res(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        is_block_start: bool,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Pre-attention aggregation.
        h = block_attn_res(
            blocks, partial_block, self.attn_res_proj, self.attn_res_norm,
        )

        # Block boundary: commit current partial into blocks list.
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None

        # Self-attention (non-fused norm: AttnRes input has no residual).
        attn_in = self.input_layernorm(h)
        attn_out = self.self_attn(
            hidden_states=attn_in,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # Pre-FFN aggregation.
        h = block_attn_res(
            blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm,
        )

        # FFN (MoE for sparse layers, dense KimiMLP otherwise).
        ffn_in = self.post_attention_layernorm(h)
        ffn_out = self.mlp(ffn_in)
        partial_block = partial_block + ffn_out

        return blocks, partial_block


# ---------------------------------------------------------------------------
# AttnRes model (replaces decoder layer iteration)
# ---------------------------------------------------------------------------


def _stack_blocks(block_list: list[torch.Tensor]) -> torch.Tensor:
    """[N x (B,T,D)] -> (N, B, T, D). PP cross-stage transport format."""
    if not block_list:
        # Sentinel for stage 0: empty stack with correct trailing shape.
        # The receiver detects len==0 and starts with empty list.
        raise ValueError("Cannot stack empty block list — caller must guard.")
    return torch.stack(block_list, dim=0)


def _unstack_blocks(blocks: torch.Tensor) -> list[torch.Tensor]:
    """(N, B, T, D) -> [N x (B,T,D)]."""
    return [blocks[i] for i in range(blocks.shape[0])]


class KimiBlockAttnResModel(KimiLinearModel):
    """Model wrapper that threads the AttnRes block list through layers.

    Replaces upstream ``KimiLinearModel.forward``'s residual flow with
    Block AttnRes block-list flow. Layer construction (FSDP/TP/EP plumbing
    is delegated to upstream's ``make_layers``) is identical except the
    layer factory swaps in :class:`KimiAttnResDecoderLayer`.

    PP transport: ``hidden_states`` carries the current ``partial_block``
    and a stacked-blocks tensor under proxy key ``"blocks"`` is shipped
    alongside; the last stage materializes the final aggregation +
    norm head.
    """

    # Override the per-layer constructor used by ``make_layers``.
    _DECODER_LAYER_CLS = KimiAttnResDecoderLayer

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # Call grandparent (nn.Module) __init__, then re-build sub-modules
        # with our layer class. Re-implementing the parent body here is
        # cleaner than monkey-patching ``make_layers`` mid-flight.
        nn.Module.__init__(self)
        from sglang.srt.utils import make_layers
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: self._DECODER_LAYER_CLS(
                config=config,
                layer_idx=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            d = config.hidden_size
            # Final aggregation (post-last-layer): same shape as per-layer
            # AttnRes, used once before the final norm + lm_head.
            self.final_attn_res_proj = ReplicatedLinear(
                d, 1, bias=False, prefix=f"{prefix}.final_attn_res_proj",
            )
            self.final_attn_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
            _zero_init(self.final_attn_res_proj)
        else:
            self.norm = PPMissingLayer()
            self.final_attn_res_proj = None
            self.final_attn_res_norm = None

        world_size = get_tensor_model_parallel_world_size()
        assert (
            config.num_attention_heads % world_size == 0
        ), "num_attention_heads must be divisible by world_size"

        # Block boundary detection (Kimi paper: layer_idx % layers_per_block == 0).
        n_blocks = getattr(config, "attn_res_num_blocks", 4)
        self.layers_per_block = max(1, config.num_hidden_layers // n_blocks)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # First-stage init.
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                h = inputs_embeds
            else:
                h = self.embed_tokens(input_ids)
            block_list: list[torch.Tensor] = []
            partial_block = h
        else:
            assert pp_proxy_tensors is not None
            partial_block = pp_proxy_tensors["hidden_states"]
            blocks_t = pp_proxy_tensors.get("blocks")
            block_list = _unstack_blocks(blocks_t) if (blocks_t is not None and blocks_t.shape[0] > 0) else []

        zero_allocator = BumpAllocator(
            buffer_size=1024,
            dtype=torch.float32,
            device=partial_block.device,
        )

        # Iterate this stage's slice of the layers.
        for layer_idx_in_module, layer in enumerate(self.layers):
            # Use the layer's stored layer_idx (set by upstream KimiDecoderLayer
            # via the make_layers prefix), not the local iteration index.
            global_idx = self.start_layer + layer_idx_in_module
            is_block_start = (global_idx % self.layers_per_block == 0)
            block_list, partial_block = layer.forward_attn_res(
                blocks=block_list,
                partial_block=partial_block,
                is_block_start=is_block_start,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            # Send (partial_block, blocks) to next PP stage.
            blocks_to_send = (
                _stack_blocks(block_list) if block_list
                else partial_block.new_zeros((0, *partial_block.shape))
            )
            return PPProxyTensors(
                {"hidden_states": partial_block, "blocks": blocks_to_send}
            )

        # Last stage: final aggregation + norm.
        h_final = block_attn_res(
            block_list, partial_block,
            self.final_attn_res_proj, self.final_attn_res_norm,
        )
        return self.norm(h_final)


# ---------------------------------------------------------------------------
# Top-level entry: KimiBlockAttnResForCausalLM
# ---------------------------------------------------------------------------


class KimiBlockAttnResForCausalLM(KimiLinearForCausalLM):
    """Causal-LM head wrapping :class:`KimiBlockAttnResModel`.

    Inherits ``forward`` and ``load_weights`` from upstream — both work
    as-is because:

    * ``forward`` calls ``self.model(...)`` which is now AttnRes-aware.
    * ``load_weights`` walks ``named_parameters()``, which now includes
      the AttnRes-specific params (``attn_res_proj.weight``,
      ``attn_res_norm.weight``, ``mlp_res_proj.weight``,
      ``mlp_res_norm.weight``, ``final_attn_res_proj.weight``,
      ``final_attn_res_norm.weight``) and matches them to keys with
      the same names in the HF safetensors (see
      ``phase10/dcp_to_hf_kimi_attn_res.py`` for the conversion).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Skip upstream KimiLinearForCausalLM.__init__'s self.model
        # construction — we want our AttnRes model class instead — and
        # call grandparent (nn.Module) __init__ directly.
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.model = KimiBlockAttnResModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model"),
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)


EntryClass = KimiBlockAttnResForCausalLM

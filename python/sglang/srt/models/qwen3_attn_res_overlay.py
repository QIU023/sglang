# Copyright (c) Anthropic and contributors.
# Licensed under the Apache License, Version 2.0 (see SGLang LICENSE).

"""Block AttnRes overlay on Qwen3 dense — generality demonstration.

Second carrier integration after the Kimi Linear overlay
(``models/attn_res_overlay.py``). Picked **Qwen3 dense** specifically
because it is **maximally different** from Kimi Linear:

* No MLA (vanilla GQA + RoPE, not Multi-head Latent Attention)
* No KDA (no linear-attention layers)
* No MoE (single dense MLP per layer, ``Qwen3MLP``)
* Uses the ``LayerCommunicator`` abstraction (not raw layernorm + add)

If the Block AttnRes algorithm in ``sglang.srt.layers.attn_res`` works
unchanged here, that's the strongest possible "model-agnostic overlay"
claim — the algorithm is the same one-liner, only the per-model wiring
(decoder-layer wrapping, PP block transport, AttnRes param init) differs.

The Kimi-overlay's compatibility patches do **not** apply here:
* No MLA → no ``w_kc/w_vc`` post-load hook
* No MoE gate → no ``e_score_correction_bias`` fp32 patch
* No ``compressed_kv.split`` → no RMSNorm-contiguous wrap needed
* Architecture name is fresh (``Qwen3BlockAttnResForCausalLM``) and Qwen3
  doesn't have arch-name-gated detection in ``model_config.py``, so the
  dual-arch hint isn't needed either.

What stays identical to the Kimi overlay (proves these patterns are the
*generic* part of an AttnRes integration, not Kimi-specific):
* Per-layer params: ``attn_res_proj``, ``attn_res_norm``,
  ``mlp_res_proj``, ``mlp_res_norm`` — ``ReplicatedLinear(d, 1)`` plus
  RMSNorm — zero-init pseudo-queries.
* Model-head params: ``final_attn_res_proj``, ``final_attn_res_norm``.
* Block list transport across PP via ``pp_proxy_tensors["blocks"]``
  (stack/unstack helpers in ``_stack_blocks`` / ``_unstack_blocks``).
* ``forward_attn_res`` per-layer dispatch using
  :func:`sglang.srt.layers.attn_res.block_attn_res`.
* PP filter in ``load_weights`` to skip top-level params not present on
  this rank's stage.

Repo positioning: this file is the proof that
``sglang/srt/layers/attn_res.py`` (algorithm) +
``models/<model>_attn_res*.py`` (thin per-model wrappers) is the right
factoring, mirroring DSv4's ``layers/mhc.py`` + ``models/deepseek_v4.py``
pattern.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attn_res import (
    all_gather_seq,
    block_attn_res,
    block_attn_res_phase1,
    block_attn_res_phase2_merge,
    reduce_scatter_seq,
    split_seq,
    zero_init_pseudo_query,
)

import logging as _logging
import os as _os
_SEQ_SHARD_ENABLED = bool(int(_os.environ.get("SGLANG_ATTN_RES_SEQ_SHARD", "0")))
# Bench-only — see attn_res_overlay.py for full doc.
_BYPASS_ATTN_RES = bool(int(_os.environ.get("SGLANG_ATTN_RES_BYPASS", "0")))
_FORCE_NAIVE_PATH = bool(int(_os.environ.get("SGLANG_ATTN_RES_NAIVE_PATH", "0")))
_logger = _logging.getLogger(__name__)


def _query_from_proj(proj):
    weight = proj.weight  # [1, D]
    if hasattr(weight, "to_local"):
        weight = weight.to_local()
    return weight.squeeze(0)  # [D]
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from sglang.srt.utils.common import add_prefix


def _maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


# ---------------------------------------------------------------------------
# AttnRes-wrapped Qwen3 decoder layer
# ---------------------------------------------------------------------------


class Qwen3AttnResDecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer with Block AttnRes pre-attention/pre-FFN aggregation.

    Inherits ``self_attn``, ``mlp``, ``input_layernorm``,
    ``post_attention_layernorm`` (and the ``layer_communicator`` book-
    keeping which we **bypass** in :meth:`forward_attn_res` because
    AttnRes replaces the residual stream rather than adding to it).

    Adds 4 small parameters (per layer, replicated across TP):
    ``attn_res_proj``, ``attn_res_norm``, ``mlp_res_proj``, ``mlp_res_norm``.
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        d = config.hidden_size
        self.attn_res_proj = ReplicatedLinear(
            d, 1, bias=False, prefix=add_prefix("attn_res_proj", prefix),
        )
        self.mlp_res_proj = ReplicatedLinear(
            d, 1, bias=False, prefix=add_prefix("mlp_res_proj", prefix),
        )
        self.attn_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
        zero_init_pseudo_query(self.attn_res_proj)
        zero_init_pseudo_query(self.mlp_res_proj)

        # Seq-shard: disable in-projection all-reduce so caller can RS.
        if _SEQ_SHARD_ENABLED:
            # Qwen3Attention.o_proj is RowParallelLinear (default reduce_results=True).
            self.self_attn.o_proj.reduce_results = False
            # Qwen3MLP.down_proj is RowParallelLinear too.
            if hasattr(self.mlp, "down_proj") and hasattr(self.mlp.down_proj, "reduce_results"):
                self.mlp.down_proj.reduce_results = False

    def _run_attn(
        self,
        attn_input: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        seq_shard: bool = False,
    ) -> torch.Tensor:
        """Single-arg layernorm + Qwen3 self-attention.

        Bypasses ``layer_communicator.prepare_attn`` (the fused PreNorm
        residual-add path) because Block AttnRes replaces the residual
        stream. When ``seq_shard=True``, RMSNorm runs on (T/P, D), then
        all-gather to (T, D) for self_attn input, then reduce-scatter
        the o_proj partial-sum back to (T/P, D).
        """
        if attn_input.shape[0] == 0:
            return attn_input
        if seq_shard:
            attn_in_shard = self.input_layernorm(attn_input)
            attn_in_replicated = all_gather_seq(attn_in_shard)
            attn_out_partial = self.self_attn(
                positions=positions,
                hidden_states=attn_in_replicated,
                forward_batch=forward_batch,
            )
            return reduce_scatter_seq(attn_out_partial)
        attn_in = self.input_layernorm(attn_input)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=attn_in,
            forward_batch=forward_batch,
        )
        # See attn_res_overlay.py for the rationale: seq-shard's
        # ``o_proj.reduce_results=False`` patch sticks at __init__, so
        # the fallback path here returns a partial sum that must be
        # explicitly all-reduced to match vanilla.
        if _SEQ_SHARD_ENABLED:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size,
                tensor_model_parallel_all_reduce,
            )
            if get_tensor_model_parallel_world_size() > 1:
                attn_out = tensor_model_parallel_all_reduce(attn_out)
        return attn_out

    def _run_mlp(self, mlp_input: torch.Tensor, seq_shard: bool = False) -> torch.Tensor:
        if seq_shard:
            ffn_in_shard = self.post_attention_layernorm(mlp_input)
            ffn_in_replicated = all_gather_seq(ffn_in_shard)
            mlp_partial = self.mlp(ffn_in_replicated)  # down_proj.reduce_results=False
            return reduce_scatter_seq(mlp_partial)
        ffn_in = self.post_attention_layernorm(mlp_input)
        mlp_out = self.mlp(ffn_in)
        if _SEQ_SHARD_ENABLED:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size,
                tensor_model_parallel_all_reduce,
            )
            if get_tensor_model_parallel_world_size() > 1:
                mlp_out = tensor_model_parallel_all_reduce(mlp_out)
        return mlp_out


# ---------------------------------------------------------------------------
# AttnRes Qwen3 model
# ---------------------------------------------------------------------------


def _stack_blocks(block_list: list[torch.Tensor]) -> torch.Tensor:
    if not block_list:
        raise ValueError("Cannot stack empty block list")
    return torch.stack(block_list, dim=0)


def _unstack_blocks(blocks: torch.Tensor) -> list[torch.Tensor]:
    return [blocks[i] for i in range(blocks.shape[0])]


class Qwen3BlockAttnResModel(Qwen3Model):
    """Qwen3 model wrapper that threads the AttnRes block list through layers.

    Constructs ``Qwen3AttnResDecoderLayer`` instances via the ``decoder_layer_type``
    hook on ``Qwen2Model.__init__`` (``Qwen3Model.__init__`` already passes
    ``decoder_layer_type=Qwen3DecoderLayer``; we override that).

    Adds two model-head AttnRes params (last PP rank only):
    ``final_attn_res_proj``, ``final_attn_res_norm`` — used to collapse
    the block list into the lm_head input after the final decoder layer.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # ``Qwen2Model.__init__`` accepts ``decoder_layer_type`` but
        # ``Qwen3Model.__init__`` hard-codes Qwen3DecoderLayer. Skip the
        # Qwen3Model body and call grandparent (Qwen2Model) directly so we
        # can swap the decoder layer class.
        from sglang.srt.models.qwen2 import Qwen2Model
        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        Qwen2Model.__init__(
            self,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3AttnResDecoderLayer,
            alt_stream=alt_stream,
        )

        # AttnRes model-head: only last PP rank materialises the final
        # aggregation pseudo-query and norm.
        if self.pp_group.is_last_rank:
            d = config.hidden_size
            self.final_attn_res_proj = ReplicatedLinear(
                d, 1, bias=False,
                prefix=add_prefix("final_attn_res_proj", prefix),
            )
            self.final_attn_res_norm = RMSNorm(d, eps=config.rms_norm_eps)
            zero_init_pseudo_query(self.final_attn_res_proj)
        else:
            self.final_attn_res_proj = None
            self.final_attn_res_norm = None

        # Block-boundary detection.
        n_blocks = getattr(config, "attn_res_num_blocks", 4)
        self.layers_per_block = max(1, config.num_hidden_layers // n_blocks)

    def _seq_shard_active(self, partial_block: torch.Tensor) -> bool:
        """See ``KimiBlockAttnResModel._seq_shard_active`` for full doc.

        Same three-condition gate; logs a once-per-instance warning when
        seq-shard is requested but num_tokens is not TP-divisible (typical
        on batch=1 decode steps).
        """
        if not _SEQ_SHARD_ENABLED:
            return False
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return False
        if partial_block.shape[0] % tp_size != 0:
            if not getattr(self, "_seq_shard_fallback_warned", False):
                _logger.warning(
                    "AttnRes seq-shard requested (SGLANG_ATTN_RES_SEQ_SHARD=1) "
                    "but num_tokens=%d is not divisible by TP=%d on this "
                    "forward; falling back to replicated mode for this call.",
                    partial_block.shape[0], tp_size,
                )
                self._seq_shard_fallback_warned = True
            return False
        return True

    def _forward_one_block(
        self,
        committed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layers_in_block: list,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        seq_shard: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Two-phase optimized forward over a block of layers."""
        L0 = layers_in_block[0]
        if committed_blocks:
            cache0 = block_attn_res_phase1(
                committed_blocks,
                [_query_from_proj(L0.attn_res_proj)],
                [L0.attn_res_norm],
            )[0]
        else:
            cache0 = None
        h = block_attn_res_phase2_merge(
            cache0, partial_block,
            _query_from_proj(L0.attn_res_proj), L0.attn_res_norm,
        )

        committed_blocks = committed_blocks + [partial_block]

        rest_queries: list[torch.Tensor] = [_query_from_proj(L0.mlp_res_proj)]
        rest_norms: list = [L0.mlp_res_norm]
        for L in layers_in_block[1:]:
            rest_queries.extend([
                _query_from_proj(L.attn_res_proj),
                _query_from_proj(L.mlp_res_proj),
            ])
            rest_norms.extend([L.attn_res_norm, L.mlp_res_norm])
        rest_cache = block_attn_res_phase1(
            committed_blocks, rest_queries, rest_norms,
        )

        cache_idx = 0

        # Layer 0
        attn_out = L0._run_attn(h, positions, forward_batch, seq_shard=seq_shard)
        partial_block = attn_out
        h = block_attn_res_phase2_merge(
            rest_cache[cache_idx], partial_block,
            rest_queries[cache_idx], rest_norms[cache_idx],
        )
        cache_idx += 1
        ffn_out = L0._run_mlp(h, seq_shard=seq_shard)
        partial_block = partial_block + ffn_out

        # Layers 1..L_block-1
        for L in layers_in_block[1:]:
            h = block_attn_res_phase2_merge(
                rest_cache[cache_idx], partial_block,
                rest_queries[cache_idx], rest_norms[cache_idx],
            )
            cache_idx += 1
            attn_out = L._run_attn(h, positions, forward_batch, seq_shard=seq_shard)
            partial_block = partial_block + attn_out
            h = block_attn_res_phase2_merge(
                rest_cache[cache_idx], partial_block,
                rest_queries[cache_idx], rest_norms[cache_idx],
            )
            cache_idx += 1
            ffn_out = L._run_mlp(h, seq_shard=seq_shard)
            partial_block = partial_block + ffn_out

        return committed_blocks, partial_block

    def _naive_per_layer_step(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer,
        is_block_start: bool,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = block_attn_res(blocks, partial_block, layer.attn_res_proj, layer.attn_res_norm)
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None
        attn_out = layer._run_attn(h, positions, forward_batch)
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        h = block_attn_res(blocks, partial_block, layer.mlp_res_proj, layer.mlp_res_norm)
        ffn_out = layer._run_mlp(h)
        partial_block = partial_block + ffn_out
        return blocks, partial_block

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            partial_block = (
                input_embeds if input_embeds is not None
                else self.embed_tokens(input_ids)
            )
            block_list: list[torch.Tensor] = []
        else:
            assert pp_proxy_tensors is not None
            partial_block = pp_proxy_tensors["hidden_states"]
            blocks_t = pp_proxy_tensors.tensors.get("blocks")
            block_list = (
                _unstack_blocks(blocks_t)
                if (blocks_t is not None and blocks_t.shape[0] > 0)
                else []
            )

        # Two-phase per-block iteration (mirrors KimiBlockAttnResModel).
        # See ``models/attn_res_overlay.py`` and the Zhihu blog
        # (https://zhuanlan.zhihu.com/p/2017528295286133070) for the full
        # IO-amortisation argument.
        L_block = self.layers_per_block
        layer_indices = list(range(self.start_layer, self.end_layer))
        block_aligned = (
            len(layer_indices) > 0
            and layer_indices[0] % L_block == 0
            and len(layer_indices) % L_block == 0
        )

        seq_shard_active = self._seq_shard_active(partial_block)
        if seq_shard_active and self.pp_group.is_first_rank:
            partial_block = split_seq(partial_block)

        # Bench-only branches: see attn_res_overlay.py for full doc on
        # SGLANG_ATTN_RES_BYPASS / SGLANG_ATTN_RES_NAIVE_PATH semantics.
        if _BYPASS_ATTN_RES:
            if seq_shard_active:
                partial_block = all_gather_seq(partial_block)
                seq_shard_active = False
            for global_idx in layer_indices:
                layer = self.layers[global_idx]
                attn_out = layer._run_attn(
                    partial_block, positions, forward_batch, seq_shard=False,
                )
                partial_block = partial_block + attn_out
                ffn_out = layer._run_mlp(partial_block, seq_shard=False)
                partial_block = partial_block + ffn_out
        elif _FORCE_NAIVE_PATH or not block_aligned:
            if seq_shard_active:
                partial_block = all_gather_seq(partial_block)
                block_list = [all_gather_seq(b) for b in block_list]
                seq_shard_active = False
            for global_idx in layer_indices:
                layer = self.layers[global_idx]
                is_block_start = (global_idx % L_block == 0)
                block_list, partial_block = self._naive_per_layer_step(
                    block_list, partial_block, layer, is_block_start,
                    positions, forward_batch,
                )
        else:
            for chunk_start in range(0, len(layer_indices), L_block):
                chunk = layer_indices[chunk_start : chunk_start + L_block]
                layers_in_block = [self.layers[i] for i in chunk]
                block_list, partial_block = self._forward_one_block(
                    block_list, partial_block, layers_in_block,
                    positions, forward_batch,
                    seq_shard=seq_shard_active,
                )

        if not self.pp_group.is_last_rank:
            blocks_to_send = (
                _stack_blocks(block_list) if block_list
                else partial_block.new_zeros((0, *partial_block.shape))
            )
            return PPProxyTensors(
                {"hidden_states": partial_block, "blocks": blocks_to_send}
            )

        # Last stage: final aggregation + norm. Both ops commute with
        # seq-dim sharding (per-token), so they run on (T/P, D) shards
        # when seq-shard is active; all-gather restores (T, D) before lm_head.
        h_final = block_attn_res(
            block_list, partial_block,
            self.final_attn_res_proj, self.final_attn_res_norm,
        )
        h_normed = self.norm(h_final)
        if seq_shard_active:
            h_normed = all_gather_seq(h_normed)
        return h_normed


# ---------------------------------------------------------------------------
# Top-level entry: Qwen3BlockAttnResForCausalLM
# ---------------------------------------------------------------------------


class Qwen3BlockAttnResForCausalLM(Qwen3ForCausalLM):
    """Causal-LM head wrapping :class:`Qwen3BlockAttnResModel`.

    Inherits ``forward`` (which calls ``self.model(...)``) and weight-load
    machinery from ``Qwen3ForCausalLM``; only the model-construction step
    in __init__ is overridden to instantiate the AttnRes-aware model.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # AttnRes overlay does not currently support DP attention. See the
        # Kimi overlay for the full reasoning; same constraint applies here
        # because we similarly bypass ``layer_communicator.prepare_attn``.
        try:
            from sglang.srt.layers.dp_attention import is_dp_attention_enabled
            if is_dp_attention_enabled():
                raise NotImplementedError(
                    "Block AttnRes overlay (Qwen3 carrier) does not yet support "
                    "DP attention. Disable via --disable-dp-attention."
                )
        except ImportError:
            pass

        # Skip Qwen3ForCausalLM.__init__'s self.model construction and
        # call grandparent (nn.Module) directly so we can wire in our
        # AttnRes model class instead.
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3BlockAttnResModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and getattr(
                config, "tie_word_embeddings", False
            ):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        # Pooler / aux state inherited from upstream where possible; the
        # AttnRes overlay doesn't change those concerns.
        from sglang.srt.layers.pooler import Pooler, PoolingType
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def load_weights(self, weights: Iterable):
        """PP-aware filter on top-level weights (mirrors the Kimi overlay).

        The PP-padding-aware layer iteration in ``Qwen3BlockAttnResModel.forward``
        already skips out-of-range layers, but the input weights iterator
        contains entries for every layer. ``Qwen2Model.load_weights`` (which
        ``Qwen3ForCausalLM.load_weights`` calls into upstream) does already
        skip layers via ``params_dict`` keys, but top-level params on PP
        ranks where they don't exist (lm_head on stage 0, embed_tokens on
        last stage, etc.) need explicit pre-filtering.
        """
        is_first = self.pp_group.is_first_rank
        is_last = self.pp_group.is_last_rank
        start = self.model.start_layer
        end = self.model.end_layer

        last_only_top = (
            "model.norm.weight",
            "lm_head.weight",
            "model.final_attn_res_proj.weight",
            "model.final_attn_res_norm.weight",
        )
        first_only_top = ("model.embed_tokens.weight",)

        def _filter():
            for entry in weights:
                name = entry[0]
                if name in last_only_top and not is_last:
                    continue
                if name in first_only_top and not is_first:
                    continue
                if name.startswith("model.layers."):
                    rest = name[len("model.layers.") :]
                    idx_str, _, _ = rest.partition(".")
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        yield entry
                        continue
                    if idx < start or idx >= end:
                        continue
                yield entry

        return super().load_weights(_filter())


EntryClass = Qwen3BlockAttnResForCausalLM

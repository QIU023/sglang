# Copyright (c) Anthropic and contributors.
# Licensed under the Apache License, Version 2.0 (see SGLang LICENSE).

"""Block Attention Residual inference overlay for SGLang.

Block AttnRes (Kimi paper § 5, Fig. 2) is a generic *residual-stream
overlay*: it replaces a stock decoder's fixed pre-norm residual stream
with a learned aggregation over a list of prior block representations
plus the current partial block. The construct is in the same family as
ByteDance's Hyper-Connections (arxiv 2409.19606) and DeepSeek's mHC
(arxiv 2512.24880) — multi-stream residual variants — and is intended
to be portable to any SGLang-supported decoder model.

This file currently houses the **Kimi Linear specialisation**
(:class:`KimiBlockAttnResForCausalLM`). When porting to other model
families (Qwen3 MoE, DeepSeek-V3, BailingMoeLinear …), the intent is
to extract :func:`block_attn_res`, :class:`KimiAttnResDecoderLayer.forward_attn_res`,
and :class:`KimiBlockAttnResModel.forward` into a model-agnostic mixin
sibling module under ``sglang.srt.layers.attn_res/``, and have each
``XxxBlockAttnResForCausalLM`` be a thin wrapper. That refactor is
deferred until a second model is integrated and the genuinely-shared
shape vs Kimi-specific quirks separate cleanly.

Forward flow (per layer)::

    h = block_attn_res(blocks, partial_block, attn_res_proj, attn_res_norm)
    if is_block_start: blocks.append(partial_block); partial_block = None
    attn_out = self_attn(input_layernorm(h))
    partial_block = attn_out if partial_block is None else partial_block + attn_out
    h = block_attn_res(blocks, partial_block, mlp_res_proj, mlp_res_norm)
    ffn_out = mlp(post_attention_layernorm(h))
    partial_block = partial_block + ffn_out
    -> (blocks, partial_block)

KV cache, RadixCache, scheduler, and other serving-runtime concerns
are inherited from upstream SGLang and not modified here. The
research focus is the *parallelism* implementation (TP/PP/EP under
inference) and its NCCL fabric pattern — not the serving stack.
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
from sglang.srt.utils.common import BumpAllocator, add_prefix

# Pure algorithm — sibling to ``sglang.srt.layers.mhc`` (DeepSeek-V4
# Manifold-Constrained Hyper-Connections). Per-model wiring stays here.
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
# Bench-only toggles (NOT meant for end users):
#   SGLANG_ATTN_RES_BYPASS=1     — skip every aggregation, run vanilla
#                                  PreNorm. Gives a fair "no-AttnRes
#                                  baseline" using the same overlay class
#                                  + same ckpt + same env-compat patches,
#                                  isolating purely the AttnRes cost.
#   SGLANG_ATTN_RES_NAIVE_PATH=1 — force the naive per-layer aggregator
#                                  (every layer reads every committed
#                                  block) instead of the two-phase batched
#                                  path. Lets the bench harness measure
#                                  the two-phase IO-amortisation gain.
_BYPASS_ATTN_RES = bool(int(_os.environ.get("SGLANG_ATTN_RES_BYPASS", "0")))
_FORCE_NAIVE_PATH = bool(int(_os.environ.get("SGLANG_ATTN_RES_NAIVE_PATH", "0")))

_logger = _logging.getLogger(__name__)


def _kimi_moe_partial_sum(mlp, hidden_states: torch.Tensor) -> torch.Tensor:
    """Inline replica of ``KimiMoE.forward`` minus the trailing all-reduce.

    Upstream ``KimiMoE.forward`` ends with ``tensor_model_parallel_all_reduce``
    on the experts+shared_experts sum, returning a *replicated* result.
    For the seq-shard inference path we instead want the *partial-sum*
    output so the caller can ``reduce_scatter`` along the seq dim.

    KimiMoE has no ``reduce_results`` flag (unlike ``RowParallelLinear``),
    so we replicate the body of its forward minus the final AR, calling
    the same submodules (``gate``, ``topk``, ``experts``, optional
    ``shared_experts``) in the same order. Stays consistent with upstream
    if ``KimiMoE.forward``'s sub-call sequence stays stable; if upstream
    rearranges it, we'd need to mirror.

    The alt_stream branch in upstream is a perf optimisation that overlaps
    shared_experts with the experts kernel via a CUDA stream — we follow
    the simpler (no-alt_stream) branch since this overlay's primary
    target is correctness, not throughput.
    """
    num_tokens, hidden_size = hidden_states.shape
    h = hidden_states.view(-1, hidden_size)

    shared_output = None
    if mlp.num_shared_experts is not None and h.shape[0] > 0:
        shared_output = mlp.shared_experts(h)

    router_logits, _ = mlp.gate(h)
    topk_output = mlp.topk(h, router_logits)
    final_hidden_states = mlp.experts(h, topk_output)

    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output

    # NB: trailing tensor_model_parallel_all_reduce intentionally elided.
    return final_hidden_states.view(num_tokens, hidden_size)


def _is_kimi_moe(mlp) -> bool:
    """Detect whether a layer's ``mlp`` is KimiMoE (has hardcoded AR).

    Avoids importing KimiMoE directly (circular under some load orders);
    duck-types on the ``num_shared_experts`` + ``gate`` + ``topk`` + ``experts``
    attributes that ``_kimi_moe_partial_sum`` depends on.
    """
    return (
        hasattr(mlp, "num_shared_experts")
        and hasattr(mlp, "gate")
        and hasattr(mlp, "topk")
        and hasattr(mlp, "experts")
    )


def _query_from_proj(proj: nn.Linear) -> torch.Tensor:
    weight = proj.weight  # [1, D]
    if hasattr(weight, "to_local"):
        weight = weight.to_local()
    return weight.squeeze(0)  # [D]


def maybe_prefix(prefix: str, name: str) -> str:
    """Compose ``prefix.name`` if prefix is non-empty, else ``name``.

    Inlined from ``sglang.srt.models.transformers.maybe_prefix`` to avoid
    importing the heavy transformers shim from a model module.
    """
    return name if not prefix else f"{prefix}.{name}"


# Kept as a private alias so existing call sites in this file don't churn;
# new code should call ``zero_init_pseudo_query`` directly.
_zero_init = zero_init_pseudo_query


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

        # Seq-shard mode: disable in-projection all-reduce so the caller
        # can reduce-scatter the partial sum along seq dim and run AttnRes
        # Phase 2 + RMSNorm on the (T/P, D) shard. Mirrors the Zhihu
        # blog's "reduce-scatter → 本地 merge → RMSNorm → all-gather" path.
        if _SEQ_SHARD_ENABLED:
            # KimiDelta + KimiMLA self_attn both use o_proj = RowParallelLinear.
            self.self_attn.o_proj.reduce_results = False
            # Dense layer 0 KimiMLP has down_proj = RowParallelLinear.
            # KimiMoE doesn't expose a flag here (its AR is hardcoded);
            # we handle it via a context manager around mlp() in the
            # per-block forward.
            mlp = self.mlp
            if hasattr(mlp, "down_proj") and hasattr(mlp.down_proj, "reduce_results"):
                mlp.down_proj.reduce_results = False
            shared = getattr(mlp, "shared_experts", None)
            if shared is not None and hasattr(shared, "down_proj") \
                    and hasattr(shared.down_proj, "reduce_results"):
                # KimiMoE.shared_experts also has its own down_proj that
                # by default does an all-reduce that we'd double-count
                # (since the parent KimiMoE then does another AR). The
                # shared_experts is constructed with reduce_results=False
                # by upstream already (line 122 of kimi_linear.py), so this
                # is just defensive.
                shared.down_proj.reduce_results = False

    def _is_mla_self_attn(self) -> bool:
        """KimiDecoderLayer's self_attn is either KimiDelta (linear KDA)
        or DeepseekV2AttentionMLA. Detect MLA by the projections it owns.
        Cached on first call."""
        cached = getattr(self, "_attnres_is_mla_cached", None)
        if cached is not None:
            return cached
        sa = self.self_attn
        is_mla = (
            hasattr(sa, "kv_a_proj_with_mqa")
            and hasattr(sa, "kv_b_proj")
            and hasattr(sa, "kv_a_layernorm")
        )
        self._attnres_is_mla_cached = is_mla
        return is_mla

    def _mla_forward_fp32(
        self,
        attn_in: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Eager fp32 MLA forward (EXTEND/prefill mode only).

        Reproduces ``forward_normal_prepare`` + manual SDPA + ``o_proj``,
        with score/softmax/value-mul in fp32. Writes the post-norm
        ``kv_a`` and post-RoPE ``k_pe`` to SGLang's MLA KV buffer in the
        same format flashinfer_mla expects, so subsequent **decode** steps
        pull these correct K/V from cache and run native flashinfer_mla
        on top — bf16 decode is numerically OK because per-step ``attn_in``
        is one token (magnitudes don't accumulate the way prefill's do).

        On Block AttnRes layer 16 (the deepest MLA), flashinfer_mla bf16
        NaNs on prefill inputs with max~77; running prefill in fp32 fixes
        the cached K/V; native bf16 decode then attends against good
        cache → coherent generations end-to-end.

        Caller invariant: only call when forward_batch.forward_mode is
        an EXTEND-like mode. DECODE calls go through self_attn directly.
        """
        sa = self.self_attn
        # Q projection. Our config has q_lora_rank=None, so q_proj is direct.
        if getattr(sa, "q_lora_rank", None) is not None:
            q_lora = sa.q_a_proj(attn_in)[0]
            q_lora = sa.q_a_layernorm(q_lora)
            q = sa.q_b_proj(q_lora)[0]
        else:
            q = sa.q_proj(attn_in)[0]
        # [T, num_local_heads * qk_head_dim] -> [T, H, qk_head_dim]
        q = q.view(-1, sa.num_local_heads, sa.qk_head_dim)

        # KV path: kv_a_proj_with_mqa -> [T, kv_lora_rank + qk_rope_head_dim]
        latent = sa.kv_a_proj_with_mqa(attn_in)[0]
        kv_a_raw, k_pe_raw = latent.split(
            [sa.kv_lora_rank, sa.qk_rope_head_dim], dim=-1
        )
        kv_a = sa.kv_a_layernorm(kv_a_raw.contiguous())
        # k_pe is shared across heads (MQA-style) -> add head dim.
        k_pe = k_pe_raw.unsqueeze(1)  # [T, 1, qk_rope_head_dim]

        # Apply RoPE to q's PE half and to k_pe (in-place on q_pe slice
        # mirrors forward_normal_prepare).
        q_pe = q[..., sa.qk_nope_head_dim:]
        if sa.rotary_emb is not None:
            q_pe, k_pe = sa.rotary_emb(positions, q_pe, k_pe)
        q[..., sa.qk_nope_head_dim:] = q_pe

        # Write post-norm kv_a + post-RoPE k_pe to SGLang's MLA KV buffer.
        # Same layout as forward_normal_prepare so a later decode step
        # using native flashinfer_mla absorb-path reads compatible cache.
        latent_cache = torch.cat(
            [kv_a.unsqueeze(1), k_pe], dim=-1
        )  # [T, 1, kv_lora_rank + qk_rope_head_dim]
        sa._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)

        # kv_b_proj: kv_a -> [T, num_local_heads * (qk_nope + v_head)]
        kv = sa.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, sa.num_local_heads, sa.qk_nope_head_dim + sa.v_head_dim
        )
        k_nope = kv[..., : sa.qk_nope_head_dim]
        v = kv[..., sa.qk_nope_head_dim:]
        # Build full K: cat(k_nope, k_pe broadcast across heads).
        k_pe_h = k_pe.expand(-1, sa.num_local_heads, -1)
        k = torch.cat([k_nope, k_pe_h], dim=-1)  # [T, H, qk_head_dim]

        # SDPA in fp32. Need [B=1, H, T, D] layout for F.sdpa.
        T = q.shape[0]
        q4 = q.transpose(0, 1).unsqueeze(0).float()  # [1, H, T, qk_head_dim]
        k4 = k.transpose(0, 1).unsqueeze(0).float()  # [1, H, T, qk_head_dim]
        v4 = v.transpose(0, 1).unsqueeze(0).float()  # [1, H, T, v_head_dim]
        scale = getattr(sa, "scaling", 1.0 / (sa.qk_head_dim ** 0.5))
        # Causal mask is correct for EXTEND-only calls (each token in the
        # extend window attends to itself + prior extend tokens). Decode
        # uses native flashinfer_mla which handles the prefix-mask itself.
        attn_out = F.scaled_dot_product_attention(
            q4, k4, v4, is_causal=True, scale=scale,
        )  # [1, H, T, v_head_dim]
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous()
        attn_out = attn_out.to(attn_in.dtype)
        attn_out = attn_out.reshape(T, sa.num_local_heads * sa.v_head_dim)
        out, _ = sa.o_proj(attn_out)
        return out

    def _run_attn(
        self,
        attn_input: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        seq_shard: bool = False,
    ) -> torch.Tensor:
        """Single-arg layernorm + KimiDelta/MLA self-attention call.

        Bypasses upstream's fused norm+residual path
        (``input_layernorm(h, residual)`` which adds residual into normed-h)
        because Block AttnRes replaces the residual stream.

        When ``seq_shard=True``: ``attn_input`` is laid out as ``(T/P, D)``
        per TP rank. Run RMSNorm locally on the shard, all-gather to
        replicated for self_attn, then reduce-scatter the (partial-sum)
        o_proj output back to ``(T/P, D)``. Requires
        ``self.self_attn.o_proj.reduce_results=False`` to be set at __init__
        so o_proj returns the partial sum.
        """
        if seq_shard:
            attn_in_shard = self.input_layernorm(attn_input)
            attn_in_replicated = all_gather_seq(attn_in_shard)
            attn_out_partial = self.self_attn(
                hidden_states=attn_in_replicated,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
            return reduce_scatter_seq(attn_out_partial)

        # Numerical-stability: when ATTNRES_FP32_NORM=1, RMSNorm the
        # attn input in fp32 to avoid bf16 quantization of large outliers
        # (Block AttnRes accumulates an unbounded residual stream which
        # standard pre-norm transformers don't have, exposing bf16
        # underflow in RMSNorm divisor for sparse-outlier inputs).
        # ATTNRES_INPUT_CLAMP=N: hard-clamp the post-RMSNorm attn input
        # to [-N, N] before flashinfer_mla. flashinfer_mla NaNs on
        # high-magnitude bf16 inputs at deep blocks (RTX 5090 SM 12.0
        # has no working alternative MLA backend). Clamping matches
        # what RMSNorm DID succeed at producing in earlier blocks
        # (max ~5-50) and prevents the deepest layer from triggering
        # the flashinfer_mla internal overflow.
        if _os.environ.get("ATTNRES_FP32_NORM", "0") == "1":
            saved_dtype = attn_input.dtype
            x = attn_input.to(torch.float32)
            ln_w = self.input_layernorm.weight.to(torch.float32)
            ln_eps = getattr(self.input_layernorm, "variance_epsilon", 1e-6)
            rms = (x * x).mean(dim=-1, keepdim=True).add(ln_eps).rsqrt()
            attn_in = (x * rms * ln_w).to(saved_dtype)
        else:
            attn_in = self.input_layernorm(attn_input)

        # ATTNRES_MLA_FP32_FALLBACK=1: bypass flashinfer_mla on MLA layers
        # **during EXTEND/prefill only**. Block AttnRes residuals grow to
        # max~77 by the deepest MLA layer; flashinfer_mla's bf16 internals
        # overflow to NaN at prefill on Blackwell (RTX 5090 SM 12.0).
        # The fp32 path writes correct ``kv_a + k_pe`` to SGLang's MLA KV
        # buffer in the format flashinfer expects, so subsequent decode
        # steps fetch good cached K/V and run native bf16 flashinfer_mla
        # without NaN (per-step decode input is 1 token; bf16 kernels
        # don't fail on the small magnitudes there).
        #
        # No host-CPU sync (.item()) — cuda graph capture stays valid
        # because forward_mode is known at trace time.
        if (
            _os.environ.get("ATTNRES_MLA_FP32_FALLBACK", "0") == "1"
            and self._is_mla_self_attn()
            and forward_batch.forward_mode.is_extend()
        ):
            return self._mla_forward_fp32(
                attn_in, positions, forward_batch
            )

        attn_out = self.self_attn(
            hidden_states=attn_in,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        # When seq-shard is *enabled* but this forward call falls back to
        # replicated (e.g. decode batch=1, num_tokens not divisible by TP),
        # ``o_proj.reduce_results`` was permanently set to False at
        # __init__, so the returned tensor is a TP-partial sum. We must
        # all-reduce it ourselves to match the standard replicated
        # output, otherwise downstream layers see a 1/P-magnitude attn
        # contribution and the model silently drifts (softmax-invariant
        # to scaling so generations look plausible but are wrong).
        if _SEQ_SHARD_ENABLED:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size,
                tensor_model_parallel_all_reduce,
            )
            if get_tensor_model_parallel_world_size() > 1:
                attn_out = tensor_model_parallel_all_reduce(attn_out)
        return attn_out

    def _run_mlp(
        self,
        mlp_input: torch.Tensor,
        seq_shard: bool = False,
    ) -> torch.Tensor:
        if seq_shard:
            mlp_in_shard = self.post_attention_layernorm(mlp_input)
            mlp_in_replicated = all_gather_seq(mlp_in_shard)
            # MoE layers: KimiMoE has a hardcoded all-reduce in its forward
            # (no ``reduce_results`` flag). Use ``_kimi_moe_partial_sum`` to
            # call the underlying gate / topk / experts modules in the same
            # order as upstream KimiMoE.forward, but skip the trailing AR.
            # Dense KimiMLP layer has ``down_proj.reduce_results=False`` set
            # at __init__, so its forward already returns the partial sum.
            if _is_kimi_moe(self.mlp):
                mlp_partial = _kimi_moe_partial_sum(self.mlp, mlp_in_replicated)
            else:
                mlp_partial = self.mlp(mlp_in_replicated)
            return reduce_scatter_seq(mlp_partial)

        ffn_in = self.post_attention_layernorm(mlp_input)
        # Same correctness fix as _run_attn: when seq-shard env is set but
        # this particular forward fell back to replicated, the dense
        # ``down_proj.reduce_results=False`` (set at __init__) means MLP
        # returns a partial sum. KimiMoE has a hardcoded AR in its forward
        # which DOES still fire under seq_shard=False (because we don't
        # invoke ``_kimi_moe_partial_sum``), so we only need to AR-fix the
        # dense path.
        if _is_kimi_moe(self.mlp):
            return self.mlp(ffn_in)  # KimiMoE.forward did its own AR
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

    def _seq_shard_active(self, partial_block: torch.Tensor) -> bool:
        """Decide whether seq-dim sharding can be safely used this forward.

        Three conditions must hold:
        1. ``SGLANG_ATTN_RES_SEQ_SHARD=1`` env var was set at module load.
        2. TP world size > 1 (else nothing to shard across).
        3. ``num_tokens % tp_size == 0`` (decode with batch size 1 has
           ``num_tokens=1`` which is not divisible — fall back to replicated).

        Logs a once-per-instance warning when the user requested seq-shard
        but condition 3 forced the fallback, so they aren't surprised by
        silent perf degradation.
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
                    "forward; falling back to replicated mode for this call. "
                    "This typically happens for decode steps where "
                    "num_tokens < TP. Prefill chunks aligned to a TP-multiple "
                    "do exercise the seq-shard path.",
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
        zero_allocator: BumpAllocator,
        seq_shard: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Two-phase optimized forward over a full block of layers.

        Block-level commit semantics:
        * Pre-attn aggregation of *layer 0* of this block uses the OLD
          committed list (length B-1) and the previous block's final
          partial. This is the "block-boundary first aggregation".
        * Then commit happens (committed grows to length B; partial = None).
        * The remaining 2·L_block - 1 aggregations (layer 0's pre-FFN
          and all layer 1+ pre-attn / pre-FFN) all use the NEW committed
          list (length B) — these can share one Phase 1 pass.
        """
        L0 = layers_in_block[0]

        # ---- Aggregation 1/2L_block: layer 0's pre-attn (OLD committed) ----
        if committed_blocks:
            cache0 = block_attn_res_phase1(
                committed_blocks,
                [_query_from_proj(L0.attn_res_proj)],
                [L0.attn_res_norm],
            )[0]
        else:
            cache0 = None  # Block 0 entry: identity on partial_block.
        h = block_attn_res_phase2_merge(
            cache0,
            partial_block,
            _query_from_proj(L0.attn_res_proj),
            L0.attn_res_norm,
        )

        # ---- Commit at block start ----
        committed_blocks = committed_blocks + [partial_block]
        # partial_block becomes self_attn(input_layernorm(h)) below.

        # ---- Phase 1 batched against NEW committed list, for the
        #      2·L_block - 1 remaining queries within this block ----
        rest_queries: list[torch.Tensor] = []
        rest_norms: list = []
        rest_queries.append(_query_from_proj(L0.mlp_res_proj))
        rest_norms.append(L0.mlp_res_norm)
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

        # Numerical-stability workaround for AttnRes residual stream
        # accumulating to NaN at deep blocks. Two complementary knobs:
        #   ATTNRES_BF16_ACCUM=1  → keep residual in bf16 (original)
        #   default                → accumulate residual in fp32
        #   ATTNRES_CLIP=N        → clamp partial_block to [-N, N] before
        #                            each attn/mlp call; default 0 = off.
        _fp32_accum = _os.environ.get("ATTNRES_BF16_ACCUM", "0") != "1"
        _input_dtype = partial_block.dtype
        if _fp32_accum:
            partial_block = partial_block.to(torch.float32)
        try:
            _clip = float(_os.environ.get("ATTNRES_CLIP", "0"))
        except Exception:
            _clip = 0.0
        def _maybe_clip(x):
            if _clip <= 0:
                return x
            return x.clamp(min=-_clip, max=_clip)

        _trace_fine = _os.environ.get("ATTNRES_NAN_TRACE", "0") == "1"
        def _stats_pb(label):
            if not _trace_fine:
                return
            t = partial_block
            try:
                nan = bool(t.isnan().any().item())
                inf = bool(t.isinf().any().item())
                am = float(t.detach().float().abs().mean().item())
                amx = float(t.detach().float().abs().max().item())
                _logger.warning(
                    f"[NaN-FINE] {label}: abs_mean={am:.4f} "
                    f"max={amx:.4f} nan={nan} inf={inf}"
                )
            except Exception as e:
                _logger.warning(f"[NaN-FINE] {label}: {e}")

        # ---- Layer 0 attn + post-attn aggregation ----
        attn_out = L0._run_attn(
            h, positions, forward_batch, zero_allocator, seq_shard=seq_shard,
        )
        partial_block = attn_out
        _stats_pb(f"L{layers_in_block[0].layer_id if hasattr(layers_in_block[0], "layer_id") else 0}_after_attn")

        h = block_attn_res_phase2_merge(
            rest_cache[cache_idx], partial_block,
            rest_queries[cache_idx], rest_norms[cache_idx],
        )
        cache_idx += 1
        ffn_out = L0._run_mlp(h, seq_shard=seq_shard)
        partial_block = partial_block + ffn_out
        _stats_pb(f"L{layers_in_block[0].layer_id if hasattr(layers_in_block[0], "layer_id") else 0}_after_mlp")

        # ---- Layers 1..L_block-1 ----
        for L in layers_in_block[1:]:
            # Pre-attn
            h = block_attn_res_phase2_merge(
                rest_cache[cache_idx], _maybe_clip(partial_block).to(_input_dtype),
                rest_queries[cache_idx], rest_norms[cache_idx],
            )
            cache_idx += 1
            attn_out = L._run_attn(
                h, positions, forward_batch, zero_allocator, seq_shard=seq_shard,
            )
            partial_block = partial_block + attn_out.to(partial_block.dtype)
            _stats_pb(f"L{L.layer_id if hasattr(L, "layer_id") else "?"}_after_attn")
            # Pre-FFN
            h = block_attn_res_phase2_merge(
                rest_cache[cache_idx], partial_block,
                rest_queries[cache_idx], rest_norms[cache_idx],
            )
            cache_idx += 1
            ffn_out = L._run_mlp(h, seq_shard=seq_shard)
            partial_block = partial_block + ffn_out
            _stats_pb(f"L{L.layer_id if hasattr(L, "layer_id") else "?"}_after_mlp")

        # Cast back to model dtype before returning so downstream code
        # (committed_blocks scatter/stack, next block's RMSNorm, etc.)
        # sees the original dtype.
        if _fp32_accum:
            partial_block = partial_block.to(_input_dtype)

        return committed_blocks, partial_block

    def _naive_per_layer_step(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer,
        is_block_start: bool,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Fallback path when this PP stage isn't block-aligned."""
        h = block_attn_res(blocks, partial_block, layer.attn_res_proj, layer.attn_res_norm)
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None
        attn_out = layer._run_attn(h, positions, forward_batch, zero_allocator)
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
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # NaN tracing: set ATTNRES_NAN_TRACE=1 to log per-layer NaN.
        _trace = _os.environ.get("ATTNRES_NAN_TRACE", "0") == "1"
        def _stats(t, name):
            if not _trace or t is None:
                return
            try:
                nan = bool(t.isnan().any().item())
                inf = bool(t.isinf().any().item())
                am = float(t.detach().float().abs().mean().item())
                amx = float(t.detach().float().abs().max().item())
                _logger.warning(
                    f"[NaN-TRACE] {name}: shape={tuple(t.shape)} "
                    f"dtype={t.dtype} abs_mean={am:.4f} max={amx:.4f} "
                    f"nan={nan} inf={inf}"
                )
            except Exception as e:
                _logger.warning(f"[NaN-TRACE] {name}: {e}")

        # First-stage init.
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                h = inputs_embeds
                _stats(h, "input_embeds")
            else:
                h = self.embed_tokens(input_ids)
                _stats(h, "embed_tokens")
            block_list: list[torch.Tensor] = []
            partial_block = h
        else:
            assert pp_proxy_tensors is not None
            partial_block = pp_proxy_tensors["hidden_states"]
            # PPProxyTensors implements ``__getitem__`` but no ``.get()``;
            # peek via the underlying ``.tensors`` dict.
            blocks_t = pp_proxy_tensors.tensors.get("blocks")
            block_list = (
                _unstack_blocks(blocks_t)
                if (blocks_t is not None and blocks_t.shape[0] > 0)
                else []
            )

        zero_allocator = BumpAllocator(
            buffer_size=1024,
            dtype=torch.float32,
            device=partial_block.device,
        )

        # Iterate this stage's slice in ``layers_per_block``-sized chunks.
        # Within each chunk run ONE Phase 1 against the (constant-within-block)
        # committed list, then per-layer Phase 2 merges. This is the
        # algorithmic IO saving from the Zhihu blog
        # (https://zhuanlan.zhihu.com/p/2017528295286133070):
        # naive  = O(2·L_block · N · T · D) committed-block reads per block
        # 2-phase = O(N · T · D + 2·L_block · T · D) per block
        # (factor ≈ L_block reduction at large N)
        L_block = self.layers_per_block
        layer_indices = list(range(self.start_layer, self.end_layer))

        # PP slicing assumption: this stage's layer range is block-aligned
        # (i.e. ``start_layer`` is a block boundary and ``end_layer - start_layer``
        # is a multiple of L_block). Holds for our num_blocks=4 configs at
        # PP up to 4. If a future config violates this, fall back to per-layer
        # naive aggregation for the partial-block on this stage.
        block_aligned = (
            len(layer_indices) > 0
            and layer_indices[0] % L_block == 0
            and len(layer_indices) % L_block == 0
        )

        # Decide seq-shard layout for this forward call. If active, the
        # ENTRY partial_block needs to be split along seq dim to (T/P, D);
        # the committed_blocks coming through pp_proxy_tensors are
        # likewise sharded if the previous stage was in seq-shard mode.
        seq_shard_active = self._seq_shard_active(partial_block)

        if seq_shard_active and self.pp_group.is_first_rank:
            # First PP rank: input came from embed_tokens which is replicated.
            # Split → shard. (Subsequent stages receive already-sharded
            # tensors via pp_proxy_tensors, so no split needed.)
            partial_block = split_seq(partial_block)

        # Bench-only: SGLANG_ATTN_RES_BYPASS=1 disables AttnRes entirely
        # (skip every aggregation, never commit blocks) and runs vanilla
        # PreNorm. Gives a directly-comparable "no-AttnRes baseline" that
        # uses this same model class + same ckpt + same env-compat patches
        # — so any latency delta vs the AttnRes-active path isolates the
        # cost of the AttnRes algorithm itself. The AttnRes-specific
        # parameters (attn_res_proj/norm/...) sit unused in memory but
        # don't impact step time. Not for end users.
        if _BYPASS_ATTN_RES:
            if seq_shard_active:
                partial_block = all_gather_seq(partial_block)
                seq_shard_active = False
            for global_idx in layer_indices:
                layer = self.layers[global_idx]
                attn_out = layer._run_attn(
                    partial_block, positions, forward_batch,
                    zero_allocator, seq_shard=False,
                )
                partial_block = partial_block + attn_out
                ffn_out = layer._run_mlp(partial_block, seq_shard=False)
                partial_block = partial_block + ffn_out
        elif _FORCE_NAIVE_PATH or not block_aligned:
            # Bench-only naive path OR fallback when this PP stage isn't
            # block-aligned. Naive path doesn't currently support seq-shard
            # — gather first.
            if seq_shard_active:
                partial_block = all_gather_seq(partial_block)
                block_list = [all_gather_seq(b) for b in block_list]
                seq_shard_active = False
            for global_idx in layer_indices:
                layer = self.layers[global_idx]
                is_block_start = (global_idx % L_block == 0)
                block_list, partial_block = self._naive_per_layer_step(
                    block_list, partial_block, layer, is_block_start,
                    positions, forward_batch, zero_allocator,
                )
        else:
            for chunk_start in range(0, len(layer_indices), L_block):
                chunk = layer_indices[chunk_start : chunk_start + L_block]
                layers_in_block = [self.layers[i] for i in chunk]
                block_list, partial_block = self._forward_one_block(
                    block_list,
                    partial_block,
                    layers_in_block,
                    positions,
                    forward_batch,
                    zero_allocator,
                    seq_shard=seq_shard_active,
                )
                _stats(partial_block, f"after_block_chunk_{chunk_start}")

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
        # Both ``block_attn_res`` (per-token softmax-aggregator) and
        # ``self.norm`` (per-token RMSNorm) commute with seq-dim sharding,
        # so they run correctly on (T/P, D) shards. The final ``all_gather``
        # restores (T, D) replicated for the lm_head.
        h_final = block_attn_res(
            block_list, partial_block,
            self.final_attn_res_proj, self.final_attn_res_norm,
        )
        h_normed = self.norm(h_final)
        if seq_shard_active:
            h_normed = all_gather_seq(h_normed)
        return h_normed


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
        # AttnRes overlay does not currently support DP attention. The
        # ``_run_attn`` / ``_run_mlp`` helpers bypass
        # ``LayerCommunicator.prepare_attn`` (because AttnRes replaces the
        # PreNorm fused-residual-add path entirely) which is also where DP
        # attention scatter routing lives. Adding DP-attn support requires
        # threading the DP scatter mode manually around our ``self_attn``
        # call. Until that's done, refuse rather than silently produce
        # wrong results.
        try:
            from sglang.srt.layers.dp_attention import is_dp_attention_enabled
            if is_dp_attention_enabled():
                raise NotImplementedError(
                    "Block AttnRes overlay does not yet support DP attention. "
                    "Disable via --disable-dp-attention (or omit --enable-dp-attention) "
                    "and re-launch. Tracking issue: phase 11 follow-up."
                )
        except ImportError:
            pass  # Older sglang without is_dp_attention_enabled — assume off.

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

        # Patch fp32-required params that upstream constructs with the
        # default torch dtype (bf16 under the engine's load context).
        # See ``_force_fp32_required_params`` for context.
        self._force_fp32_required_params()

    def load_weights(self, weights):
        """Filter PP-missing weights before delegating to upstream.

        Under PP > 1, this rank only owns ``layers[start_layer:end_layer)``
        and a subset of the top-level params (``embed_tokens`` only on
        the first PP rank; ``norm``, ``lm_head``, ``final_attn_res_*``
        only on the last). Upstream ``KimiLinearForCausalLM.load_weights``
        looks up every weight name in ``params_dict`` and raises
        ``KeyError`` for the missing ones — the
        ``is_pp_missing_parameter`` helper in upstream is incomplete.

        We pre-filter the weights iterator so missing-on-this-rank
        entries are silently dropped, keeping upstream's load logic
        unchanged for everything else.
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
                # Top-level PP-stage filtering.
                if name in last_only_top and not is_last:
                    continue
                if name in first_only_top and not is_first:
                    continue
                # Layer-range filtering.
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

        # Upstream's load_weights post-loop iterates
        # ``self.config.full_attention_layer_ids`` to materialise the
        # MLA ``w_kc`` / ``w_vc`` cached projections per full-attn
        # layer. Out-of-range layers on this PP rank are
        # ``PPMissingLayer`` objects with no ``self_attn`` attribute,
        # so the loop AttributeErrors. Temporarily shadow the
        # config's ``full_attention_layer_ids`` property with an
        # in-range subset for the duration of the upstream call,
        # then restore.
        cfg_cls = type(self.config)
        in_range_full_attn = [
            i for i in self.config.full_attention_layer_ids
            if start <= i < end
        ]
        orig_prop = cfg_cls.__dict__.get("full_attention_layer_ids")
        cfg_cls.full_attention_layer_ids = property(
            lambda _self, _v=in_range_full_attn: _v
        )
        try:
            return super().load_weights(_filter())
        finally:
            if orig_prop is not None:
                cfg_cls.full_attention_layer_ids = orig_prop
            else:
                delattr(cfg_cls, "full_attention_layer_ids")

    def _force_fp32_required_params(self) -> None:
        """Apply post-construction instance-scoped patches to make this
        non-mainline overlay class run under upstream's stock kernels.

        Two scoped patches:

        1. ``e_score_correction_bias`` → fp32 (single Parameter mutation
           per MoE gate, ~26 params total at 16 layers; no global state
           change). Upstream ``KimiMoE.__init__``
           (sglang/srt/models/kimi_linear.py:91) sets
           ``self.gate.e_score_correction_bias = nn.Parameter(
           torch.empty(num_experts))`` with no dtype kwarg, picking
           default bf16 under the engine's load context. The downstream
           ``biased_grouped_topk_gpu`` path (layers/moe/topk.py:842)
           casts gating_output to fp32 then passes the bias unchanged;
           the sgl_kernel ``moe_fused_gate`` then crashes with
           "input and bias should have the same dtype". Mutate the
           Parameter ``.data`` in place so the existing
           ``TopK(correction_bias=...)`` reference automatically picks up
           the new dtype. Should ideally be a one-line upstream fix in
           ``KimiMoE.__init__``; until then this overlay carries it.

        2. Per-instance ``.contiguous()`` shim on each MLA layer's
           ``self_attn.kv_a_layernorm``. ``forward_absorb_prepare`` in
           ``deepseek_common`` splits ``compressed_kv`` along dim -1
           into ``(k_nope, k_pe)`` producing non-contig views. The
           sgl_kernel rmsnorm requires ``stride[0]`` aligned to 8
           elements; for some kv_lora_rank values (notably the original
           436M's 584 + 36 = 620) the resulting stride isn't, and the
           kernel ``ValueError("Invalid mX.strides[0]...")``. Aligned
           variants (kv_lora_rank multiple of 64) don't trigger this,
           so on the canonical phase 11 ckpt (kv_lora=512) this shim is
           a no-op. Scoped to ``kv_a_layernorm`` instances **owned by
           this model only** — does NOT mutate the upstream ``RMSNorm``
           class itself, so other RMSNorm uses in the engine are
           unaffected.
        """
        # Patch 1 — fp32 e_score_correction_bias on every MoE gate.
        for module in self.modules():
            gate = getattr(module, "gate", None)
            if gate is None:
                continue
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is None or bias.dtype == torch.float32:
                continue
            bias.data = bias.data.to(torch.float32)

        # Patch 2 — instance-scoped .contiguous() shim on each MLA's
        # kv_a_layernorm only. Avoids the previous version's class-level
        # mutation of ``RMSNorm.forward_cuda`` which had been bleeding
        # into every RMSNorm in the engine (input_layernorm,
        # post_attention_layernorm, k_norm, q_norm, ...) — most of those
        # don't see non-contig input and the wrap was wasted work.
        for module in self.modules():
            attn = getattr(module, "self_attn", None)
            if attn is None:
                continue
            kva = getattr(attn, "kv_a_layernorm", None)
            if kva is None:
                continue
            # If this layernorm has never been shimmed, wrap its forward.
            if getattr(kva, "_attnres_kva_contig_shim", False):
                continue
            orig_forward = kva.forward

            def _make_shim(orig):
                def shim(x, *a, **kw):
                    if not x.is_contiguous():
                        x = x.contiguous()
                    return orig(x, *a, **kw)
                return shim

            kva.forward = _make_shim(orig_forward)
            kva._attnres_kva_contig_shim = True


EntryClass = KimiBlockAttnResForCausalLM

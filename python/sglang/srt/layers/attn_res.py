# Copyright (c) Anthropic and contributors.
# Licensed under the Apache License, Version 2.0 (see SGLang LICENSE).

"""Block Attention Residual algorithm — model-agnostic, two-phase optimized.

Block AttnRes (Kimi paper § 5, Fig. 2) is a residual-stream overlay:
each layer's pre-attn and pre-FFN inputs are produced by attention-
aggregating over a list of prior committed-block representations plus
the current partial block. The construct is in the same family as
ByteDance's Hyper-Connections (arxiv 2409.19606) and DeepSeek's mHC
(arxiv 2512.24880) — multi-stream residual variants.

This module hosts:

* :func:`block_attn_res` — the **naive** per-layer aggregator. Every
  layer re-reads every committed block. Kept as the numerical
  reference and as a torch-only fallback path. **Not used at inference**.

* :func:`block_attn_res_phase1` — the **batched committed-side**
  aggregator. Run once at block boundary against all per-layer
  pseudo-queries inside the block. Outputs ``(committed_part, lse)``
  per query. Amortises the IO cost from ``O(2L_block · N · T · D)``
  reads of committed blocks down to ``O(N · T · D + 2L_block · T · D)``.

* :func:`block_attn_res_phase2_merge` — the **per-layer online-softmax
  merge** between ``(committed_part, lse)`` from Phase 1 and the
  current ``partial_block``. Mathematically exact via max-stable LSE
  merge (same trick as Flash Attention's split-K).

Sibling of ``sglang.srt.layers.mhc`` (DeepSeek-V4 mHC, on the upstream
``deepseek_v4`` branch). Both expose pure-functional algorithm
modules used by per-model overlays under ``models/<model>_attn_res*``.

Lineage:
* Hyper-Connections (ByteDance, arxiv 2409.19606)
* Block AttnRes (Kimi, arxiv 2603.15031)
* mHC (DeepSeek 2026, arxiv 2512.24880)

Two-phase reference: https://zhuanlan.zhihu.com/p/2017528295286133070
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query_from_proj(proj: nn.Linear) -> torch.Tensor:
    """Extract the per-layer pseudo-query ``w_l`` from a ``[1, D]`` linear.

    ``proj.weight`` IS the pseudo-query. The optional ``.to_local()``
    handles DTensor-wrapped weights from FSDP/TP.
    """
    weight = proj.weight  # [1, D]
    if hasattr(weight, "to_local"):
        weight = weight.to_local()
    return weight.squeeze(0)  # [D]


def zero_init_pseudo_query(linear: nn.Linear) -> None:
    """Paper § 5: zero-init pseudo-queries so initial softmax is uniform.

    With ``w_l = 0``, the first forward pass has uniform softmax over
    the block list — Block AttnRes degenerates to a vanilla mean over
    blocks plus partial. The model gradually learns specialised
    aggregation; without zero-init, early-step gradients are dominated
    by the noise of randomly-init'd pseudo-queries.
    """
    with torch.no_grad():
        linear.weight.zero_()


# ---------------------------------------------------------------------------
# Naive single-pass aggregator (numerical reference; not used at inference)
# ---------------------------------------------------------------------------


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Naive per-layer aggregator (Kimi paper, Fig. 2). Reference impl.

    Stacks ``blocks + [partial_block]`` to a single ``(N+1, *, D)``
    tensor, applies one RMSNorm, computes softmax(q · norm(V)) over
    the leading axis, and weight-sums V back. Mathematically equivalent
    to :func:`block_attn_res_phase1` + :func:`block_attn_res_phase2_merge`,
    just without the IO amortisation.

    Used by:
    * Numerical-equivalence regression test (assert two-phase ≡ naive).
    * Torch-only fallback path when tilelang/deepgemm kernels are
      unavailable.
    * Training-time aggregator (kept simple — paper says marginal
      training overhead, no need to two-phase the training path).
    """
    V = torch.stack(blocks + [partial_block], dim=0)   # (N+1, *, D)
    # SGLang's sgl_kernel rmsnorm is 2D-only (M, D). Flatten leading
    # dims for the norm call (lets cuda-graph capture route the kernel
    # cleanly), then restore shape for the einsum / weighted sum.
    leading_shape = V.shape[:-1]
    K = norm(V.reshape(-1, V.shape[-1])).reshape(*leading_shape, V.shape[-1])
    query = _query_from_proj(proj)                     # (D,)
    logits = torch.einsum("d, n...d -> n...", query, K)
    weights = F.softmax(logits, dim=0)
    return torch.einsum("n..., n...d -> ...d", weights, V)


# ---------------------------------------------------------------------------
# Two-phase optimized inference path
# ---------------------------------------------------------------------------


def block_attn_res_phase1(
    committed_blocks: list[torch.Tensor],
    queries: list[torch.Tensor],
    norms: list[nn.Module],
) -> list[Optional[tuple[torch.Tensor, torch.Tensor]]]:
    """Phase 1 — batched committed-side attention, run ONCE per block.

    For every (query, norm) pair belonging to a layer in the current
    block, compute the partial attention against ``committed_blocks``
    and the corresponding LSE. The committed list is constant within
    a block, so this batched pass amortises ``N × T × D`` block-list
    reads to **once per block** rather than ``2 × L_block`` times.

    Args:
        committed_blocks: list of N prior block representations,
            each shape ``(*, D)``. May be empty (block 0 entry).
        queries: list of pseudo-query vectors for layers in this
            block, each shape ``(D,)``. Length ``2 × L_block`` —
            both pre-attn and pre-FFN queries are passed here so
            they share the single Phase 1 pass.
        norms: list of RMSNorm modules paired with ``queries`` —
            each layer's ``attn_res_norm`` and ``mlp_res_norm``
            share the same V (committed_blocks) but apply distinct
            gamma scaling.

    Returns:
        list of length ``len(queries)``. Each element is either
        ``None`` (when ``committed_blocks`` is empty — caller must
        short-circuit Phase 2 to return ``partial_block`` directly)
        or a tuple ``(committed_part, lse)``:
        * ``committed_part`` — partial attention over committed blocks,
          shape ``(*, D)``.
        * ``lse`` — log-sum-exp of (q · norm(V)) along the block axis,
          shape ``(*,)``. Encodes both max-shift and Z normaliser for
          numerically stable Phase 2 merge.
    """
    if not committed_blocks:
        return [None] * len(queries)

    assert len(queries) == len(norms), \
        f"queries ({len(queries)}) and norms ({len(norms)}) length mismatch"

    V = torch.stack(committed_blocks, dim=0)   # (N, *, D)
    Q = len(queries)

    # Try the vectorised path first: when all per-layer RMSNorms share the
    # same epsilon (true for Kimi/Qwen3 where every layernorm is constructed
    # with ``eps=config.rms_norm_eps``), we can compute ``rsqrt(V^2.mean)``
    # ONCE on the shared V, then fold each layer's gamma into a per-query
    # effective query ``q_eff = q * gamma``. The whole batch then collapses
    # to two batched einsums plus one max+softmax, replacing the per-query
    # RMSNorm + einsum loop. ~Q× speedup at the cost of one assertion.
    eps_set = set()
    for n in norms:
        e = getattr(n, "variance_epsilon", None)
        if e is None:
            e = getattr(n, "eps", None)
        eps_set.add(e)

    can_vectorise = (
        len(eps_set) == 1
        and None not in eps_set
        and all(hasattr(n, "weight") for n in norms)
    )

    if can_vectorise:
        eps = next(iter(eps_set))
        # Stack gammas (Q, D) and queries (Q, D); fold gamma into query.
        gammas = torch.stack(
            [n.weight.to_local() if hasattr(n.weight, "to_local") else n.weight
             for n in norms],
            dim=0,
        )  # (Q, D)
        queries_t = torch.stack(queries, dim=0).to(gammas.dtype)  # (Q, D)
        q_eff = queries_t * gammas  # (Q, D)

        # Shared rsqrt over V (fp32 for stability — matches sgl_kernel rmsnorm).
        V_f32 = V.to(torch.float32)
        rsqrt = torch.rsqrt(V_f32.pow(2).mean(-1, keepdim=True) + eps)
        V_normed = (V_f32 * rsqrt).to(V.dtype)  # (N, *, D)

        # Batched score: logits[q, n, t...] = sum_d q_eff[q, d] * V_normed[n, t..., d]
        logits = torch.einsum("qd, n...d -> qn...", q_eff, V_normed)  # (Q, N, *)

        # Numerically-stable softmax over the N (block) axis (dim=1 of (Q,N,*)).
        m = logits.max(dim=1, keepdim=True).values                     # (Q, 1, *)
        exp_l = torch.exp(logits - m)                                  # (Q, N, *)
        Z = exp_l.sum(dim=1)                                           # (Q, *)
        # Weighted V sum:  out[q, t..., d] = sum_n exp_l[q, n, t...] * V[n, t..., d] / Z[q, t...]
        committed_parts = (
            torch.einsum("qn..., n...d -> q...d", exp_l, V)
            / Z.unsqueeze(-1)
        )  # (Q, *, D)
        lses = m.squeeze(1) + torch.log(Z)  # (Q, *)

        return [(committed_parts[i], lses[i]) for i in range(Q)]

    # Fallback: per-query loop. Hits when norms have heterogeneous eps,
    # which would happen if a downstream caller passes RMSNorms from
    # multiple model files with different defaults. Same math, slower.
    out: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
    leading_shape = V.shape[:-1]
    for q, norm in zip(queries, norms):
        # Flatten leading dims for sgl_kernel's 2D-only rmsnorm.
        K = norm(V.reshape(-1, V.shape[-1])).reshape(
            *leading_shape, V.shape[-1]
        )                                                            # (N, *, D)
        logits = torch.einsum("d, n...d -> n...", q, K)              # (N, *)
        m = logits.max(dim=0).values                                 # (*,)
        exp_l = torch.exp(logits - m.unsqueeze(0))                   # (N, *)
        Z = exp_l.sum(dim=0)                                         # (*,)
        committed_part = (
            torch.einsum("n..., n...d -> ...d", exp_l, V) / Z.unsqueeze(-1)
        )                                                            # (*, D)
        lse = m + torch.log(Z)                                       # (*,)
        out.append((committed_part, lse))
    return out


def block_attn_res_phase2_merge(
    committed_cache: Optional[tuple[torch.Tensor, torch.Tensor]],
    partial_block: torch.Tensor,
    query: torch.Tensor,
    norm: nn.Module,
) -> torch.Tensor:
    """Phase 2 — online-softmax merge of committed-side (Phase 1) with partial_block.

    Mathematically equivalent to the naive softmax over the full
    ``committed + [partial]`` stack — same as :func:`block_attn_res`
    — but only **one extra** RMSNorm + dot-product is computed per
    layer (vs N+1 in the naive impl). The committed-side score (LSE)
    has already been computed once in Phase 1.

    Args:
        committed_cache: ``(committed_part, lse)`` from
            :func:`block_attn_res_phase1`, or ``None`` when
            ``committed_blocks`` was empty (block 0 entry, where the
            aggregation degenerates to identity on partial_block).
        partial_block: current uncommitted block, shape ``(*, D)``.
        query: pseudo-query for THIS layer, shape ``(D,)``. Must
            correspond to the same index used in Phase 1.
        norm: RMSNorm module for THIS layer's aggregation. Same
            module used in Phase 1 — consistency required.

    Returns:
        Aggregated state, shape ``(*, D)``. Numerically equivalent
        (within fp16 epsilon) to :func:`block_attn_res` over
        ``committed + [partial_block]``.
    """
    # Block 0 entry: no committed blocks — softmax over the singleton
    # {partial_block} is identity on partial_block.
    if committed_cache is None:
        return partial_block

    committed_part, lse_committed = committed_cache

    # Compute partial-side logit only — single (1, *) score vs naive
    # (N+1, *) score in :func:`block_attn_res`. Flatten leading dims
    # for sgl_kernel's 2D-only rmsnorm so cuda-graph capture cleanly
    # routes the kernel.
    leading_shape = partial_block.shape[:-1]
    K_partial = norm(
        partial_block.reshape(-1, partial_block.shape[-1])
    ).reshape(*leading_shape, partial_block.shape[-1])               # (*, D)
    logit_partial = torch.einsum("d, ...d -> ...", query, K_partial)  # (*,)

    # Max-stable online softmax merge.
    m_new = torch.maximum(lse_committed, logit_partial)              # (*,)
    w_committed = torch.exp(lse_committed - m_new)                   # (*,)
    w_partial = torch.exp(logit_partial - m_new)                     # (*,)
    denom = w_committed + w_partial                                  # (*,)
    out = (
        w_committed.unsqueeze(-1) * committed_part
        + w_partial.unsqueeze(-1) * partial_block
    ) / denom.unsqueeze(-1)                                          # (*, D)
    return out


# ---------------------------------------------------------------------------
# Sequence-dim TP shard helpers
# ---------------------------------------------------------------------------
# These wrap ``torch.distributed`` reduce_scatter / all_gather for
# concatenated tensors along the leading (sequence) dim, with safe
# no-op behaviour at TP=1 so call sites can use the same code path
# regardless of TP size.
#
# Memory model under seq-shard:
#   committed_blocks per rank: list of N tensors of shape (T/P, D)
#   partial_block per rank:    shape (T/P, D)
#   Phase 1 cache per (q, norm): (committed_part of (T/P, D), lse of (T/P,))
#   Phase 2 merge output:      (T/P, D)
#
# Communication per AttnRes layer (under TP=P, seq-shard ON):
#   1. all_gather seq-shard h after Phase 2 merge → (T, D) before attn input
#   2. self_attn / o_proj called with reduce_results=False → returns partial
#      sum (T, D) — NOT all-reduced
#   3. reduce_scatter the partial sum → (T/P, D) per rank
#   4. add to partial_block (sharded)
#   5. same for the FFN side: AG → mlp → RS → add
# Net comm per layer = 2 × (RS + AG) = 2 × AR-equivalent — IDENTICAL to
# the standard SGLang path (which is 2 AR per layer), with the bonus
# that all in-between compute (Phase2 merge + RMSNorm + AttnRes
# aggregation) operates on (T/P, D) shards.
# This is the integration described in the Zhihu blog
# (https://zhuanlan.zhihu.com/p/2017528295286133070).


def all_gather_seq(x: torch.Tensor) -> torch.Tensor:
    """All-gather a (T/P, D) shard along dim 0 → (T, D) replicated.

    No-op when TP=1. Uses ``torch.distributed.all_gather_into_tensor``
    via SGLang's TP group.
    """
    from sglang.srt.distributed import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    out = torch.empty(
        (x.shape[0] * tp_size, *x.shape[1:]), dtype=x.dtype, device=x.device,
    )
    torch.distributed.all_gather_into_tensor(
        out, x.contiguous(), group=get_tp_group().device_group,
    )
    return out


def reduce_scatter_seq(x: torch.Tensor) -> torch.Tensor:
    """Reduce-scatter a (T, D) partial sum along dim 0 → (T/P, D) reduced.

    Replaces the standard ``tensor_model_parallel_all_reduce`` when the
    caller wants the result sharded along sequence rather than
    replicated. Comm cost = AR / 2 (only one direction of the ring).

    No-op when TP=1.
    """
    from sglang.srt.distributed import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    assert x.shape[0] % tp_size == 0, (
        f"reduce_scatter_seq requires dim 0 ({x.shape[0]}) divisible by "
        f"TP ({tp_size})"
    )
    out = torch.empty(
        (x.shape[0] // tp_size, *x.shape[1:]), dtype=x.dtype, device=x.device,
    )
    torch.distributed.reduce_scatter_tensor(
        out, x.contiguous(), group=get_tp_group().device_group,
    )
    return out


def split_seq(x: torch.Tensor) -> torch.Tensor:
    """Local split: take this TP rank's slice along dim 0.

    Used as a "soft fallback" when an upstream module produces
    replicated output (e.g. KimiMoE which does its own all-reduce
    internally and we can't easily skip it). Cheaper than RS but
    requires the data to already be replicated.

    No-op when TP=1.
    """
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return x
    assert x.shape[0] % tp_size == 0, (
        f"split_seq requires dim 0 ({x.shape[0]}) divisible by TP ({tp_size})"
    )
    rank = get_tensor_model_parallel_rank()
    chunk = x.shape[0] // tp_size
    return x.narrow(0, rank * chunk, chunk).contiguous()


# ---------------------------------------------------------------------------
# Numerical-equivalence test (importable as a smoke check)
# ---------------------------------------------------------------------------


@torch.no_grad()
def assert_two_phase_equivalent(
    *,
    num_blocks: int = 3,
    num_tokens: int = 16,
    hidden_dim: int = 64,
    num_layers_in_block: int = 2,
    rtol: float = 5e-3,
    atol: float = 5e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Verify two-phase output matches naive within (rtol, atol).

    Ran from `phase11/test_two_phase_numerical.py`; also importable
    so callers can validate after kernel changes.

    Uses an inline pure-torch RMSNorm (independent of sglang's CUDA-only
    ``layers.layernorm.RMSNorm``) so the test runs on both CPU and CUDA
    without environment-specific kernels.
    """
    class _PureTorchRMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps

        def forward(self, x):
            x32 = x.to(torch.float32)
            rms = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
            return (x32 / rms).to(x.dtype) * self.weight

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(0)

    # Random committed blocks and partial.
    committed = [
        torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
        for _ in range(num_blocks)
    ]
    partial = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)

    # 2 * num_layers_in_block queries (pre-attn + pre-FFN per layer).
    queries = [
        torch.randn(hidden_dim, device=device, dtype=dtype)
        for _ in range(2 * num_layers_in_block)
    ]
    norms = [
        _PureTorchRMSNorm(hidden_dim).to(device=device, dtype=dtype)
        for _ in range(2 * num_layers_in_block)
    ]
    # Randomise norm gamma so each layer's K is genuinely different.
    with torch.no_grad():
        for n in norms:
            n.weight.normal_(mean=1.0, std=0.05)

    # Phase 1: cache committed-side stats.
    cache = block_attn_res_phase1(committed, queries, norms)

    # For each query, compare:
    # * naive: softmax over (committed + [partial])
    # * two-phase: phase2_merge(cache_q, partial, q, norm)
    for i, (q, n) in enumerate(zip(queries, norms)):
        proj = nn.Linear(hidden_dim, 1, bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            proj.weight.copy_(q.unsqueeze(0))

        naive = block_attn_res(committed, partial, proj, n)
        two_phase = block_attn_res_phase2_merge(cache[i], partial, q, n)

        assert torch.allclose(naive, two_phase, rtol=rtol, atol=atol), (
            f"two-phase divergence on query {i}: "
            f"max diff = {(naive - two_phase).abs().max().item():.2e}"
        )

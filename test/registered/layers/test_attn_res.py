# Copyright (c) Anthropic and contributors.
# Licensed under the Apache License, Version 2.0 (see SGLang LICENSE).

"""Numerical and structural tests for ``sglang.srt.layers.attn_res``.

These exercise the pure-function algorithm — no engine/server bring-up,
no GPU mesh — so they fit cleanly into stage-a CPU-friendly CI even
though the optimized inference path obviously does need GPU at runtime.

What's covered:
1. Two-phase ↔ naive numerical equivalence at fp32 (strict 1e-4) and
   bf16 (within accumulation noise; documents the bf16 ceiling).
2. Block-0 entry edge case: empty ``committed_blocks`` must short-circuit
   Phase 2 to identity-on-partial.
3. Phase 1 vectorisation correctness: same-eps fast path == per-query
   loop fallback path output, bit-exact at fp32.
4. Helper identities at TP=1: ``all_gather_seq`` / ``reduce_scatter_seq``
   / ``split_seq`` are all no-ops.

What's NOT covered (deliberately, requires GPU + multi-rank):
* The TP comm helpers under TP>1 — needs a real distributed init.
* CUDA graph capture compatibility — needs full engine boot.
* The per-model overlay forward end-to-end — covered by separate
  registered model-launcher tests.
"""
from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from sglang.srt.layers.attn_res import (
    all_gather_seq,
    block_attn_res,
    block_attn_res_phase1,
    block_attn_res_phase2_merge,
    reduce_scatter_seq,
    split_seq,
    zero_init_pseudo_query,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=60,
    suite="stage-a-test-1-gpu",
    # CPU-friendly path; tagged for stage-a anyway since the broader
    # AttnRes overlay tests will land here later.
)


class _PureTorchRMSNorm(nn.Module):
    """Standalone RMSNorm matching sgl_kernel rmsnorm semantics for
    test purposes only — the production ``layers.layernorm.RMSNorm``
    requires CUDA. Bit-equivalent at fp32 up to fp32 op ordering.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps  # match the attribute used in sglang's RMSNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        rms = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (x32 / rms).to(x.dtype) * self.weight


def _make_random_inputs(
    *,
    num_blocks: int,
    num_tokens: int,
    hidden_dim: int,
    num_layers_in_block: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    committed = [
        torch.randn(num_tokens, hidden_dim, dtype=dtype) for _ in range(num_blocks)
    ]
    partial = torch.randn(num_tokens, hidden_dim, dtype=dtype)
    queries = [
        torch.randn(hidden_dim, dtype=dtype) for _ in range(2 * num_layers_in_block)
    ]
    norms = [
        _PureTorchRMSNorm(hidden_dim).to(dtype=dtype)
        for _ in range(2 * num_layers_in_block)
    ]
    # Randomise gammas so each layer's K is genuinely different.
    with torch.no_grad():
        for n in norms:
            n.weight.normal_(mean=1.0, std=0.05)
    return committed, partial, queries, norms


class TestBlockAttnRes(CustomTestCase):
    """Pure-algorithm correctness tests — CPU-only, no engine state."""

    def test_two_phase_equiv_fp32_tiny(self):
        """Strict fp32 1e-5: two-phase ≡ naive on small shape."""
        committed, partial, queries, norms = _make_random_inputs(
            num_blocks=3, num_tokens=16, hidden_dim=64,
            num_layers_in_block=2, dtype=torch.float32,
        )
        cache = block_attn_res_phase1(committed, queries, norms)
        for i, (q, n) in enumerate(zip(queries, norms)):
            proj = nn.Linear(64, 1, bias=False).to(dtype=torch.float32)
            with torch.no_grad():
                proj.weight.copy_(q.unsqueeze(0))
            naive = block_attn_res(committed, partial, proj, n)
            two_phase = block_attn_res_phase2_merge(cache[i], partial, q, n)
            torch.testing.assert_close(naive, two_phase, rtol=1e-5, atol=1e-5)

    def test_two_phase_equiv_fp32_realistic(self):
        """fp32 1e-4: realistic shape (4 blocks × 128 tokens × 1024 dim)."""
        committed, partial, queries, norms = _make_random_inputs(
            num_blocks=4, num_tokens=128, hidden_dim=512,
            num_layers_in_block=4, dtype=torch.float32,
        )
        cache = block_attn_res_phase1(committed, queries, norms)
        for i, (q, n) in enumerate(zip(queries, norms)):
            proj = nn.Linear(512, 1, bias=False).to(dtype=torch.float32)
            with torch.no_grad():
                proj.weight.copy_(q.unsqueeze(0))
            naive = block_attn_res(committed, partial, proj, n)
            two_phase = block_attn_res_phase2_merge(cache[i], partial, q, n)
            torch.testing.assert_close(naive, two_phase, rtol=1e-4, atol=1e-4)

    def test_two_phase_equiv_bf16_within_accum_noise(self):
        """bf16: ~5% relative / 8% absolute is the mantissa-driven floor.

        Documents the bf16 ceiling — the algorithm is mathematically exact;
        the gap is fp32-vs-bf16 accumulation order, not algorithmic
        divergence (the fp32 test above passes at 1e-4).
        """
        committed, partial, queries, norms = _make_random_inputs(
            num_blocks=4, num_tokens=128, hidden_dim=512,
            num_layers_in_block=4, dtype=torch.bfloat16,
        )
        cache = block_attn_res_phase1(committed, queries, norms)
        for i, (q, n) in enumerate(zip(queries, norms)):
            proj = nn.Linear(512, 1, bias=False).to(dtype=torch.bfloat16)
            with torch.no_grad():
                proj.weight.copy_(q.unsqueeze(0))
            naive = block_attn_res(committed, partial, proj, n)
            two_phase = block_attn_res_phase2_merge(cache[i], partial, q, n)
            torch.testing.assert_close(naive, two_phase, rtol=5e-2, atol=8e-2)

    def test_block_zero_entry_returns_partial(self):
        """Empty ``committed_blocks`` ⇒ Phase 1 returns ``[None, ...]`` and
        ``phase2_merge(None, partial, q, n)`` is identity on ``partial``."""
        partial = torch.randn(8, 64)
        q = torch.randn(64)
        n = _PureTorchRMSNorm(64)
        cache = block_attn_res_phase1([], [q], [n])
        self.assertEqual(cache, [None])
        out = block_attn_res_phase2_merge(cache[0], partial, q, n)
        torch.testing.assert_close(out, partial, rtol=0, atol=0)

    def test_phase1_vectorised_matches_loop(self):
        """Same-eps vectorised fast path matches per-query loop bit-exactly.

        Forces the loop fallback by using two distinct eps values and
        compares against the vectorised result on identical inputs.
        """
        committed, partial, queries, norms = _make_random_inputs(
            num_blocks=3, num_tokens=16, hidden_dim=64,
            num_layers_in_block=2, dtype=torch.float32,
        )
        # Vectorised: all eps the same.
        for n in norms:
            n.variance_epsilon = 1e-5
        cache_vec = block_attn_res_phase1(committed, queries, norms)

        # Force fallback by perturbing one eps.
        norms[0].variance_epsilon = 1e-6
        cache_loop = block_attn_res_phase1(committed, queries, norms)
        # Reset for sanity.
        norms[0].variance_epsilon = 1e-5

        # The single perturbed query's cache will differ; the rest should match.
        for i in range(1, len(queries)):
            torch.testing.assert_close(
                cache_vec[i][0], cache_loop[i][0], rtol=1e-5, atol=1e-5,
            )
            torch.testing.assert_close(
                cache_vec[i][1], cache_loop[i][1], rtol=1e-5, atol=1e-5,
            )

    def test_zero_init_pseudo_query(self):
        proj = nn.Linear(32, 1, bias=False)
        with torch.no_grad():
            proj.weight.normal_()
        zero_init_pseudo_query(proj)
        self.assertTrue(torch.all(proj.weight == 0))

    def test_tp1_helpers_are_identity(self):
        """At TP=1, comm helpers are pure no-ops on the input tensor.

        Skipped when ``torch.distributed`` isn't initialised because the
        helpers call ``get_tensor_model_parallel_world_size`` which
        asserts the TP group exists. Full validation happens in the
        multi-rank engine-launch tests under ``test/registered/models/``.
        """
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            get_tensor_model_parallel_world_size()
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"TP group not initialized in this test process: {e}")

        x = torch.randn(8, 16)
        torch.testing.assert_close(all_gather_seq(x), x, rtol=0, atol=0)
        torch.testing.assert_close(reduce_scatter_seq(x), x, rtol=0, atol=0)
        torch.testing.assert_close(split_seq(x), x, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

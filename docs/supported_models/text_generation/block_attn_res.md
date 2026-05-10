# Block Attention Residual (AttnRes)

Block AttnRes ([Kimi tech report, arxiv 2603.15031](https://arxiv.org/abs/2603.15031))
is a **residual-stream overlay**: it replaces a stock decoder's fixed
PreNorm `x + f(x)` accumulation with a learned attention-aggregation
over a list of prior committed-block representations. Same family as
ByteDance Hyper-Connections ([2409.19606](https://arxiv.org/abs/2409.19606))
and DeepSeek mHC ([2512.24880](https://arxiv.org/abs/2512.24880)).

This integration is **pure-PyTorch** (no custom CUDA / triton / tilelang
kernels), built on top of `torch.distributed` primitives.

## Layout

* `sglang/srt/layers/attn_res.py` — algorithm only:
  * `block_attn_res(...)` — naive single-pass aggregator (numerical reference).
  * `block_attn_res_phase1(...)` — batched committed-side attention, run
    once per block. Vectorised across all per-layer pseudo-queries when
    they share the same RMSNorm `eps` (true for Kimi/Qwen3); falls back
    to a per-query loop otherwise.
  * `block_attn_res_phase2_merge(...)` — per-layer max-stable LSE online
    softmax merge with the current `partial_block`.
  * `all_gather_seq` / `reduce_scatter_seq` / `split_seq` — TP comm
    helpers for sequence-dim sharding of block representations.

* Per-carrier overlays in `sglang/srt/models/`:
  * `attn_res_overlay.py` — Kimi Linear (MLA + KDA + MoE).
  * `qwen3_attn_res_overlay.py` — Qwen3 dense (no MLA / no MoE).

To add a new carrier, mirror the Qwen3 file (~290 LOC); inherit the
target's `*ForCausalLM`, `*Model`, `*DecoderLayer`, override `__init__`
to add four `attn_res_proj/norm` + `mlp_res_proj/norm` parameters per
layer, and replace `forward` with the per-block `_forward_one_block`
pattern from the Kimi overlay.

## Two-phase computation

> Reference: <https://zhuanlan.zhihu.com/p/2017528295286133070>

The naive aggregator re-reads every committed block on **every layer**
within a block, costing `O(2·L_block · N · T · D)` reads of the committed
list. Two-phase amortises that down to:

```
Phase 1 (per block, once):  one batched attention against committed_blocks
                            for ALL queries in this block
Phase 2 (per layer, twice): online-softmax merge of (Phase 1 cache, partial_block)
```

The merge is **mathematically exact** via max-stable LSE — same trick as
Flash Attention's split-K. A numerical-equivalence regression
(`assert_two_phase_equivalent`) verifies fp32 1e-5 strict, bf16 within
mantissa-driven accumulation noise.

Total: `O(N · T · D + 2·L_block · T · D)` reads of the committed list per
block. Reduction factor ≈ `L_block` at large `N`.

## Sequence-dim TP shard

Set `SGLANG_ATTN_RES_SEQ_SHARD=1` to enable. Block representations
are stored as `(T/P, D)` shards per TP rank; Phase 2 merge + RMSNorm
run on the shard; reduce-scatter / all-gather replace the standard
all-reduce path:

```
attn_in_replicated  =  all_gather(input_layernorm(h_shard))
attn_out_partial    =  self_attn(attn_in_replicated)   # o_proj reduce_results=False
attn_out_shard      =  reduce_scatter(attn_out_partial)
partial_block_shard +=  attn_out_shard
```

Net comm = `2 × (RS + AG)` per layer ≡ `2 × AllReduce` cost (NCCL-wise).
The win is **memory**: at long context the block representations
themselves can dominate (paper: 128 K context, 8 blocks, d=7168 →
15 GB → 1.9 GB per rank with TP=8 shard).

When `num_tokens % TP != 0` (typical batch=1 decode steps) the path
auto-falls back to replicated mode and logs a once-per-instance
warning.

## Limitations / future work

* **DP attention not supported.** The overlay bypasses
  `LayerCommunicator.prepare_attn` because the fused PreNorm
  residual-add path conflicts with AttnRes's softmax-aggregation
  semantics. DP-attn scatter routing also lives there. Engine init
  raises `NotImplementedError` if `enable_dp_attention` is on; tracked
  as phase 11 follow-up.
* **CUDA graph capture** has not been validated end-to-end on the
  two-phase path. Run with `--disable-cuda-graph` for now if you hit
  capture-time errors.
* **Fused kernels.** Pure-PyTorch only. The aggregation algorithm is
  sufficiently lightweight (one batched einsum + one elementwise
  merge) that a fused kernel would be a small relative speedup; not
  on the immediate roadmap.

## Running

Construct a config with `attn_res_num_blocks=N` and use a registered
overlay arch (e.g. `KimiBlockAttnResForCausalLM`,
`Qwen3BlockAttnResForCausalLM`). Block boundaries are at every
`num_hidden_layers // attn_res_num_blocks` layers (1-indexed).

```python
import sglang as sgl
e = sgl.Engine(
    model_path="path/to/your/attn_res_ckpt",
    skip_tokenizer_init=False,
    tp_size=2, pp_size=2, ep_size=2,
    dtype="bfloat16",
    attention_backend="flashinfer",
    linear_attn_backend="triton",
)
out = e.generate(["Hello"], {"max_new_tokens": 32, "temperature": 0})
```

For the seq-shard memory optimisation:

```bash
SGLANG_ATTN_RES_SEQ_SHARD=1 python3 -m sglang.launch_server ...
```

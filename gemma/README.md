# Gemma-2B persistent decode prototype

A standalone **Megakernels-style** implementation for `google/gemma-2b` on
NVIDIA B200. Each decode forward pass and greedy sample runs inside one
cooperative persistent CUDA launch. CTAs follow static instruction streams,
partition output tiles, and synchronize between operations. This is new CUDA
code inspired by the scheduling approach; it does not use the original
ThunderKittens Llama kernels, their TMA pipelines, or their instruction ABI.
The original Llama demo is unchanged.

Full pretrained model: 18 layers, hidden size 2048, MLP size 16384, 8 query
heads and one KV head, head dimension 256, vocabulary 256000. BF16 weights
and activations, FP32 reductions. Implemented operations include Gemma's
scaled embeddings, offset RMSNorm, non-interleaved RoPE, MQA with KV append,
residual projections, GELU-gated MLP, tied embedding/output weights, and
vocabulary argmax. Prefill uses Transformers and transfers its real KV cache.

## Build and run

```bash
cd /workspace/Megakernels/gemma
make
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/mirage/.venv/bin/python bench.py \
  > results/mk.log 2>&1
PATH=/workspace/Megakernels/local/vllm-venv/bin:$PATH \
  HF_HUB_OFFLINE=1 /workspace/Megakernels/local/vllm-venv/bin/python bench_vllm.py \
  > results/vllm.log 2>&1
```

Run sequentially. The model is already downloaded to the Hugging Face cache;
new installations require authorized access to `google/gemma-2b`.
Build: CUDA 12.8, `sm_100`, register limit 64. GEMV uses 128-bit loads
and four independent FP32 accumulation chains. Runtime: Python 3.12,
PyTorch 2.8.0+cu128, Transformers 4.57.1, using the existing Mirage environment.
vLLM uses the existing Megakernels venv: vLLM 0.31.0, PyTorch 2.13.0+cu130.
No dependencies were changed in that vLLM environment.

## Measured results

2026-10-15, single NVIDIA B200, batch size 1. Identical prompt token IDs,
greedy generation, 128 output tokens, EOS ignored for fixed work. Decode
measurement excludes prefill and the first output token: **127 decode tokens**.
Two warmup requests followed by five measured requests, median shown.

| Implementation | Decode time | Time/token | Decode tokens/s |
|---|---:|---:|---:|
| Gemma persistent prototype | 267.67 ms | 2.108 ms | 474.47 |
| vLLM | 257.75 ms | 2.030 ms | 492.73 |

The optimized implementation is **about 2× faster than its original 4.18 ms/token
version**. A fresh vLLM run is about 4% faster than this implementation.
Paired measurements loading baseline and optimized code into the same process
confirm 4.17 vs 2.10 ms/token (1.99×). Historical vLLM measurements were slower;
use the refreshed numbers above when comparing the final code.

The selected launch uses 592 resident CTAs (four per SM). Stage timing identified
matrix-vector products as about 88% of the original runtime. Wider weight loads
and independent accumulation chains are the retained optimization. Residual/norm
fusion, two warps per output row, and loop unrolling all regressed and were
reverted. Their source and measurements remain in separate commits; see
[OPTIMIZATIONS.md](OPTIMIZATIONS.md).

This remains a standalone prototype with full-grid barriers, simple GEMV, and
one-CTA-per-head attention; it lacks the original project's memory pipelines.

Custom timing uses synchronized wall time around all 127 steps, including
output token copies. vLLM decode time is its engine's `last_token_ts` minus
`first_token_ts`; full request wall time is also saved. Normal vLLM compilation,
FlashInfer attention, and CUDA graphs are enabled, with prefix caching disabled.
Startup, compilation, cache transfer, and warmup are excluded. Different
PyTorch/CUDA software stacks and timing instrumentation remain limitations.
This is a short-context, batch-one decode comparison, not a serving benchmark.

## Correctness evidence

- All **128 generated token IDs exactly match vLLM** for the benchmark prompt.
- Twelve teacher-forced checks across three other prompts and four decode
  positions each match Transformers' next-token argmax (12/12).
- Minimum logit cosine similarity in those checks: **0.9999719**.
- Additional checks at KV positions 128, 511, and 1023 match the original
  implementation and Transformers' next-token argmax; minimum cosine 0.9999975.
- Logits are not bitwise equal. These are smoke checks, not a comprehensive
  model accuracy evaluation; exact error metrics are in `results/validation.json`.

The prompt is `The solar system contains eight planets. The largest planet is`.
Both implementations begin ` Jupiter, which has a diameter of 88,846 kilometers.`
The generated text includes incorrect facts from the base model; matching
outputs verifies this workload, not factual accuracy. Forced fixed-length
execution continues with EOS tokens after a natural stop.

Raw artifacts in `results/`: `summary.json`, `mk.json`, `vllm.json`,
`validation.json`, `correctness.json`, `tuning.json`,
`same_session_comparison.json`, build and runtime logs.

For a reproducible comparison against the original commit (builds its library
in a temporary directory), run:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/mirage/.venv/bin/python verify_optimization.py
```

`profile_stages.py` instruments stage cycles in a separate profiling run; these
cycle fractions include each stage's barrier and are diagnostic, not standalone
operation microbenchmarks.

Limitations: original Gemma-2B only (not Gemma 2 2B), one sequence, greedy,
CUDA BF16, at most 1024 KV positions (512 used here), no server or continuous
batching. `model.py` rejects unsupported model geometry and KV overflow.
The Cohere-style implementation was dropped at the user's request.

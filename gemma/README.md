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
Build: CUDA 12.8, `sm_100`, register limit 64. Runtime: Python 3.12,
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
| Gemma persistent prototype | 531.09 ms | 4.182 ms | 239.13 |
| vLLM | 284.31 ms | 2.239 ms | 446.70 |

**vLLM is 1.87× faster.** This prototype does not improve on vLLM.
Paired BF16 loads and worker tuning improved the initial prototype from
about 12.58 ms/token to 4.18 ms/token. The selected launch uses 592 resident
CTAs (four per SM). Tuning results and compiler occupancy evidence are saved.
The prototype still has full-grid barriers, naive GEMV, and one-CTA-per-head
attention; it lacks the original project's optimized memory pipelines.

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
- Initial check: cosine 0.9999964, mean absolute error 0.0957, maximum absolute
  error 0.7235. Logits are not bitwise equal; these are smoke checks, not a
  comprehensive model accuracy evaluation.

The prompt is `The solar system contains eight planets. The largest planet is`.
Both implementations begin ` Jupiter, which has a diameter of 88,846 kilometers.`
The generated text includes incorrect facts from the base model; matching
outputs verifies this workload, not factual accuracy. Forced fixed-length
execution continues with EOS tokens after a natural stop.

Raw artifacts in `results/`: `summary.json`, `mk.json`, `vllm.json`,
`validation.json`, `correctness.json`, `tuning.json`, build and runtime logs.

Limitations: original Gemma-2B only (not Gemma 2 2B), one sequence, greedy,
CUDA BF16, at most 1024 KV positions (512 used here), no server or continuous
batching. `model.py` rejects unsupported model geometry and KV overflow.
The Cohere-style implementation was dropped at the user's request.

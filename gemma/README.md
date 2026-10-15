# Gemma-2B on the upstream Megakernels VM

This implementation runs pretrained **google/gemma-2b** on a single B200 using
`megakernel::mk` from `include/megakernel.cuh`, not a cooperative-grid loop.
The original controller, instruction fetch, page allocator, loader, launcher,
consumer, storer, and instruction-ring protocol execute the model.

Each SM owns a packed instruction stream. There are **no `grid.sync()` calls
or cooperative launches** in the active implementation. Cross-SM synchronization
uses device-scope acquire/release counters and per-tile generation flags:

- RoPE waits for its query head's QKV tiles; the KV task waits for K and V.
- Attention waits for its query-head RoPE tile and the common KV append.
- Residual and GELU tasks wait for the particular producer tiles they read.
- Gate and up projections independently depend on their normalized input.
- RMSNorm, dense reductions, and sampling wait for all required vector inputs,
  rather than requiring every worker to reach a grid-wide barrier.

All 91 matrix projections use the original three-stage TMA matvec pipeline in
`demos/low-latency-llama/matvec_pipeline.cuh`, including its weight pages, shared
semaphores, asynchronous loader, consumer computation, and output storer.
A backward-compatible activation-provider callback supports 16384-wide down
projections as eight 2048-element chunks. Their FP32 partial sums combine before
rounding to BF16. The original Llama operators keep their existing behavior.

The full Gemma geometry is supported: 18 layers, hidden size 2048, MLP size
16384, eight query heads, one KV head, head dimension 256, and vocabulary 256000.
Gemma's embedding scale, offset RMSNorm, GELU gating, and non-interleaved RoPE
are preserved. Transformers handles prefill and supplies the real KV cache.
One native VM launch computes a decode forward pass and greedy sample.

## Build and run

```bash
cd /workspace/Megakernels/gemma
make
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/mirage/.venv/bin/python bench.py \
  > results/native-final-bench.log 2>&1
PATH=/workspace/Megakernels/local/vllm-venv/bin:$PATH \
  HF_HUB_OFFLINE=1 /workspace/Megakernels/local/vllm-venv/bin/python bench_vllm.py \
  > results/native-vllm-bench.log 2>&1
```

Run GPU benchmarks sequentially. CUDA 12.8 builds `sm_100a`, using this checkout's
ThunderKittens headers and native VM. Runtime: Python 3.12, PyTorch 2.8.0+cu128,
Transformers 4.57.1 in the existing Mirage venv. vLLM uses the existing
Megakernels venv (vLLM 0.31.0, PyTorch 2.13.0+cu130). No package changes were
needed. Authorized Gemma weights are already cached, revision
`9cf48e52b224239de00d483ec8eb84fb8d0f3a3a`.

## Current measurements

2026-10-15, one B200, batch size one, BF16, greedy, identical prompt IDs,
128 output tokens with EOS ignored. Measurement covers the remaining **127
decode tokens after the first output token**, excluding prefill, cache transfer,
compilation, and two warmup requests. Median of five measured requests:

| Implementation | Time/token | Decode tokens/s |
|---|---:|---:|
| Native VM, TMA, and tile counters | 2.576 ms | 388.16 |
| vLLM, refreshed | 2.135 ms | 468.41 |

**vLLM is 1.21× faster.** The native port also regresses versus the prior
optimized cooperative prototype: the same-process comparison measures 2.575
versus 2.102 ms/token (about 22.5% slower). Using the intended VM and finer
synchronization did not by itself improve this Gemma implementation's speed.
This port does not yet reproduce the original Llama demo's fused operator
boundaries or tuned Gemma-specific schedules. Norm, RoPE, residual, GELU,
attention, and sampling still use straightforward consumer adapters.

Custom timing is synchronized wall time including output-token copies. vLLM
uses its engine's last-token minus first-token timestamps; full request wall
time is saved too. Normal vLLM compilation, FlashInfer attention, and CUDA graphs
are enabled; prefix caching is disabled. Different software stacks and timing
boundaries remain limitations. This is not a serving-throughput benchmark.

## Verification and history

- All 128 benchmark output token IDs exactly match vLLM.
- All 12 teacher-forced next-token checks match Transformers.
- Checks at KV positions 128, 511, and 1023 match Transformers and the previous
  cooperative implementation.
- After 127 repeated launches, all 220 stage counts and all 44,514 producer-tile
  generations match the expected values. Generation tags avoid per-token resets.
- The original Llama demo rebuilds with the shared pipeline extension and
  generates a coherent 32-token response, measuring about 1610 tokens/s.

These are correctness smoke checks, not a comprehensive accuracy evaluation.
The base model can generate incorrect facts; exact output agreement does not
establish factual accuracy.

Reviewable commits and earlier experiments are documented in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md). The old cooperative implementation remains
at `4a404a0`; the active working tree contains only the native kernel.

Reproduce the same-process comparison against that commit:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/mirage/.venv/bin/python verify_optimization.py
```

Record the upstream VM's instruction events in a separate instrumented build:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/mirage/.venv/bin/python profile_stages.py
```

Its instruction lifetimes overlap and must not be added as operation times.
`results/native_profile.json` contains those measurements. Other raw evidence:
`summary.json`, `mk.json`, `vllm.json`, `native_same_session_comparison.json`,
`correctness.json`, `validation.json`, all port benchmark logs, and Llama
regression logs. `mk.json` records hashes of the measured source files.

Limits: original Gemma-2B (not Gemma 2 2B), one sequence, greedy, BF16 on B200,
at most 1024 KV positions, no server or continuous batching. `close()` releases
native descriptor resources after pending GPU work completes.

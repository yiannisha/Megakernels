# Gemma implementation and optimization history

## Complete fused native implementation

The current tree implements every Gemma forward operation as a named fused
native operator. It has no generic operation adapter or cooperative kernel.
Native prefill also runs through the same VM, with no framework forward calls.

| Commit | Change |
|---|---|
| `d7f35f6` | Fused normalization/QKV/RoPE/KV append, gated GELU, residual projections, normalized LM head, and double-buffered split MQA |
| `672f98a` | Tensor-core probability/value attention with BF16 online softmax tiles; restores exact benchmark-token agreement |
| `54030f2` | Native prefill API and context-boundary validation up to position 8191 |
| Final completion commit | Entire 8192-token native prefill, standalone native demo, historical comparisons, profiling, and refreshed vLLM benchmark |

Main result: **1.825 ms/token, 548 tokens/s**, versus refreshed vLLM
**2.247 ms/token, 445 tokens/s** (1.23× throughput) for the short-context,
batch-one decode workload. All 128 output IDs match. Paired same-process
comparisons verify 1.40× over `08e198d` and 1.14× over `4a404a0`.

The graph shrinks from 220 to 111 stages. The 512-position configuration shrinks
producer synchronization state from 44,514 tile flags to 1,044 grouped counters.
QKV tiles pair RoPE partners; gate/up weight tiles alternate so GELU and gating
happen in the storer; residuals and final norm are fused. Attention uses TMA for
both K and V, tensor cores for both matrix products, and stable split combination.
There is no global normalized-activation, raw-QKV, gate, or up intermediate.

`validate_fused.py` verifies the complete 8192-token native prefill while
framework forward is patched to raise. It also checks native prefill on three
short prompts, nine real-cache boundaries, and every stage/counter invariant.
Native prefill is tokenwise (about 20 seconds for 8192 tokens), not a batched
prefill optimization. Decode benchmark speedups exclude prefill.

See README.md and `results/summary.json` for final commands and measurement
boundaries. The earlier sections below describe historical implementations.

## Earlier unfused native VM port

The earlier unfused implementation used the actual upstream `megakernel::mk` VM, with
three-stage TMA weight pipelines, native shared-memory page management, and
acquire/release tile counters. The cooperative kernel is preserved in Git at
`4a404a0` and removed from the active build.

| Commit | Change |
|---|---|
| `8889d2a` | Port to upstream VM; reuse TMA pipeline for 2048-wide projections |
| `86cead1` | Per-tile generation flags and actual producer-tile dependencies |
| `665a3e9` | TMA for 16384-wide projections; FP32 chunk accumulation; balanced instruction streams |
| `09e6c2e` | Acquire loads instead of atomic read-modify-write polling; release publication |
| `08e198d` | Counter-state checks, resource cleanup, native profiling, and final validation |

That version measured **2.576 ms/token**, versus refreshed vLLM **2.135 ms/token**.
The previous cooperative prototype is faster: an alternating, same-process
comparison measures 2.102 vs 2.575 ms/token. The port satisfies the native
scheduler/synchronization requirement but is not a speedup. The shared pipeline's
new activation-provider overload preserves the default behavior for existing
operators; the original Llama build and 32-token generation both pass.

Fine-grained tile flags apply to QKV/RoPE, attention, residuals, and GELU.
Completion-counter updates are batched by instruction; polling uses acquire
loads, not RMW atomics. Stage counts and tile generations are verified after
repeated launches. All model GEMVs use the original TMA input/output ring;
16384-wide inputs change the register activation vector for each 2048-element
chunk via a backward-compatible callback.

See `results/native_same_session_comparison.json`, `native_profile.json`,
`native-final-bench.log`, `native-vllm-bench.log`, and `llama-regression-run.log`.
The old cycle percentages below describe the cooperative prototype, not this VM.

## Earlier cooperative prototype history

All versions below are local commits in `/workspace/Megakernels`. Changes affect
only the new `gemma/` implementation. The original Llama kernels are preserved.

| Commit | Change | Measured ms/decode token | Disposition |
|---|---|---:|---|
| `ba2c15d` | Original BF16 cooperative persistent implementation | 4.182 | Baseline |
| `5489f01` | Add per-stage GPU cycle instrumentation | — | Retained profiling interface |
| `dc32922` | 128-bit loads, four independent GEMV accumulators | 2.106 | Retained optimization |
| `89946a5` | Fuse residual addition with offset RMSNorm | 2.338 | Rejected regression |
| `fcf5809` | Revert fusion; use two warps per narrow output row | about 2.5 | Rejected regression |
| `2c7bdcb` | Revert split rows; unroll GEMV loop four times | 2.767 | Rejected regression |
| `4a404a0` | Restore wider loads; add same-process and long-context verification | 2.108 | Selected implementation |

These are sequential reviewable commits. Each experimental commit explicitly
replaces the previous trial; compare the retained optimization to experiments
with `git diff dc32922 <experimental-commit> -- gemma/ops.cuh gemma/model.py`.
Timing uses two warmups followed by five requests of 127 decode tokens.
Compiler: CUDA 12.8, `-O3 -maxrregcount=64 -arch=sm_100`. Same pretrained weights,
precision, cache lengths, output length, and selected worker count are used.
Trial timings include complete decode-loop wall time; raw logs are committed.

## Why wider loads helped

Original stage instrumentation attributes about 75% of cycles to model GEMVs
and another 13% to the vocabulary projection. Attention is about 3%. The useful
change loads eight BF16 elements (128 bits) per lane per iteration instead of
two, reducing load/address/loop instruction counts. Four independent FP32 sums
reduce the serial FMA dependency chain. The warp reduction combines those sums
at the end. It does not change weights, quantize activations, approximate GELU,
or replace model operations.

The later vector-only profile shifts GEMV plus vocabulary projection to about
71% and normalization/barrier stages to about 12%. Fractions include barrier
wait time, so they are not pure compute times. The later experiments show why
reducing stage count alone is not sufficient; no hardware-counter evidence was
collected to attribute each rejected regression more precisely.

## Verification and comparison

The earlier verification run built the baseline directly from `ba2c15d` in a
temporary directory and alternated both versions with the same model. Result: baseline **4.17 ms/token**,
optimized **2.10 ms/token**, **1.99× speedup**. All sampled output sequences match.
Long-context checks at positions 128, 511, and 1023 match both baseline and
Transformers. See `results/same_session_comparison.json` and `verification.log`.

The cooperative version's standard benchmark also matched all 128 generated vLLM token IDs and
passes 12 short-context teacher-forced checks. The refreshed vLLM baseline is
**2.030 ms/token**, versus **2.108 ms/token** here: vLLM remains approximately 4%
faster on this single prompt at batch size one. This does not establish broader
serving throughput or comprehensive model accuracy.

To inspect a version, use `git show <commit>:gemma/ops.cuh` or a separate worktree.
The benchmark scripts need the cached model and the installed Python environments
described in README.md. Rebuild `kernel.so` after changing versions; compiled
libraries are deliberately not committed.

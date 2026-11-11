# Performance-first execution contract

The human explicitly authorized runtime optimization before accuracy debugging, while retaining 50,000 complete accepted physical vectors in every end-to-end benchmark. The corrected raw33/context4096 unoptimized control still fails the original physical screen. That result remains visible and does not veto this performance phase. Public code, issue and benchmark updates were explicitly reconfirmed on 2026-11-11.

Current tracking is [issue 8](https://github.com/BrieucB/PosteriorForge/issues/8). The [completed control evidence](../evidence/performance-first-v1/controls/report.md) contains actual posterior, timing, memory and bounded-trace PNGs. Three control request times are 2364.5507, 2383.4860 and 2380.3091 seconds; the median is 2380.3091. Repetition 0 reproduced the earlier corrected sample payload byte for byte. No optimized full-request speedup has yet qualified.

## Execution and sampling

`execution_cache.py` explicitly prepares the stock eight-member KV engine for one autoregressive stage. All eight models and their KV tensors stay on the GPU during that stage; they are released afterward. The pinned SDK files and scientific source remain unchanged. A guarded private iterator clone removes only the stock per-prediction CPU offload. Model calls restore CPU and selected CUDA random state; posterior streams are separately declared.

`execution_mixture.py` samples the arithmetic probability mixture without evaluating all eight members for every row. Each member still retains its full statistical weight. For every logical row and stage, propose a uniform member independently. Let `p_i` be its raw corrected canonical bin probabilities and `m_i=sum(p_i)`. Accept the proposed member with probability `m_i/M`, using a checked envelope `M=1.00001`, then draw from that member's normalized canonical finite-bin distribution. Accepted member weights are `m_i/sum(m_i)`, so the resulting bin distribution is `sum(p_i)/sum(m_i)`. The three direct reconstruction checks separately measure the remaining FP32 mean/log/softmax differences.

Head proposals, head acceptance and sampling uniforms use distinct recorded CPU generators. Each stage preassigns sampling uniforms by logical row; head retries retain that row's uniform. A single first-stage full-mixture density can serve the identical observation rows, with independent uniforms for every output. That predicted density is proposal-wave-local and cannot leak to another query or case. Component outputs do not claim full-mixture log densities.

All 21 parameters are generated before the original whole-vector support rejection. There is no partial-vector filter, clipping, tail repair, iteration fallback or invalid-row acceptance. `performance_runner.py` requires 50,000 outputs for a standalone benchmark. Only an authenticated worker contract permits 25,000 or 12,500 local outputs, and those worker artifacts explicitly are not end-to-end benchmarks. `run_execution_workers.py` gathers exactly 50,000 complete rows from one, two or four independent full model/context replicas on one node. Its timer includes child startup and aggregation. Per-worker CPU thread allocation avoids multiplying the full allocation's thread count on every rank.

## Qualification and retained failures

FP32 pointwise thresholds are normalization error `1e-5`, maximum conditional CDF difference `2e-5`, and row total variation `1e-4`. Borders, member configurations, dtype, preprocessing, normalization, embedding/state and corrected translator execution must also match. Fixed conditional probes are engineering tests; they never substitute for a complete posterior benchmark.

Probe 01 is retained as failed. Its inverse-CDF coordinate bracket incorrectly treated ideal quantile values as stable under the pinned FP32 cumulative arithmetic, and also failed control-versus-control. Independently reviewed contract v2 retains exact stock arithmetic replay and uses probability-space residuals. It neither repairs production samples nor relaxes the probability thresholds. Endpoint anomalies remain recorded; `u=1` is stress-only.

Probe 02 genuinely failed at stage 10/member 5: the member CDF changed by about `2.94e-4` and the final mixture by `3.76e-5`. Probe 03 also failed at stage 13/member 6, whose CDF change narrowly exceeded `2e-5`. Exact borders and preprocessing identities passed. These failures are not relabelled as passes. Separate stage records classify the remaining map. A new explicitly declared hybrid path is being qualified, retaining corrected uncached execution for failing members while preserving all eight mixture weights. No hybrid full-21 pass has yet been claimed.

The cached and uncached multiquery paths use different FP32 projection and attention shapes. This is a concrete source difference to investigate; it is not yet a proven cause of either mismatch.

## Performance and accuracy decisions

Relative empirical CDF change above `.01`, mean change above `.05` reference SD, or SD change above `5%` triggers investigation. The unchanged controls themselves show small CDF flags and one large SD flag: one retained `a_a3=90.87128` value accounts for 88.8% of that repetition's squared deviation. It lies inside the inherited broad terminal finite bar; its original per-draw logits/uniform were not retained, so precise tail-event causality is unproven. No row or flag is removed.

Physical accuracy remains separately assessed with the original numerical references and thresholds. After the fastest qualified engine is frozen, separately declared sufficient-statistic and other authorized accuracy ablations follow. Every GPU allocation, failure and retry counts toward the existing immutable 200 allocated GPU-hour ledger. Request wall-products and bounded trace timings are distinct from scheduler allocation accounting and complete-request measurements.

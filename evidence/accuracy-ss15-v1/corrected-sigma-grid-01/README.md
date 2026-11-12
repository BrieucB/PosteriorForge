# Corrected sigma-first grid: partial failure

Built with PriorLabs-TabPFN. **6/24 points assessed: 2 pass, 4 model failures,
18 pending in this snapshot. The all-point candidate gate has failed.**
The fixed 3-context-seed by 8-case diagnostic continues, retaining every result.

The method is explicitly **hybrid NPE-PFN/Gibbs**, not pure TFM inference:
SS15 observations, all-three-global block-local conditioning, the fixed
sigma-first order in all six blocks, and one exact joint refresh of the three
globals. The scientific rule, original 50,000-vector count, source-bound
execution policy and physical reference limits are fixed; there is no
outcome-dependent mask, order, precision or kernel change during this grid.
The exact global refresh preserves all 18 local/noise columns byte for byte.

This is a development diagnostic, **not evidence of global calibration, SBC,
or an untouched confirmation set**. The retained
[development history](../structured-candidates-01/README.md) includes the failed
MAP2 numerical screen and earlier prototype. The first point here is the
**fresh corrected case 0 cohort**, not that prototype. The numerical execution
screen covers separate finite case 0 prefixes; it does not screen other cases
or establish posterior accuracy.

## Current results

| Seed | Case | Status | Marginal CDF upper | Joint CDF upper | Mean-bias upper / ref SD |
|---|---:|---|---:|---:|---:|
| 20261111 | 0 | [pass](seed-20261111-case-0/README.md) | 0.090541 | 0.044952 | 0.199851 |
| 20261111 | 1 | [pass](seed-20261111-case-1/README.md) | 0.084424 | 0.089961 | 0.185531 |
| 20261111 | 2 | [model_failure](seed-20261111-case-2/README.md) | 0.116391 | 0.091392 | 0.272693 |
| 20261111 | 3 | [model_failure](seed-20261111-case-3/README.md) | 0.231336 | 0.126481 | 0.585035 |
| 20261111 | 4 | [model_failure](seed-20261111-case-4/README.md) | 0.098715 | 0.074394 | 0.205065 |
| 20261111 | 5 | [model_failure](seed-20261111-case-5/README.md) | 0.215403 | 0.166750 | 0.488252 |

Case 2 retains its qualified-reference model failure: a_m2 and b_m2 fail
both marginal CDF and mean-bias limits. Its joint CDF and SD-ratio checks pass.
Case 3 also fails marginal, mean, joint and SD-ratio criteria: marginal maximum
0.231336352, joint maximum 0.126480968, mean-bias maximum 0.585035025 reference SD,
and sigma_m1 SD-ratio lower bound 0.751748871. Technical controls pass for both.
Case 4 fails only the sigma_m2 mean-bias limit (upper 0.205064535 reference SD);
its CDF and SD-ratio checks pass. This failure is also retained.
No failed point is dropped and no method fallback is introduced. Original
limits remain CDF upper <=0.10, mean-bias upper <=0.20 reference SD,
SD-ratio interval inside [0.8,1.2], and reference precision <=0.005.

![Case 2 full densities, including mechanical-2 failures](seed-20261111-case-2/joint-global-density.png)

![Case 3 full densities, retaining all failed coordinates](seed-20261111-case-3/joint-global-density.png)

## First corrected point

Seed 20261111 / case 0 passes the unchanged physical gate. Its largest conservative
mean-bias bound is 0.199851420 reference SD, close to the 0.20 limit; this single
point does not establish generalization. These four original figures use all
50,000 rows. The focus panel is predeclared, not selected from the hybrid result.

![First corrected point: fixed focus](seed-20261111-case-0/joint-global-density-focus.png)

![First corrected point: all marginal densities](seed-20261111-case-0/joint-global-density.png)

![First corrected point: marginal CDF bounds](seed-20261111-case-0/joint-global-marginal-cdf.png)

![First corrected point: joint CDF bounds](seed-20261111-case-0/joint-global-joint-cdf.png)

Its neural posterior request took 200.903236 seconds; wrapper time was 206.314274
seconds and the single-GPU allocation was 209 seconds with zero restarts.
The separate CPU global refresh is recorded separately, not folded into the
neural request time.

## Exact artifacts and extension

grid-index.csv and grid-index.json retain all 24 predefined rows, including
pending points. Standard subsequent assessments come from
sigma-native-grid-assessment-01/seed-SEED-case-CASE/.
Add only checksum-closed assessments; preserve failed results and update the
counts, point files and publication checksums together.

Each point has its exact assessment, figures, original checksums, a provenance
JSON file, and gzip copies of the **unmodified raw parent/refreshed manifests**.
Decompress each archive and compare it with its corresponding raw SHA256
sidecar. publication-checksums.json checks every packaged file.
No raw sample arrays or private CV material are included. The frozen source
and proof identities are recorded in grid-index.json.

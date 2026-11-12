# Prior-coordinate accuracy pilot v1

Built with PriorLabs-TabPFN. This is a new accuracy-debugging candidate and a predeclared bounded experiment, not a performance study or an accuracy claim. Public technical publication is authorized. Personal CV and interview material remain private.

The frozen sigma-native 24-point grid must first finish unchanged: seeds 20261111, 20261112 and 20261113, cases 0 through 7, exactly 50,000 complete physical vectors per point, one GPU at a time. Preserve all original failures, parent and hybrid assessments, and before/after figures. The next candidate cannot inherit that candidate's numerical qualification or native fallback decisions. A separately authenticated completion receipt must establish all 24 requests, technical/reference assessments and reconciled GPU allocations; physical gate failure does not prevent the new experiment.

## Fixed scientific construction

Keep the same 21 physical parameters, source prior and simulator, original 4096-world banks, eight held-out cases, each with 33 raw observations, fixed SS15 representation, frozen TabPFN v2 checkpoint, eight members and original qualified references. No new worlds, observation transforms, training, weights, reference-fitted constants, performance changes or multiple GPUs are authorized.

Transform globals with the source covariance's non-diagonal lower Cholesky factor. Transform local coefficients by subtracting their source conditional mean given all three globals, then whitening by the CONDITIONAL prior covariance. Transform each of the six noise parameters by its source truncated Gaussian prior CDF followed by the normal quantile: location .255, scale .1225, bounds [.01,.5]. Use stable lower/upper tail arithmetic. The first global is an affine relabeling and is not claimed to correct its learned density after SDK target standardization.

Use the fixed sigma-first stage order [0,1,2,6,3,4,5,10,7,8,9,14,11,12,13,16,15,18,17,20,19]. Local factors retain all transformed globals, their own SS15 observations and preceding transformed locals from the same dataset. Keep the SDK's existing preprocessing. Bind original physical Float64 arrays, latent Float64 arrays, the single final model Float32 cast, masked inputs and transformed target labels separately.

Sample in the regression coordinate using the unchanged mixture/component sampler and private random streams. Apply the correct causal inverse AFTER sampling latent bins; mapping nonlinear bin borders to physical coordinates followed by uniform physical-bin sampling is forbidden. Decode complete proposal vectors into original physical order before the unchanged FIFO and support sampler. Do not silently recast the sampled latent vector before decoding. Invalid, saturated or numerically inadmissible output fails the entire request, including generated proposal rows that FIFO would later leave unused.

## Boundary and CPU qualification

Inventory every value in all three original training banks and exactly the first 1024 saved accepted rows of original seed20261111/case2 parent manifest 28de1d121e5e89a5b55f630625c0f60e86551473c8bbaa3b2840797683f12e6a, payload 9ce954f75b73c54348733c3068179d542ed050ec1aaae772734b3def67fc752b. The prefix rule was fixed before inventory. Do not substitute or discard rows. Saved heldout observation bytes may be inspected; heldout truth parameters must not be used.

Training endpoints, nonfinite latents, CDF or inverse saturation, out-of-support decoded noise, and failed declared roundtrips are explicit qualification/request failures. No epsilon or nextafter clipping, inward moves, row deletion, coordinate repair or silent endpoint atoms. A later complete-vector numerical rejection policy would need a separate proposal and evidence before activation.

Separate Float64 physical roundtrips, Float64 latent conditioning/admissibility, and Float32 model-cast checks in BOTH locked environments. The frozen codec declares physical Float64 error at most 256 eps64 times (1+absolute standardized physical coordinate), latent Float64 admissibility at most 8 sqrt(eps64) times (1+absolute latent), and Float32 bounds of 32 eps32 in their stated units. The latent bound is an admissibility bound, not an ulp-Float64 accuracy guarantee. Physical units are source global marginal SDs, local CONDITIONAL SDs, and underlying noise scale .1225. These checks do not change model CDF, TV, normalization or quantile limits.

## Fresh backend qualification

Use a new immutable source, policy, cache and provenance identity. The initial policy has an EMPTY native fallback map. Compare original uncached FP32 with cached FP32 on IDENTICAL transformed masked inputs and labels at all 21 stages, fixed N=1/N=1024 prefix cases and declared stress inputs. Preserve independent sampler and causal decode tests, actual SDK alias restoration, source and random-state custody. Use the original limits: normalization 1e-5, CDF 2e-5, TV 1e-4, quantile error per span 2e-6. Collect all 21 independent fixed-prefix numerical comparisons when hard source, RNG, state and validity guards pass. Any cached-FP32 tolerance failure retains FAILED status and blocks pilot activation, while later independent stages are still inventoried. Stop immediately on hard integrity, source, RNG, state or validity failure. Do not silently reuse the old fallback map. Compare FP16 separately, preserving its strict failures and drift without calling it FP32 qualification.

The new producer stays disabled until its exact frozen sources, BOTH CPU inventory receipts, old-grid completion receipt and fresh all21 backend proof are pinned and authenticated. Qualification uses fixed prior/bank/prefix information and no reference fitting or posterior accuracy tuning.

## One authorized physical pilot

After completion and qualification pass, run exactly ONE full request at context seed 20261111, case 2, repetition 0, context size 4096, exactly 50,000 complete finite physical 21-vectors. Use qualified cached FP32 initially. This changes BOTH prior coordinates and forward precision relative to the old mixed FP16/FP32 grid; an improvement alone cannot isolate the causal benefit of the coordinate change. FP16 remains a separately labeled comparison, with no FP16 posterior pilot activated. Reserve and reconcile every GPU allocation through the existing 200 GPUh ledger, one GPU and no requeue; all failed and idle allocation time counts. Do not launch further validation automatically.

Decode first, then apply exactly the same one-sweep joint three-global prior conditional Gibbs update with its existing private stream role. All 18 physical columns 3 through 20 must remain byte-identical. Discard latent state after refresh, or re-encode against refreshed globals; never decode old local residuals against new globals.

Assess both neural parent and hybrid against the SAME original physical reference, using all 50,000 rows, all 21 marginal CDFs and all 32 fixed full-vector projections. Keep conservative CDF upper <= .10, mean-bias upper <= .20 reference SD, SD uncertainty interval inside [.8,1.2], reference CDF precision <= .005 and original moment uncertainty. No gate relaxation, clipping, postselection or hidden partial vectors. Save the original density/focus and CDF evidence, exact manifest and payload hashes, strict failures, actual posterior-request and wrapper seconds, separate CPU refresh/evaluation timing and allocated GPU cost. Do not claim a GPU-only timer is end-to-end hybrid latency.

Report the outcome, limitations and costs before choosing any subsequent validation. This is a shared-case development diagnostic, not SBC, global calibration or untouched confirmation.

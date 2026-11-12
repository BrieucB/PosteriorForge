# Controlled SS15 accuracy candidate

The reviewed single-GPU FP16 path is frozen for this scientific iteration. The first pilot uses context seed 20261111, context 4096, held-out case 0 and exactly 50,000 complete accepted 21-dimensional physical vectors. Additional performance experiments are deferred. If the pilot is promising, evaluate all 24 seed/case combinations before claiming scientific success.

For each fixed ten-point mechanical Gaussian block, choose a checksummed orthonormal basis Q spanning the rank3 signal design X. Retain u=Qᵀ(y−offset) and the explicit squared residual energy E=||(I−QQᵀ)(y−offset)||². The three blocks contribute12 features; the three acoustic observations stay unchanged. The first candidate uses raw residual energy, without an added log transform or target transform. All three globals and all 15 observation features remain available at every autoregressive target stage.

The identity ||y−offset−Xβ||²=||u−QᵀXβ||²+E preserves the likelihood's parameter dependence. Noise is inferred, so residual energy and the original ten-observation normalization sigma^-10 are retained. The seven-dimensional residual contributes a parameter-independent observation-measure factor to the normalized summary density; it cancels from the posterior. Four summary coordinates are not treated as four independent Gaussian observations. The transform uses fixed forward geometry and original Float64 observations, with no true parameters or reference-posterior input.

A fixed integer polynomial basis removes QR-sign/rotation ambiguity. Explicit residual squares avoid subtracting nearly equal squared norms. A separate audit checks Float64 likelihood identity and final Float32 feature-casting error. Across three existing4096-row banks and eight held-out observations, the maximum absolute Float64 log-likelihood discrepancy is 4.6566e-10. This algebraic/statistical sufficiency does not guarantee that the frozen estimator will learn an accurate posterior.

The new representation, execution cache and input descriptors have distinct source identities. Matching fresh unoptimized FP32 SS15 conditionals/goldens are compared with the SS15 cached FP32 path. FP16 conditional hard checks and approximation drift remain separately labeled, including strict failures. Raw33 numerical qualification does not transfer automatically. The original mixture/component sampler, wave-local eight-head stage 0 density, FIFO complete proposals, finite-bin inverse arithmetic, unlimited whole-vector rejection and final accepted-row trim remain unchanged; only the wave's observation reshape changes33→15.

The pilot is assessed against the same qualified case reference using the original marginal/joint CDF, mean, SD and support checks. It must retain tails, failures and ordinary posterior density/CDF plots. No checkpoint, world-bank expansion, prior/noise change, context filtering, reference integration or threshold relaxation is part of this candidate.


The matching execution screen preserves its initial failed result: stage 20/member4 exceeded the2e-5 raw-member CDF limit. The new declared policy uses the unchanged unoptimized FP32 route for that component, alongside stage 10/member5. A fresh stage 20 replay passed; a source-bound composite selects the unchanged passing stages0–19 and the new stage 20 while retaining both actual CDF contexts and the failed input. All 21 FP16 approximation strict-drift flags remain failures. This gates the first50000-vector scientific pilot; it is not an accuracy or posterior-equivalence result.


## First fixed pilot result

The first SS15 request completed on one GPU (Slurm 5408971,301 allocated seconds, zero restarts) and returned exactly 50,000 finite, prior-supported complete 21-dimensional physical vectors. Posterior request time was197.740194s; complete wrapper time was285.948231s. This is one accuracy pilot timing observation. The difference between request and wrapper scopes is not attributed to one operation.

Against the same saved qualified original reference and unchanged conservative physical gate:

| Conservative physical metric | Raw33 FP16 | SS15 FP16 |
| --- | ---: | ---: |
| Worst marginal CDF upper bound | .224425 | .109972 |
| Worst of 32 joint CDF upper bounds | .166952 | .064554 |
| Worst absolute mean bias upper bound, reference SD | .445224 | .250365 |
| SD-ratio interval envelope | .900915–1.395516 | .906986–1.120452 |
| Failed marginal / joint / mean / SD checks | 7 /7 /8 /4 | 1 /0 /4 /0 |

The result remains **model_failure**. All 32 joint projections and all 21 SD intervals pass; mu_a still fails the marginal-CDF check, and mu_a, b_m1, a_m3, b_m3 fail mean-bias checks. The mu_a marginal worsened: its CDF upper bound increased from.093531 to.109972 and its mean-bias upper bound from.206695 to.250365. Improvement is not uniform and is not a calibration or equivalence result.

The chain begins mu_a, mu_b, mu_d0. Stage0 mu_a sees all 15 SS15 observations but no generated targets. Replacing only normalized downstream factors leaves its proposal marginal unchanged. Whole-row support conditioning may reweight its returned marginal. The actual run generated 51,200 full proposals, screened 50,395, accepted 50,128 before trim, rejected 267, trimmed 128 surplus accepted rows, and left 805 FIFO rows unscreened. Its observed rejection fraction is 267/50395=.00529814. This estimates population rejection probability; it is not itself a population guarantee. For a fixed proposal law P and support event A, total variation between P and P(.|A) equals P(A-complement), so each marginal CDF displacement is at most that population rejection probability. This does not bound means for the unbounded global coordinates.

The six local blocks are conditionally independent given all three globals and the observations. Within a mechanical block, the quadratic design couples a, b, d0 through the likelihood, and its inferred sigma is coupled to all three. The correlated global prior and the integrated local likelihood couple the globals. Direct graph neighbors alone therefore do not justify scalar conditional feature deletion.

[Ordinary physical density and CDF figures, unchanged-gate assessment and custody hashes](../evidence/accuracy-ss15-v1/pilot-01/README.md) document the complete retained cohorts. All 24 predefined seed/case combinations remain a required diagnostic grid; this first development case is not confirmation. GPU optimization, retention timing, compilation and 1/2/4-GPU scaling remain deferred.

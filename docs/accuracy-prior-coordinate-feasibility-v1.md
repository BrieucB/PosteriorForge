# Prior coordinates: feasibility and ranking, version 1

**PREPARATION / RANKING ONLY. No candidate implementation, accuracy claim, model call, posterior draw, or change to the running fixed grid.**

Built with PriorLabs-TabPFN.

This is a source-based design memo. It uses the declared scientific prior and the installed, pinned inference sources, not reference posterior moments, truth, fitted posterior constants, or new observations from the grid. The current sigma-first method, its native execution policy, and the one-pass exact global refresh remain frozen. Rankings below are hypotheses about representation, not selections made from later grid outcomes.

The most distinctive prior-derived change would be a causal transport that removes the known global-to-local prior dependence, together with an explicitly qualified noise-prior transform. Existing preprocessing already performs empirical target and feature transformations; there is no source evidence that repeating marginal scaling alone supplies this transport. A normal prior in the transported coordinates does not imply a normal posterior, a match to TabPFN's pretraining task distribution, or improved posterior accuracy.

## What the frozen implementation already does

The upstream NPE core stores the supplied simulation parameters and features, concatenates them, and fits each next parameter against observations and the preceding parameter columns. Its optional embedding acts on observations; the inspected core contains no Gaussian hierarchy-aware parameter transport. It calls the regressor's criterion sampler and appends the returned scalar to the next query. See [S1], lines 73–81 and 127–162.

The frozen regressor settings use eight estimators and default inference configuration ([S2], lines 40–46). In TabPFN 2.0.9, the regression target is first standardized with the empirical training mean and standard deviation ([S3], lines 521–528). Ensemble defaults also include both no additional target transform and SafePower ([S4], lines 111–128); the latter is fitted to the one-dimensional target alone ([S5], lines 574–583). This is more than affine scaling. It is not the row-dependent subtraction of a conditional prior mean computed from that row's global parameters.

Feature defaults are quantile-uniform features with originals retained and an appended SVD branch, and a SafePower feature branch ([S5], lines 192–202). The SVD branch keeps a passthrough and appends truncated SVD components after empirical feature scaling ([S6], lines 782–815). Its name “global transformer” means a transformer of the feature matrix, not the scientific three-global prior. The model input encoder additionally supports feature normalization and outlier handling using fitted statistics ([S7], lines 676–708, 721–753 and 771–798). Thus it would be incorrect to describe the existing path as having only raw features or only marginal standardization.

The order matters: empirical target standardization occurs **before** the optional per-member SafePower transformation. The regressor passes the normalized y to the inference engine ([S3], lines 521–545), which passes that same y into preprocessing ([S17], lines 260–269 and 398–407); only then is the per-member target transform fitted ([S5], lines 577–583). Thus SafePower does not rescue an otherwise cancelled positive affine marginal relabeling.

For y' = a y + b with a > 0, fixed predictors, ensemble configuration and seed, the canonical standardized target satisfies

\[
\frac{y'-\operatorname{mean}(y')}{\operatorname{sd}(y')}
=\frac{y-\operatorname{mean}(y)}{\operatorname{sd}(y)}.
\]

Both untransformed and SafePower members consequently receive the same canonical targets. Rescaling the final criterion and applying the inverse affine coordinate map then cancel in the physical distribution too. The actual code adds 1e-20 to the empirical SD and executes finite-precision arithmetic, so this is the structural cancellation, not a promise of bitwise equality or invariance at numerically degenerate scales. It supplies no substantive first-stage density correction for this nondegenerate Gaussian prior. Nonlinear feature transformations and fingerprint behavior need separate consideration if the predictors themselves change.

In particular, the first lower-Cholesky global coordinate is z_G,0 = (G_0-m_G,0)/L_G,00, a positive affine relabeling, while stage-0 observation predictors are unchanged. Global whitening alone therefore cannot directly fix its learned first-stage physical density, beyond the numerical guard/rounding effects just distinguished. Later whole-vector rejection could reweight its accepted marginal; that is not a corrected stage-0 model.

The genuinely distinct operations are prefix-dependent conditional centering: the second/third Cholesky global coordinates subtract earlier globals, and each local coordinate subtracts A_i G + c_i. These are not one fixed affine transformation of that target column. Nonlinear noise-prior CDF normalization is distinct as well. A target transformer fitted only to y cannot evaluate beta_i minus A_i G for each row, and an X-only SVD does not replace the current target by that residual. This conclusion is restricted to the inspected NPE/regressor/preprocessing path, not every possible TabPFN configuration or what its neural weights can approximate.

## The actual prior, rather than a diagonal approximation

Write G = (mu_a, mu_b, mu_d0). The source fixes

\[
m_G=(0.3,-0.2,0.1), \qquad
\Sigma_G=\begin{pmatrix}
1&0.12&0.02\\
0.12&1.2&-0.03\\
0.02&-0.03&0.4
\end{pmatrix}.
\]

The covariance is **not diagonal**. Its off-diagonal correlations are modest. Marginal positive affine target normalization alone is structurally cancelled as above; global whitening's distinct part is the prefix-dependent subtraction for G_1 and G_2, not a new first-global density. This must be separated from possible effects of changing preceding global feature columns at later stages, including their empirical feature transforms and fingerprints. No bitwise invariance of the full SDK is asserted.

For each mechanical block, beta_i = (a_i,b_i,d0_i) has conditional mean A_i G + c_i with A_i = I, c_i = 0, and

\[
D_i=\operatorname{diag}(0.45^2,0.70^2,0.25^2).
\]

For each acoustic block, beta_i is scalar, A_i = (1,0,0), c_i = 0, and D_i = 0.45^2. These are the **conditional** covariances given G, not marginal covariances after integrating G. The source checks these exact constants and the global covariance ([S8], lines 50–71). The prior's sampling and log-density use precisely A_i G + c_i and D_i ([S9], lines 220–233 and 265–274).

All six independent noise standard deviations have an underlying Gaussian with location 0.255 and scale 0.1225, truncated to [0.01,0.5]. Those numbers are not the mean and standard deviation of the truncated law ([S8], lines 57–71; [S10], lines 28–34 and 132–142). They are independent of G and of the conditional coefficient residuals. No marginal covariance estimated from a bank or reference is needed for any of these constants.

## A causal full-prior transport and its inverse

Let L_G and L_i be lower-triangular Cholesky factors of Sigma_G and D_i. In column-vector notation, define

\[
z_G=L_G^{-1}(G-m_G),\qquad
z_i=L_i^{-1}\{\beta_i-A_iG-c_i\}.
\]

Let F_i denote the normalized, truncated noise-prior CDF and Phi the standard normal CDF. For an interior noise value,

\[
z_{\sigma_i}=\Phi^{-1}\{F_i(\sigma_i)\}.
\]

In exact real arithmetic these coordinates are mutually independent standard normals under the scientific prior. This describes a coordinate change of the same prior, not replacement by an independent physical prior.

The causal inverse is

\[
G=m_G+L_Gz_G,\qquad
\beta_i=A_iG+c_i+L_i z_i,\qquad
\sigma_i=F_i^{-1}\{\Phi(z_{\sigma_i})\}.
\]

For a scalar global stage k, decode G_k using only z_G,0:k and the already available lower-triangular rows. An arbitrary symmetric whitening would generally require future global coordinates and is unsuitable for this direct autoregressive inverse. Decode all three globals before any local block. A lower-triangular L_i similarly supports decoding each local coefficient from preceding same-block coordinates; here D_i is diagonal, so the local inverse is especially simple.

This is conditional whitening: using the marginal local covariance A_i Sigma_G A_i' + D_i instead would leave the known G-to-local dependence in the coordinates. A separate coefficient-only ablation would still require the mean subtraction A_i G + c_i; dividing beta_i by its empirical or marginal standard deviation is not equivalent.

The six sigma-first block orders can be retained. Independent prior noise transport does not require future coefficients, and the local coefficient inverse needs the globals already generated. All three global factors must still see all SS15 observations; local factors can retain all three transformed global coordinates, their own block statistics and preceding transformed same-block targets. Given all transformed globals, G is known by a bijection, so the existing block-conditional independence rationale is preserved. A local graph rule that drops one or more globals is not justified by this argument. The current fixed order and masks are explicit in [S11], lines 26–38 and 91–109.

Prior independence is not posterior independence within a block. The likelihood still couples its coefficients and noise, and the posterior in z need not be Gaussian or easy for the checkpoint. This proposal uses no analytic local posterior update.

## Noise-only normalization versus other unconstraining maps

For lo = 0.01, hi = 0.5, location m = 0.255 and scale s = 0.1225, set a = (lo-m)/s and b = (hi-m)/s. In real arithmetic,

\[
F(\sigma)=
\frac{\Phi((\sigma-m)/s)-\Phi(a)}{\Phi(b)-\Phi(a)}.
\]

A noise-only candidate would use z_sigma = Phi^-1(F(sigma)) and leave all coefficient/global coordinates physical. It directly changes the six noise targets, including the first target within each local block. It is narrower to implement and attribute than a full transport, although its finite-precision boundary handling is still a substantive new seam.

Merely setting (sigma-m)/s leaves a truncated normal, not an unbounded standard normal. The map logit((sigma-lo)/(hi-lo)) unconstrains the interval but does not normalize its truncated Gaussian prior. The map logit(F(sigma)) instead gives a standard logistic prior. These are different declared target distributions, not interchangeable names for the same operation.

A coefficient-only change cannot directly alter the sigma_m2 conditional in the present sigma-first order if global/noise coordinates, its training target and retained predictors are held fixed: that sigma stage precedes its own coefficients, while other blocks' local prefixes are masked. It therefore cannot be presented as a direct remedy for the case-4 noise mean failure. Different whole-vector support rejection or different stream consumption can indirectly change the accepted finite cohort; that is a separate mechanism, not evidence of a changed earlier conditional.

## Boundary and floating-point policy must be explicit

There is no globally exact floating-point bijection between all real normal latents and the finite physical interval. The frozen physical prior sampler itself uses truncnorm.ppf followed by clipping to the declared bounds ([S10], lines 111–127). This source does not prove that saved bank values avoid exact endpoints. The current physical support is inclusive at both bounds but rejects every nonfinite coordinate ([S2], lines 57–63).

A new codec must not silently exploit those facts:

- An input latent of positive or negative infinity must be detected **before** normal CDF and physical decoding. Otherwise Phi maps it to 1 or 0, and SciPy's PPF returns a finite interval endpoint that the old support accepts. SciPy's endpoint branch is explicit in [S14], lines 2293–2312.
- A finite latent can also make Phi round to 0 or 1; an interior physical value can yield a rounded prior CDF endpoint. Exact physical endpoints encode to infinite normal coordinates. Each condition needs a counted, explicit failure or representational policy; checking only that the latent was finite is insufficient.
- Do not clip probabilities to epsilon or nextafter endpoints, move stored physical values inward, remove training rows, replace nonfinite targets, resample individual coordinates, or silently add endpoint atoms. Such operations are additional model/sampling changes.
- Stable lower/upper log-tail calculations and log-probability inverse-normal routines are plausible numerical tools, not a proof of a global round trip. SciPy already uses log-mass/log-survival and ndtri_exp branches for truncnorm ([S15], lines 10327–10384); the composed transform still needs endpoint and saturation checks.
- The conservative initial codec contract would fail the entire qualification/request on unsupported training endpoints, nonfinite latents, saturation, or a failed declared round trip. If a later design instead emits an invalid complete physical proposal for the original whole-vector rejection loop, that extra finite-precision rejection rule must be declared and measured. It must not be called an unchanged ideal pushforward.
- Compute a source-defined transform from the original saved Float64 physical bank, then make one declared model cast. Report physical-to-latent-to-physical Float64 errors separately from the additional Float32 target/prefix errors. No claim of Float64 byte identity after a Float32 model cast is appropriate.

The existing finite-bar sampler deliberately retains u=0/zero-probability NaNs for later complete-vector rejection ([S12], lines 224–252). A nonlinear decoder must preserve their invalidity rather than make them look like valid physical bounds.

Sampling a finite-bar latent value and then applying its inverse map gives the intended pushforward of that learned latent distribution. Transforming only the bar borders and then sampling uniformly inside each *physical* bin would instead change its within-bin measure for a nonlinear inverse. The SDK's own target-transform reconstruction uses transformed borders and CDF translation ([S3], lines 670–710); its conventions remain part of the learned output path, not a substitute for an explicit new coordinate decoder. No additional importance weight is needed for drawing through a bijection. Any physical log-density claim would require the affine and noise Jacobians, and cannot reuse an uncorrected latent score as a physical density.

## Exact global refresh must hold the physical locals fixed

The frozen joint-global kernel copies complete physical rows and replaces only columns 0–2 ([S13], lines 119–128 and 139–160). For a transported sampler, decode to the original physical 21-vector **first**, restore the original physical order, and preserve all physical coefficient/noise values throughout the final exact G3 update.

Keeping the old z_i and decoding again after a new G would give

\[
\beta_i^{new}=\beta_i^{old}+A_i(G^{new}-G^{old}),
\]

which changes the locals and is not the declared Gibbs kernel. If transformed state is retained after that update, recompute z_i = L_i^-1(beta_i_fixed-A_i G_new-c_i), or discard the old transformed coordinates. Do not reinterpret them under the new globals. The fixed physical 18-column identity remains an independently checkable requirement.

## Preparation ranking and required evidence

This ranking concerns distinct source-based justification and engineering scope. It does not authorize a new candidate or predict a gate outcome.

| Rank | Preparation option | Rationale and limitation |
|---|---|---|
| 1 | Full causal prior normalization | The conditional local mean subtraction is the largest structural distinction from existing marginal preprocessing. Noise PIT also changes the early sigma targets; lower-triangular global whitening completes one coherent prior transport. Highest codec/prefix/finite-tail validation cost, with no posterior-normality guarantee. |
| 2 | Noise-only prior CDF to normal | A smaller, interpretable prior-only intervention that directly changes all six sigma targets. It leaves global/local prior dependence for the network and still needs the full boundary contract. A useful isolated alternative if full transport is too large to qualify. |
| 3 | Further observation-coordinate engineering | Current SS15 retains the exact real-arithmetic Gaussian likelihood dependence, including residual energy and sigma normalization ([S16], lines 115–137 and 151–170). A further invertible rescaling/rotation could change model geometry, but adds no information; discarding residual energy loses information about unknown noise. There is no source-only reason here to prefer another ad hoc observation compression. |

Global-only whitening is a lower-priority sub-ablation of rank 1. Its first coordinate is affine-equivalent in the canonical target path; only the second/third coordinates' prefix-dependent centering and downstream changed predictor columns add distinct structure. The covariance is not diagonal, so those later operations are not wholly redundant, but they are a weaker distinction than removing the conditional local mean. Coefficient-only whitening is more distinctive, but cannot directly affect an earlier unchanged noise factor. The rank-1 full transport likewise makes **no direct stage-0 first-global density correction**.

These transports use only declared priors and bank parameters and require no extra simulator worlds or likelihood calls. They are therefore feasible with an expensive simulator whenever its prior exposes the necessary conditional maps/CDFs. For a different hierarchy whose prior has no tractable causal inverse, the present derivation does not automatically apply. Reference-fitted rotations, centering, calibration constants, or endpoint tolerances are outside this proposal.

Any future implementation needs a new source/version/cache identity binding the full codec, order, mask, original physical bank, transformed targets/prefixes, and decoded physical output. It cannot inherit the current SS15 or sigma-native numerical pass. Matching unoptimized and cached conditionals must receive identical transformed training/query inputs and target labels; FP32 structural checks, FP16 drift labels, physical target bindings, member/mass/CDF checks, and sampler/decode tests remain separate. The old stage-index native policy and its physical target labels must be requalified under the new coordinate geometry rather than assumed valid. Finite prefixes are not a proof of all off-prefix joint behavior.

After any independently authorized new proof and fixed 50,000-vector pilot, evaluation would still use the unchanged original physical support, reference custody, physical criteria, and complete fixed grid. Existing failures and the distinction between development selection and validation must remain visible. This memo makes none of those changes.

## Verified primary-source inventory

Checked read-only on Karolina on 2026-11-12. Repository root is /home/it4i-bbenvegnen/workspace/PosteriorForge. Paths below identify the actual files whose bytes were hashed; line numbers in the prose refer to those bytes. The installed SDK root is /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/lib64/python3.11/site-packages/tabpfn. SciPy references identify the installed reference environment's 1.15.3 implementation; they do not assert that another installed version has identical source bytes.

| ID | Exact file path (relative only where a root is defined) | SHA256 |
|---|---|---|
| S1 | src/posteriorforge/upstream/npe_pfn/npe_pfn.py | 29517328b3ff23e2e8040d19f33e79adef5fc7f91db2cffccf60234344399a85 |
| S2 | src/posteriorforge/inference.py | 95366efc4871b96c3a1a3f19414ed1fd77d058f02135f4afd95a64b5918b3d3e |
| S3 | SDK root/regressor.py | bbb7d6dec265b18a30a54ff28719c957bc05d4164f693f76d998380132d2c46e |
| S4 | SDK root/config.py | d6ac4523f5553b469b49a1e2b954bef6828d52dc23163d67f62656556c36cc8e |
| S5 | SDK root/preprocessing.py | fba9df553f3f7d190972a531c5627f09b88e8b8f537d298df7083f7dd972843d |
| S6 | SDK root/model/preprocessing.py | 7da6b7c6d060a396b4aa7024fd5ac2c9bfddbc7d62029b113962cecbdaca6787 |
| S7 | SDK root/model/encoders.py | e16e03c320c69dcd2e9823925d59fa20a0c52008d8cc77ca961c919c7221213e |
| S8 | src/posteriorforge/science/analytical_benchmark_v2.py | 8d5f286f6d8512391dea2ae036313bec05782d28c7803f336490cbcdce3bd99f |
| S9 | src/posteriorforge/science/hierarchical_exact.py | db001593cc7307b1290b981622705961c5953001a6c1245fae5ec779938e9687 |
| S10 | src/posteriorforge/science/hierarchical_noise.py | cbf0b8c956362a5eaa31923e9b94a3db3e3ede2072603add51e78105b8c0cbf0 |
| S11 | src/posteriorforge/execution_ss15_sigma_first.py | d48d1638ef9e540bdd629d321fba657af4db757700b1092fc58fb9ded1f79ad2 |
| S12 | src/posteriorforge/execution_mixture.py | e71f43b73ae8259c20b5d7b52feccc7c851e9a2fbb86d869bc0a5f7f3af792a1 |
| S13 | src/posteriorforge/conditional_globals.py | e346e25bc9825abf016c2bd75addaf2b025d3e3d94e1a0ea6f6e31dd02997a43 |
| S14 | /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/reference/lib/python3.11/site-packages/scipy/stats/_distn_infrastructure.py | 9df9372d87b6e8f8d9ceb4ee83e64328a088faab264ec4027e3f7df9f47d4efc |
| S15 | /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/reference/lib/python3.11/site-packages/scipy/stats/_continuous_distns.py | 4011a3b7e9308639b3537f9766d9e42525f2e0a0ba88893b0defe4dd364215fe |
| S16 | src/posteriorforge/representation_ss15.py | 46323c0b94301c2f7e678538881d5cbbdd8bc52c62a60021798176bffe81eb85 |
| S17 | SDK root/inference.py | 62470bbfdd4879dd52db86ea7a6e53ae41fbcc22379bfb2866ad906adc71a0bd |

This source documentation intentionally contains no new experiment figures or generated cohort results.

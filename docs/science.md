# Frozen analytical benchmark

PosteriorForge uses the baseline quadratic/scalar inferred-noise Gaussian
hierarchy: 21 parameters and 33 scalar observations, ordered as declared by
`build_problem().parameter_order` and `DATASET_IDS`. It is an analytical
prior-predictive provider; it is not a learned physical surrogate. The global
Gaussian population has three coordinates. There are 12 conditionally Gaussian
local coordinates and six independent bounded noise standard deviations.

The six modules in `src/posteriorforge/science/` are a byte-for-byte relative-import
closure from BayesArena revision `bc455e0e2389307fb72dac7dd11b62b17b58d56e`,
which attributes MesoUQ revision `f0df511e4273032fc2619a406f33f0a009703b34`.
The MIT notice is retained beside the closure and source/target SHA256 values are
in `provenance/science-source.json`. No private package is needed to install or
run this code. Original scientific schema strings are retained deliberately:
the fingerprint is `216a240a9fb347348f433804b78eeebb6e040332db6d2fd40b164a7ed74b7966`.
PosteriorForge bank, reference, and inference artifacts use their own identities.

`banks.prepare_banks` creates three complete4096-world joint-prior contexts with
seeds20261111,20261112,20261113 and nested256/1024/4096 prefixes. Eight held-out
worlds use the distinct seed20261114 and are shared by every context/variant.
Held-out selection rejects entire prior worlds until each population coordinate,
conditional local residual and noise quantile lies in its coordinate-wise
central90%. Observation errors are then sampled. This selection changes the
held-out population, never the inference prior. No normalization/validation banks
from the legacy campaign are generated. NPZ theta/obs arrays remain float64;
JSON manifests bind shape, order, numerical priors/noise, seeds, hashes, and cost.

Stream derivation uses NumPy SeedSequence and explicit hashed role namespaces.
The source indexed draw algorithm is retained, with PosteriorForge context and
held-out roles; full-prior blocks have512 worlds and exact random access.
`banks.derive_seed` additionally separates role/case/rank/repetition for compute
streams. Eight cases are a diagnostic screen, not a global SBC certificate.

# Reference contract

`reference.prepare_reference` requires a billed Slurm allocation, even for its
CPU-side independent quadrature. It uses float64 JAX GPU or NumPy CPU algebra
and checks conditional results against independent33-dimensional observation
covariance algebra. The unchanged reference integrates six noise coordinates
using four scrambled Sobol banks at powers16,18,20,22. Gaussian physical
coordinates are integrated analytically. Scramble banks and their hashes are
retained and reusable; exhaustion and timeout remain inconclusive.

The retained refinement gate requires both maximum final-scramble CDF spread
and change from the previous power at most0.005. Its53 fixed directions contain
21 prior-scaled axes and32 complete-vector projections fixed by seed763019.
A denser513-point table over[-8,8] is independently rechecked at the last two
retained powers for candidate CDF comparisons. Dense CDFs, analytic mean/SD,
directions, queries and their hashes are saved together. This is a finite-grid
numerical precision contract, not a theorem for arbitrary multivariate metrics.
Candidate validation bounds differences between grid points and at both tails.

The independent check integrates global coordinates by Gauss-Hermite and each
noise coordinate by Gauss-Legendre with separate global/noise refinement corners
(default20x128,40x64,40x128). It must agree with the retained Sobol mixture in
noise CDFs<=0.005, log evidence<=0.01, and absolute global mean/covariance and
noise mean errors<=0.001 in physical units. These conservative declared checks
can return inconclusive; no failed/exhausted reference yields scientific pass.
The public manifest records each condition separately. Production refinements
and dense tables must run within the same200 allocated GPU-hour cap.

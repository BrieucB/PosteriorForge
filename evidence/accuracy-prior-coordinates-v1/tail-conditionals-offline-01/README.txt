Built with PriorLabs-TabPFN.

NEW CONDITIONAL EVALUATIONS AT SAVED PREFIXES — DIAGNOSIS ONLY
Counterfactual known1024/N1 batches and all8 heads, using authenticated
reconstructed Float32 U values. Historical selected member/grouping/density
is NOT recreated. No accepted50000 cohort, G3, live source or policy change.

Main finding at the two focal reconstructed U values:
stage15 N1024 stock noise-decode failures8/8 original,7/8 candidate; exact-
normalized EXISTING-F32-weight reference has0/8 decode AND noise-only RT32/
RT64 failures in both arms. Stage15 N1 normalized counts are also0/8.
stage19 N1024 stock outsidefinitegrid7/8 both arms. Normalization removes
that extrapolation, but normalized quantiles STILL fail strict noise decode
3/8 original and6/8 candidate; noise-only RT32 AND RT64 fail8/8 BOTH.
Therefore normalization alone is NOT a cure for the unchanged codec domain.

RT counts concern nearestFloat64 reference quantiles and the two frozen
latent tolerances. They are scalar noise diagnostics, NOT full d8 qualification
and NOT an implemented normalized-Float32 sampler. All16 original/candidate
member comparisons pass existing thresholds (maxCDF about6.56e-7), yet
near-one quantiles diverge. No threshold waiver or posterior accuracy claim.

The reference normalizes actual captured Float32 softmax weights by their
EXACT sum with integer/Fraction arithmetic. It does not recompute softmax64,
clip U, move endpoints, or use a scalar-row recomputation of productionp.
Capturedp/CDF/source intermediates come from the same wholebatch return-
observer computation, byte-parity checked against unchanged frozene71.
u1 is separate OUT-OF-CONTRACT stress; frozen code rejects it. The reference
defines the mathematical endpoint extension, not a production-policy fix.

Figure: tail-density-cdf-diagnostic.png
Predeclared member0/original+candidate/N1024; both saved prefixes, normalized
density, capturedCDF edges versus normalized cumulative, and quantiles.
Historical scalar stars do not imply historical density recreation.
All8 metrics, decode/RT counts and both arms/N labels remain in JSON records.
No invented representability threshold or fullposterior interpretation.

Ranked FUTURE choices only:
1 New normalized cumulative/inverse of EXISTINGF32 weights, not softmax64:
  addresses observed support extrapolation but is insufficient by itself.
2 Explicit whole-ROW representability/RT rejection q|D with counted mass and
  fresh qualification: minimal domain-policy change, NEVER salvage/filter
  this wave. Requires reweighting/moment/support review, no hidden redraw.
3 Broader mixed/physical-noise representation: new scientific candidate.
Decoder-only logCDF/more accurate intermediates cannot make historical
real inverses into correctly rounded interiorFloat64 outputs. No endpoint/
predecessor substitution, clipping, tolerance change or candidate implemented.

All48 focal NPZ files are byte-copied rows/full per-member borders, with
source wholebatch file SHA/row index and derived array hashes. No recalc or
cast. Full wholebatch archives remain onKarolina. No model weights or raw
training arrays are duplicated here. Focal NPZ payloads stay OUTSIDE Git.

Source-checkout CPU reproduction (no neural calls):
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src INFERENCE_PYTHON scripts/analyze_prior_tail_conditionals_v1.py --capture-manifest CAPTURE/manifest.json --capture-sha256 cf11be3c542eb11c7ed9423845eb1f80fb046df59a2b85254cb9f20d25663461 --output FRESH_DIR
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src INFERENCE_PYTHON scripts/plot_prior_tail_conditionals_v1.py --report FRESH_DIR/report.json --report-sha256 ITS_SHA --output FRESH_DIR/tail-density-cdf-diagnostic.png
The immutable original report98ecc, actual arrays and scripts are hashed.

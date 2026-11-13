Built with PriorLabs-TabPFN.

SAVED48FOCAL CPU POSTPROCESSING DIAGNOSIS ONLY — NOT A NEW CANDIDATE
Exact reportcbc8 is immutable. No GPU/NN/model constructor/world/posterior/G3,
live sampler/RNG draw, refit, border reconstruction or production change.
Known1024/N1 batches/all8heads remain counterfactual, not exacthistorical
member/grouping/density reconstruction. Both reconstructed Float32 U values
are evaluated on every saved density; ownstageU is identified explicitly.

SOURCE ORDER AND A/B/C MEANINGS
raw_model_logits is the capture callback's ALREADY temperature0.9-divided,
optionalcancel-mask(-inf)-adjusted Float32 input. It is not unscaledrawNNoutput.
NO secondtemperature division. Average_before_softmax=False is SOURCE-bound
(inference.py/settings and SDKargument), not a saved runtimeboolean. Selected
one-head route is stack.mean(identity)->GPUlog->CPUcopy->CPUsoftmax.
from/to borders are captured standard-coordinate Float32bits promotedF64,
not refit; final inverseborders remain captured prior-latent z target units.

A actual captured FINAL CPUF32softmax weights, exact-normalizedreference.
Actual GPUpretranslationsoftmax and preclipCDF/diffs were NOT captured.
Do not infer zero preclipnegativecounts/clippedmass from postclamprawP.
ActualGPUF64mass, recomputedCPUrawPsum and finalCPU-Psum/CDF are separate.

B NEW CPU TorchF32softmax replay (not actualGPU!), then TorchF64CDF/diff/
log/CPUsoftmax, exact-normalizedinverse. B shows huge CDFupperclipping and
zero endpoint-tailmass: misleading lowerquantiles, not evidence of improvement.

C NEW CPU TorchF64softmax of SAME capturedadjustedF32logits, TorchF64source-
order correctedCDF/diff, log, CPUsoftmax, exact-normalizedinverse.
All primary C operations are TorchCPU64. An unused NumPysoftmax64 convenience
definition exists in source but is NOT executed. Root's separate NumPy64
crosscheck is corroboration, not silently identical arithmetic. Directraw
normalization and F32output-boundary CPUlog/softmax are separatelyrecorded.

CDF follows cumsum(p)-p, NOT a prependedprefix; corrected parentheses;
relativeCDF share sourceclamp[0,1], boundary0/1 overrides, CDFclip[0,1],
forcedfirst0/last1, adjacentdiff, clamp_min0. NEW B/C clippingcounts refer
only to those NEW arithmetic computations, never unseen originalGPUcounts.

PRIMARY FINDING
C stage19ownU quantiles about7.87654..8.21560 in both arms: endpointnoise
decode passes all8, yet noise-only RT32 AND RT64 fail8/8 BOTH. Positive-
endpoint-tailmass remains about4.56e-7..5.35e-7. Improved numerical
postprocessing alone does not solve the unchanged representability/RTcontract.
The endpoint threshold is not the fullRTdomain. All48/BOTH-U metrics remain
in exactreport; compactsummary preserves groups/clipping/provenance.

Prospective ranked choices only: numericalC with explicitnewlaw/source/
qualification; explicit wholeROWrepresentability+RT rejectionq|D/counts/
qualification; broader mixed/physical-noise representation. Decoder-only
logCDF/moreaccurateintermediates cannot recover historical interiorF64.
No endpoint/predecessor replacement, clippingrepair, tolerancewaiver,
oldwavefiltering/recovery or actualcandidate implementation is authorized.

Reproduce from checkout in CPU INFERENCE environment:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python scripts/diagnose_prior_tail_postprocessing_v1.py --analysis INPUT/tail-conditionals-offline-01/report.json --output FRESH_DIR
CapturedinputarraySHA/dtype and RNGbefore/after are checked. Derivedweights
NPZ arrays are NEW CPUreferencevalues, NOT actualGPUcapture. They remain
outsideGit; no modelweights/rawtrainingarrays are distributed here.

# Original support and scientific-gate contract: source trace

Built with PriorLabs-TabPFN.

All six physical noise coordinates—sigma_m1/m2/m3 at slots 6/10/14 and
sigma_a1/a2/a3 at slots 16/18/20—share original bounds **[0.01,0.5] with both
endpoints included**. The provided original Torch support checks finite complete
21-vectors and inclusive >=/<=. The unmodified NPE wrapper uses this check first;
its finite-log_prob fallback runs only if check is unavailable. The original
accept/reject routine filters whole rows, concatenates in proposal order and
trims to the requested count without a dtype conversion. Actual requests set
max_iter_rejection=None, so its optional early-exit escape is inactive.

The actual scientific assessment path is `gate -> support_mask -> contains`
and `assess_samples -> sample_integrity -> physical mean/SD and marginal/joint
CDF metrics`. Scientific coordinate.contains and sample_integrity bounds also
use inclusive >=/<=. **No `to_unconstrained` call occurs in gate.py, validation.py
or inference.py**; the strict-open generic logit helper is a separate analysis
coordinate map, not a hidden part of this sample gate. The original gate limits
remain count 50,000, CDF 0.10, bias 0.20 reference SD, SD ratio [0.8,1.2], and reference
precision 0.005. No criterion or numerical setting changes here.

Dtype scope matters: Torch support.check does not cast its input; log_prob
moves values to NumPy and problem.log_prior coerces Float64, with an explicit
closed contains mask overriding raw prior density outside support. The baseline
writer later casts returned values to Float64 and rechecks support. Scientific
sample_integrity preserves initial sample dtype, then valid metric calculations
use Float64. Therefore raw SciPy prior density, Torch support, and the gate-facing
Float64 mask must not be collapsed into one finite-precision observation. The
independent endpoint receipt owns actual endpoint/adjacent-value evaluations.

The frozen d8 codec is deliberately different: physical noise encode/decode
requires **strictly inside** 0.01 and 0.5, then applies finite-latent and numerical
round-trip admissibility. Original prior/gate support can accept a boundary
which this finite coordinate representation cannot encode or accept. A supported
interior can also fail the codec's numerical round-trip condition. Neither is
an automatic scientific-support rejection or permission to relax/round/repair.
The strict generic logit helper and d8 codec are separately identified in the
receipt; no prior or gate semantics are relaxed or changed.

`source-contract.json` pins exact complete source-file hashes and line excerpts
for all paths, plus the source-only all-six table and dtype/call-graph trace.
No scientific arrays are evaluated, no constructors or transforms run, and no
samples, worlds, posterior/reference integrations, RNG/GPU/NN/G3 operations,
production edits or changes to previous evidence occur. `SHA256SUMS` excludes
itself.

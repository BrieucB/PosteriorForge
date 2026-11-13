# Frozen postprocessing route and capture availability

Built with PriorLabs-TabPFN.

This is source/metadata inspection only, bound to closed capture `cf11be3c...`.
No arrays are evaluated and no constructor, transform, preprocessing, model,
RNG or GPU operation runs. Exact line excerpts and complete file SHA-256 values
are in `route.json`; earlier packages and frozen implementations stay unchanged.

**The archive name `raw_model_logits` is not untempered NN output.** The SDK has
already applied `output.float()/0.9` and any target-transform cancellation mask
before the capture hook at translator entry. Applying 0.9 a second time would
be a different computation. Actual pretranslation GPU softmax weights were
**not** saved. The saved `probabilities` are the later actual CPU FP32 softmax of
the returned selected-component logits, captured in the same whole-batch inverse
calculation. Actual GPU translation returns are separately saved as
`raw_translated_probabilities`; all 48 member calls have Float32 input logits,
from/to/API borders, translated weights, returned logits/probabilities/CDF,
and Float64 raw masses in their descriptors.

The corrected finite CDF computes softmax, **cumsum(p)-p**, bucket lookup and
`(ys-left)/width` interpolation with a share clamped to [0,1]. It forces outer
source CDF values 0/1 and clips CDF[0,1]. The translator additionally forces the
first/last destination CDF to 0/1, takes adjacent differences and clips negative
weights to zero. **It does not explicitly normalize the translated mass.**
This subtraction is distinct from the inverse's prepended-cumsum lookup.

No explicit runtime `average_before_softmax` field was saved. The frozen,
source-bound policy is **False**: `inference.py:40–46` sets it, the original and
cached paths retain it, and the SDK constructor stores it. With only the
selected member present, the supported route is raw translated weights →
**one-member stack.mean(0) → GPU log → CPU copy → CPU softmax in e71**.
There is no additional GPU softmax in that active False branch. The SDK True
branch would add `log.mean.softmax` before its final log, but must not be
presented as observed here. This is an explicit source derivation, not an
invented direct runtime-attribute capture.

`from_borders` maps the member's inverse target transform back to standardized
regressor-target units; `to_borders` is the canonical checkpoint grid in those
units. Saved API `borders` equals that grid scaled by actual target SD and mean,
so here it is in **prior-latent z target units**, not physical sigma units.
The active CDF and inverse use finite linear-uniform bars; the distribution
class name and separate half-normal density formulas do not change this path.

For the authorized saved-data diagnostics, B must label its **CPU-F32 softmax
replay** limitation; it cannot claim actual archived GPU softmax parity. C must
use the **same already temperature-adjusted captured F32 logits** in softmax64.
A higher-precision translation followed by normalization is a diagnostic
arithmetic choice. Retain the posttranslation log/CPU-softmax layer distinction,
with no sampler, production, support, threshold or coordinate-policy change.
`SHA256SUMS` excludes itself.

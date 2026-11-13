# Prior-coordinate FP64 proposal policy v3

Status: CPU implementation and validation preparation only. This document does not
authorize neural execution, GPU allocation, a posterior request or a G3 update.
Any future GPU screen requires new Root authorization after committed source,
input, protocol and CPU-test closure.

## Fixed science, new execution identity

The original prior, all 21 physical parameters, fixed SS15 observations,
sigma-first order and six-block masks remain unchanged. The eight-member checkpoint
remains SHA2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736.
FP32 neural execution preserves0199: all8 original stock members at ordered
stages6/10/14 (physical slots5/9/13), cached execution at the other18.
There is no FP16, model/feature/bank change, reference-fitted constant, analytic
local update or multi-GPU method.

V3 is a NEW numerical execution/proposal-domain policy. It starts with zero
numerically qualified pairs and an empty production proof registry. A v1/v2
screen, old case2 request, old24-point result, failed v2 case0 request or
counterfactual probe cannot activate it. Old raw statuses and missing-footer
facts stay unchanged.

PairSpec402 and dual-CPU panel receipt6f35 are data ancestry only: the same
3x4096 original banks and case-specific saved pure FIRST1024 prefixes.
No bank or world is regenerated. The first future screen is seed20261111/case0,
repetition0, using the authenticated original Float32 model inputs and labels.

## Policy C probability path

The SDK neural prefix stays Float32 through division by temperature0.9, inverse
target-border handling and cancellation. CPU Float64 processing starts immediately
before canonical translation; temperature is not applied twice.
Actual adjusted Float32 logits, pretranslation probabilities, translated raw
weights, normalized API probabilities and represented sampling intervals have
separate capture fields. Adjusted logits are not untempered neural logits.

New execution_probability_v3.py performs CPU softmax, corrected finite CDF
translation, canonical weighting and inverse arithmetic in Float64. Its active
descriptor names this policy. It cannot describe active v3 probability execution
as legacy parentheses-only/all-FP32 processing. Legacy CDF descriptors remain
separate preparation/source guards.

The returned criterion promotes the ORIGINAL fitted Float32 criterion border
bits to Float64; target scaling is not recomputed in Float64. Target mean/SD,
fitted transforms and neural normalizers retain their original source/dtype.

Each head retains raw_weights64 and actual Float64 raw mass BEFORE normalization.
Original raw-mass head acceptance and its envelope stay unchanged.
normalized_raw_probabilities64 is ordinary raw/sum normalization.
API probabilities64 is actual softmax64(log RAW weights).
sampling_cdf64 is cumulative API probability divided by its terminal cumulative
value, with last edge exactly1. sampling_interval_weights64 is the difference
of these actual normalized CDF edges and is the inverse's represented measure.
These quantities are separately recorded and their rounding difference quantified.

Stage0 averages all8 actual RAW64 canonical members before the shared API/CDF
normalization path. It does not average separately normalized heads.
Positive weights that collapse in a finite CDF representation are counted with
their masses; there is no threshold-based tail deletion.

The private component stream keeps original CPU Float32 U values, source roles,
seed derivation, ordering and row reuse over head retries. Policy C widens supplied
U for inverse arithmetic and draws no replacement values. It uses represented
CDF-interval denominators, without clipping U, fractions or sampled parameters.

V3 explicitly extends U=0 to the left border of the FIRST positive CDF interval.
This replaces legacy leading-zero-bin0/0 NaN and is tested/reported as a NEW
numerical edge policy. Positive-U plateau ties use left search. U=1 is rejected;
zero-mass intervals are never divided. No redraw, output repair or inward
parameter movement is introduced.

Generated latent stage returns and wave storage retain the ORIGINAL numpy.float32
contract. Float64 inverse results pass through the original explicit Float32 stage
cast; this is not a Float64 latent-wave policy. Neural query prefixes therefore
stay Float32. Only decoded physical proposals and physical FIFO output are Float64.
The qualifier distinguishes raw inverse arithmetic from its declared stage-storage
cast and runs proposal-decoder checks on that stored Float32 result.

## Physical decoder and support

New prior_proposals_v3.py preserves the d8 causal lower-Cholesky global inverse,
conditional-local inverse and log-tail noise inverse formula. Training uses the
EXACT strict d8 encode_for_model alias and identical original model-input bytes.

Original physical noise support is CLOSED Float64[0.01,0.5], separate from d8's
stricter training-coordinate domain. Finite inverse outputs, including represented
physical endpoints, are retained. No value is clipped or moved into support and
no coordinate/row is locally filtered. Finite out-of-support complete proposals
reach the unchanged original whole-vector support check, which owns rejection
and counts. Float64 physical output avoids the observed Float32 lower-bound seam.
Nonfinite latent inputs fail the request/precision check; they do not become
accepted proposals. Legacy zero-U NaNs remain old-API diagnostics.

The decoder replaces universal tiny NOISE-latent roundtrip admissibility with
conditioning-aware physical/probability interval evidence. The original prior-unit
physical inequality retains coefficient256*eps64. Exact rational interval roots
and inward Float64 VALIDATION probes never modify a proposal. Prior log-CDF/log-SF
evaluations are ordinary SciPy Float64: they are not certified directed probability
bounds or a universal physical-error guarantee. Finite CPU proof scope and
conservative rejections remain explicit, with zero added probability tolerance.
Affine source guards remain unchanged. No reference/posterior outcome fits these
constants and no conditional numeric tolerance is widened.

Any later authorized exact physical G3 update replaces only the three globals;
the other18 physical columns remain byte-identical. Old latent locals must not
be redecoded under refreshed globals.

## Fresh all21 conditional screen

New qualify_prior_coordinate_pair_v3.py has its own schema, protocol and source
proof. It compares the ORIGINAL unoptimized FP32 neural path against the candidate
FP32 forward routes, with BOTH arms explicitly using Policy C for matching.
Original constructors, eval mode, weights, fitted context, transforms and all8
member lists are protected. Native cold/warm and cached prepared-state scopes
remain distinct. Planned N1 is an actual comparison, not an undisclosed warmup.

Each stage0..20 evaluates all8 heads on identical direct N1, fixed1024 and stress8
inputs. Actual masks/target bytes, FP32 neural witnesses, Float64 probability,
mass, criterion and inverse evidence, source/RNG/restoration and counters are
recorded. Hard-invalid control/source/input/state evidence stops before candidate
execution. Numeric failures are retained across all21 when hard evidence is valid.
Partial or missing coverage cannot pass.

Existing conditional limits remain unchanged:
- absolute normalization error1e-5;
- maximum CDF error2e-5;
- maximum row total variation1e-4;
- same-input inverse quantile error2e-6 times the declared criterion span.

Raw weights, API probabilities and represented sampling intervals are checked
separately. Normalization cannot hide an upstream raw-mass or execution failure.
Same-input inverse replay includes production Float32 u_max and the new U=0 rule.
Independent Float64 postprocessing replay distinguishes actual captured values
from CPU recomputations. Decoder engineering checks use predeclared scalar rows
and do not claim all-production-domain coverage.

Legacy Float32 API diagnostics are separate from the v3 execution gate.
Same-adjusted-logit CPU replay names its platform/precision and is not an
independent extra neural control or transferred qualification.
The screen produces no posterior vectors or reference integrations.

Original SCIENTIFIC POSTERIOR gates also remain unchanged: all21 physical
marginal CDF criteria, all32 fixed projection criteria, reference-scaled mean/bias
and SD interval criteria. A conditional pass gives no posterior qualification
or equivalence claim. No scientific accuracy gate is relaxed.

## CPU closure and activation

Both locked environments test source/dtype/mass/inverse contracts, wrong-pair,
proof and input rejection, no old credit, and old source/global immutability.
CPU preparation uses no neural constructor/forward, GPU, world/posterior draw,
RNG-stream replay or G3 update.

A future pass requires its OWN completed same-pair artifact: all21 hard/numeric
predicates replayed, exact source/Git/data bindings, complete neural/probability
counters, default-RNG evidence and final restoration.
Production remains empty until Root binds that actual v3 artifact.
Only then may Root separately authorize one exactly50000-row request under the
original full-vector support/FIFO/sampler law.

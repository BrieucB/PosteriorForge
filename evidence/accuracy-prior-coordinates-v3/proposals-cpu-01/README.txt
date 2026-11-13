V3 finite physical proposals: CPU evidence
Built with PriorLabs-TabPFN.

Scope and status
This NEW decoder is a proposal contract, not an acceptance rule. It starts with
ZERO neural-qualified pairs and transfers no v1/v2 proof credit. Only23 declared
scalar cases and tiny deterministic engineering vectors were exercised. No saved
51200-row wave was decoded/recovered; no model/NN/GPU/world/posterior/G3/RNG draw
occurred. The former d8 code/formulas/globals and failed requests stay unchanged.

Training versus proposal decoding
Strict training encode_physical/encode_for_model are EXACT frozen d8 aliases;
the original finalFloat32 cast and endpoint/nonfinite/roundtrip failures remain.
The proposal inverse reuses the exact frozen d8 noise and causal-prefix
CODEOBJECTS in new private globals. Only the output strict-noise-domain validator
is replaced by a finiteness check. No finite value is clipped, perturbed,
nextafter-adjusted, replaced, redrawn or omitted. Every row reaches the original
Float64 FIFO/support. Computed endpoints are retained; computed finite values
slightly outside physical bounds remain proposals for original whole-row
rejection. This decoder itself never accepts a row. Genuine nonfinite input,
arithmetic, or precision validation hard-fails the request.

Conditioning-aware precision validation
The original noise physical inequality is unchanged:
 |computed-ideal|/s <=256eps64*(1+abs((ideal-mu)/s)).
Its exact rational roots give the permissible interval of ideal physical x for
each computed y. Floating validation probes are directed INWARD to that interval.
Those validation-edge adjustments NEVER touch proposal/output bytes. Input Normal
lower/upper log-tail probability must lie in the original prior-CDF image of that
physical interval; no extra probability tolerance is introduced. The old universal
absolute noise latent-roundtrip bound is not imposed because physical quantization
makes it unattainable in the documented tails. Original affine latent/physical
checks remain. The transport is explicitly not lossless near the bounds.

Independent precision evidence and limits
Both locked NumPy/SciPy environments pass the same23 scalar cases: all16 saved
Policy-C stage19/N1024 own-U values plus-205.2463,-13.8767,-8,0,+8,+13.8767,+205.2463.
The independent100/180-decimal reference uses exact binary source constants and
a rescaled positive boundary-gap integral, so it does not subtract an invisible
~1e-9150 gap from an endpoint. All physical errors are below the SAME original
F64 physical budget; maximum error/bound0.004597413, maximum44.36ULPs. This is
bounded scalar evidence, NOT universal correct rounding and NOT certified
interval arithmetic. The physical formula can be inaccurate in ULP units near
a bound while still well within the declared prior-unit physical error budget.

Reference environment has no mpmath installed. Its scalar proof explicitly loads
the existing inference-env mpmath source package BY FILE (recorded version and
full .py SHA map); no inference site-packages is put on sys.path, no environment
is modified, and reference NumPy2.2.6/SciPy1.15.3 remain active.

Tests and receipts
36/36 focused tests PASS in EACH locked environment, no skips. Tests include
unchanged theta->latent->theta and finalF32 bytes, all21 causal prefixes,
nonfinite/overflow hard failure, endpoint/outside proposal preservation, original
support decisions, corruption detection, private/callable mutation guards, and
no default RNG/input mutation.
The immutable tests receipt additionally hashed an earlier proof script before a
PROOF-ONLY inward-boundary-spacing correction. Decoder/test bytes did not change;
proof-receipt.json captures the exact FINAL proof script and both fresh02 runs.
Earlier01 reports remain preserved outside this package; no result was overwritten.

Original closed physical support, complete-row rejection, scientific CDF/moment/
SD gates and exact physical18 invariance under any future G3 remain separate and
unchanged. No posterior quality, endpoint atom equivalence, new proposal success,
or GPU qualification is claimed here. Root owns all later source review/budget
decisions. Source/proof/hash paths and exact commands are in the receipts.

READ-ONLY SAVED NOISE-LATENT DIAGNOSIS — request remains FAILED

This package authenticates the existing failed seed20261111/case0 v2 request,
manifest 92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce. It contains metadata, scalar diagnostic
values, and deterministic diagnostic code only. Raw sample/latent arrays remain
on Karolina and are not included here. Built with PriorLabs-TabPFN.

RESULT
Exactly 2 offending noise coordinates in 2 of 51,200 raw rows (0.00390625% of
rows; 2/307,200 noise scalar coordinates). Both are within raw row positions
[0,50000): 2/50,000 = 0.004% of those rows. None is in positions [50000,51200).
These last 1,200 positions are a NOMINAL raw-provider lookahead tail, not an
actually finalized unused physical FIFO buffer: strict decode failed before
the provider returned any physical wave, support calls=0, delivered=0,
FIFO finalized=False. No completed 50,000-row cohort or accuracy result exists.

Zero-based offending entries:
- row 4355, ordered stage 15, physical slot 16 sigma_a1:
  Float32 latent 13.876729011535645, bytes 15075e41.
- row 41765, ordered stage 19, physical slot 20 sigma_a3:
  Float32 latent 205.2462615966797, bytes 0b3f4d43.
Both actual frozen d8 scalar noise inverses produce 0.5  before rejection.
The saved first-failure row/stage/slot/value/hex/full-row bytes match the NPZ.
No other noise coordinate fails the unchanged component decode/encode and
declared roundtrip checks. All 51,200 rows and all 6 noise slots were examined.

EXACT SOURCE / ARCHIVES
Frozen codec d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5:
prior_coordinates_v1.py:137 calls strict_noise; line110 rejects endpoint,
out-of-support or saturation; line43 raises CoordinateError. Full causal
prefix decode calls the noise inverse at176 via decode_ordered_prefix:234.
F32->F64 widening is the unchanged operation at233. No tolerance was changed.

latent-proposal-waves.npz 1ba89c1a4e53c5895ad1623b5a396e4086154c3dc830f8036da388f234d90349
contains one Float32 [51200,21] ORDERED LATENT wave, not physical samples.
Its raw array SHA ise51e00f9e8ec686a09b2280a9f59c417d4f98c306843a20785ed7579c6a1ef7e.
latent-stage-returns.npz 6989b4930fcb8dda2a144811c7988530dec120b8e6c1bfd62b269f47261a14af
contains 21 actual Float32 [51200] stage-return vectors, a bool [21] filled mask,
and a duplicate Float32 ordered prefix. All 21 vectors directly byte-match
their ordered wave columns; no extra cast is needed. This verifies original
performance_runner.py:470,472 storage convention (source 23ccad8b...), preserved
by SS15 wave 4037d488.... Both archives and the failed manifest remain unchanged.

COMPARISON TABLE (all51200 generated latent values)
stage  3 z_sigma_m1   generated[-1.77935195,3.08686471] q99=1.1759913 q999=1.86352898; training[-3.502141,3.53904366]; originalFIRST1024[-1.46290231,1.49017453]
stage  7 z_sigma_m2   generated[-2.11426711,2.59498739] q99=0.658933411 q999=1.39151382; training[-3.56248283,3.27795792]; originalFIRST1024[-1.53011799,1.4807893]
stage 11 z_sigma_m3   generated[-2.27961898,1.95064878] q99=0.199386346 q999=0.717763578; training[-3.71590328,4.05666685]; originalFIRST1024[-1.84604836,0.547746003]
stage 15 z_sigma_a1   generated[-3.74332786,13.876729] q99=2.1975297 q999=2.87609114; training[-3.71831155,3.44761229]; originalFIRST1024[-2.71650648,3.28479481]
stage 17 z_sigma_a2   generated[-4.41265392,5.98021555] q99=2.19754131 q999=2.84376554; training[-3.60713291,3.41744161]; originalFIRST1024[-2.81509233,2.83050942]
stage 19 z_sigma_a3   generated[-5.78154373,205.246262] q99=2.19300062 q999=2.96602146; training[-4.02613449,3.82169366]; originalFIRST1024[-3.2240715,3.31362033]

Ordinary quantiles use NumPy linear on the fixed grid recorded in diagnostic.json.
The 4096 training values are from the authenticated ORIGINAL seed11 prior bank.
The FIRST 1024 fixture is from the authenticated ORIGINAL pure pre-G3 physical
case0 cohort (manifest 3e0c158f..., payload a9ecadf2...), encoded by the exact
frozen noise map and one declared Float32 cast. Generated values condition on
held-out case 0 and generated prefixes; the bank is unconditional prior data and
the old fixture is an earlier learned-model output. These range/quantile
comparisons are descriptive, not a reference-posterior or calibration test.
Training-range exceedance is not evidence of canonical-border exceedance.

MECHANISM / REPRESENTABILITY
The full saved manifest contains 165 border-named fields: dtype strings and
six dtype/shape/SHA descriptors for native-stage 6/10/14 criterion borders.
For the offending cached stages 15/19 it records border dtypes only, no numeric
prediction-specific border values or min/max. Historical selected member,
uniform, logits/probabilities, bin, CDF, and interpolation fraction are absent.
The archive cannot distinguish wide-border ordinary interpolation from
rounded-CDF extrapolation, or attribute the latent extremes to either.

Separate sibling evidence ../noise-inverse-math-diagnostic-01 reports that the
ideal prior inverse values for these two z are closer to 0.5 than half its F64
predecessor gap: a correctly rounded inverse is endpoint 0.5 for both.
That scalar analysis explains strict representability failure for the SAVED
values; it does not explain why the neural/ICDF path produced them.

If separately authorized, the highest-information next diagnostic is a sharply
capped conditional/ICDF observation at the saved offending prefixes, recording
all 8 current head borders, probability normalization, and u/head/bin/fraction.
New observations cannot recreate missing historical witnesses. Numerical-policy
or coordinate changes should be considered after mechanism evidence; an extra
full posterior request is not needed for this evidence. This package authorizes
no probe, retry, policy change, filtering, repair, or recovery.

OBSERVED GUARDS AND LIMITS
Existing genuine exception observations show all 21 stage returns/aliases restored,
CDF 1141 and restored alias, postexception source closure passed, and observed
default RNG equality. The literal completed_stage_indices remains[] and git_dirty
remains True. Failed wrapper timing 777.6754412082955s is not a successful request
latency. Missing normal-success execution/sampling/prediction/memory/rejection
footers remain missing; see the exact list in summary.json. No failure observations
are relabeled as a normal success footer.

Diagnostic scope: exact per-noise frozen functions on unchanged contiguous
components, recursive subdivision only to locate failures, all rows retained
and accounted. No generated full 21D physical matrix was assembled or saved.
No model, sampler, world, reference integration, G3, random draw/state capture,
clipping, filtering, repair, resampling, coordinate/source/threshold change.
Torch was not imported. Sources and original data SHA verified before/after.
The deterministic diagnostic remains independent of any future experiment.

REPRODUCE (authorized saved HPC artifacts required)
cd /home/it4i-bbenvegnen/workspace/PosteriorForge
PYTHONPATH=src OMP_NUM_THREADS=16 /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python \
  <scratch-folder>/diagnose.py --repo /home/it4i-bbenvegnen/workspace/PosteriorForge \
  --scratch-root /scratch/project/eu-26-17/eubrieucb/posteriorforge
Runtime versions: Python 3.11.13, NumPy 1.26.4, SciPy 1.12.0.
diagnostic.json SHA c08076c86e7d329f65ddb417224c2430ceb4b3dfdb0e24e00ae8f6000e691f93
No code or raw artifact from the failed request is modified.

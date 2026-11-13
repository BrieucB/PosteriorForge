Saved Policy-C noise-conditioning scalar diagnostic
Built with PriorLabs-TabPFN.

Scope
This is a deterministic CPU-only mathematical diagnosis of all16 stage19/N1024
original/candidate own-U Policy-C values from closed postprocessing reportcbc8.
Those densities were NEW conditional evaluations at saved prefixes, not the
historical failed posterior density/member grouping. No new density, neural
forward, model constructor, world, posterior, G3, reference integration or RNG
draw occurs here. The 16 scalars are diagnostic inputs, not a physical cohort.
Frozen source d8 and all existing physical accuracy gates remain unchanged.

Finding
For z7.876539762950072--8.215596689438383, the frozen d8 physical inverse is only
0--2 ULPs from the correctly rounded high-precision inverse. Its absolute
physical error is6.55e-18--9.45e-17 versus the unchanged scalar F64 physical
bound about2.089e-14 (error/bound0.000314--0.004524). Physical inverse accuracy
and latent round-trip admissibility are different requirements.

For EACH of the16 saved z values, no physical Float64 exists whose EXACT PRIOR
encoding falls inside either unchanged RT32 or RT64 latent interval. The best
exact-prior latent error across all Float64 values is0.000657894--0.007231660:
19.429--206.562 times RT32,621.731--6609.998 times RT64. This is not solely a
frozen decoder-accuracy defect. Conditioning of the ideal inverse followed by
Float64 rounding already makes the current latent tolerances unattainable.

Representability argument and numerical evidence
The ideal prior transform is strictly increasing. The two physical Float64
numbers bracketing the real inverse therefore minimize latent error over ALL
Float64 numbers. Independently, the report inverts each permitted latent
interval and counts Float64 values inside that physical interval: zero.
The nearest-Float64 rounding-cell latent preimages are also recorded.
A separate exhaustive frozen d8 encoder check covers every strict-interior
Float64 cell inside each unchanged F64 physical-accuracy interval (380--441
cells/case): all observed monotone, zero RT32/RT64 solutions. This finite scan
does NOT assert global numerical monotonicity outside the checked interval.

High-precision reference and limitations
The mathematical reference uses exact binary rational source mu=.255,
sd=.1225, low=.01, high=.5 from the archived scalar math report, with exact
real truncation bounds. It also computes the rounded-geometry a=-2,b=2 ideal:
physical inverse differences are at most1.46e-32, changing no Float64 conclusion.
Mpmath100/180 decimal precision inverses agree within4.51e-101 physical units.
A1e-80 physical stability envelope and independent inverse/reencode sign and
residual checks are recorded. These are HIGH-PRECISION NUMERICAL checks, NOT a
certified interval-arithmetic error certificate. The Float64/tolerance gaps
are many orders larger than that envelope. No original qualified posterior
reference is integrated, sampled or replaced by this scalar reference.

Contract scope
Original prior closed-support endpoint semantics are a separate source audit.
This report does not accept endpoints, weaken strict d8 domain/round-trip
checks, substitute neighboring physical cells, salvage failed proposals,
alter any CDF/mean/SD accuracy gate, or select a future production policy.
Prospective options, not implementations
1. Because the original scientific prior/gate uses CLOSED Float64 physical
support at both .01 and .5, explicitly review a Float64-rounded-output contract
FIRST: verify physical inverse rounding accuracy and preserve unchanged physical
CDF/moment gates, while honestly replacing an unattainable latent-RT requirement
only through an approved numerical contract. Closed support alone does not prove
the rounded output law equals the ideal continuous posterior; rounding cells and
any endpoint probability must be declared. No such contract is accepted here.
2. An explicit whole-row strict representability/RT rejection policy q|D would
instead change the accepted law and require counted mass plus fresh qualification;
it cannot salvage/filter the current failed wave.
3. Broader physical/mixed noise coordinates would be a new representation.
No clipping, endpoint/predecessor replacement, tolerance waiver or current-source
change is implemented or authorized by this diagnostic.
Original frozen scalar results and exact conditional comparisons are retained.

Artifacts and reproduction
exact-report.json.gz expands byte-exactly to report440b5f6b...; see hashes.
reproduce.py is an exact copy of scripts/diagnose_prior_noise_conditioning_v1.py.
Run in the existing inference CPU environment, CUDA_VISIBLE_DEVICES='', with
PYTHONPATH=src and --report pointing to tail-postprocessing-offline-01/report.json,
--math-report evidence/accuracy-prior-coordinates-v1/noise-inverse-math-diagnostic-01/reproducible-scalars.json,
and --output pointing to a fresh folder. No Torch is imported in this script.
The output report records versions, unchanged input/source hashes and default
NumPy/Python RNG state hashes before/after. No samples/NPZ are in this package.

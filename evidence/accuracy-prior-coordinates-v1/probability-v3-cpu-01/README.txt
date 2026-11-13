# Probability v3 focused CPU checks

This is a new CPU Float64 probability candidate over the unchanged FP32 neural policy. No neural forward, model construction, preprocessing replay, GPU, new posterior/world, G3, or random draw was executed by these checks. Existing saved densities were read.

The inference environment passed all21 focused tests. The locked reference environment passed7 source/guard tests and explicitly skipped14 Torch-dependent tests because Torch is absent; this is not a second numerical execution of the Torch path.

All48 recomputed RAW translated PolicyC arrays are bit-identical to the previously saved independent CPU PolicyC arrays. All288 provided-uniform inverses satisfy represented intervals and independently checked exact-dyadic RAW/API-weight CDF bounds. The maximum checked CDF residual is approximately1.11e-16. These are saved-density arithmetic checks, not GPU, whole-posterior, decoder or physical-accuracy qualification.

The new inverse uses the normalized represented Float64 CDF and records its differences from normalized raw weights, including naturally collapsed positive intervals. No epsilon floor, threshold tail deletion, uniform repair or parameter clipping is introduced. The u=0 first-positive-interval extension is explicit; u=1 remains an error. The head-acceptance callable is the exact original function with its original envelope sealed. Stage0 means RAW8 weights before one shared normalization.

The receipt preserves candidate sources, exact commands/logs, actual environment/package metadata and source-only preflights that did not import Torch. New-file committed-source closure is pending Root's final candidate commit. Earlier local test attempts remain in the scratch phase; attempt01 failed only on a test's saved-reference key name and was corrected before the final recorded suite.

No NPZ or raw density arrays are included in this package.

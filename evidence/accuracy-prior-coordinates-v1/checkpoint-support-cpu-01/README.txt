CPU CHECKPOINT / COMMON FINITE-BAR SUPPORT RECEIPT

This separate metadata-only package determines what frozen source, the pinned
checkpoint state dictionary, and saved target mean/SD can establish without
neural evaluation. Existing array and mathematical diagnostic packages remain
unchanged. Built with PriorLabs-TabPFN.

VERDICT (all values are prior-latent target units, not decoded physical sigma)
Stage15 / row4355 / z_sigma_a1:
  Saved output13.876729011535645 lies inside reconstructed common all8 envelope
  [-93.85807037353516,87.60953521728516]. INSUFFICIENT to call it outside a
  member's actual positive-mass support or diagnose its ICDF calculation.
Stage19 / row41765 / z_sigma_a3:
  Saved output205.2462615966797 lies above reconstructed common all8 maximum
 86.4041519165039 by118.84210968017578. Common envelope:
  [-92.5218505859375,86.4041519165039].
  This is OUTSIDE the source-bound finite-bar output envelope for every head;
  ordinary interpolation within a finite bin cannot explain the value.
  The exact historical numerical cause remains unidentified.

WHAT WAS OBSERVED / RECONSTRUCTED
The failure manifest92755b... saved target mean/SD (unchanged from cold to final)
but did not capture numerical prediction borders at these stages. The current
CPU inspection loaded only checkpoint2ab5... state_dict, no model construction.
Its criterion.borders is Float32[5001], endpoints[-93.08673095703125,
86.94256591796875], raw border SHA4e9e53f00186e68f15628f55647b44d4b790039c2d155da14606cd493eda4a83.
The envelope is RECONSTRUCTED using the exact stock Float32 expression:
  bardist_.borders * y_train_std_ + y_train_mean_
Saved stage15 mean/SD: -.027640938758850098 /1.0079892873764038.
Saved stage19 mean/SD: -.0055904993787407875 /.9938715696334839.

Stock regressor.py:521-529 standardizes target and creates that criterion.
Lines672-685 inverse-transform each member's target borders; lines695-701
translate each member back onto the common checkpoint bardist grid; lines747-753
return the single renormalized criterion. Thus SafePower/inverse-target fitted
parameters are not needed for this COMMON envelope. They and the unrecorded
prediction probabilities may restrict actual positive-mass support inside it.
This is an argument about execution_mixture.py:224-252 finite_bar_icdf used by
the frozen sampler, not arbitrary full-support-tail methods on a similarly
named distribution class. No target-transform/preprocessing replay was done.

HISTORICAL ROW / MEMBER / BATCH LIMIT
The actual query journals give stage, member, ordering, dtype/shape/SHA for
masked/unmasked queries, but no raw row indices or per-row assignment/acceptance
decisions. Stage15 had3 head rejections; stage19 had1. Count-only evidence does
not identify which raw row retried or its final member/chunk. Component U is
preassigned by raw row and retained over retries; its separately authenticated
reconstruction does not determine assignments or rejection history.
No assignment/acceptance stream replay or subset/brute-force hash matching was
performed here. No historical logits/bin/probability/CDF/fraction is inferred.

A fixed new1024-row probe at [4096,5120) contains row4355 at offset259; a new
window [40960,41984) contains row41765 at offset805. Those are new contiguous
query batches, not authenticated historical member-grouped batches. A new
observation can investigate mechanisms but cannot manufacture missing history.

RNG / CUDA / SOURCE SCOPE
Default NumPy/Python/TorchCPU states were hashed after imports and before/after
checkpoint loading plus CPU Float32 border arithmetic. Exact before/after
hashes match; no claim is made about import-time initialization. CUDA remained
uninitialized. There were zero model constructors/forwards, preprocessing
replays, target-transform replays, private RNG draws/replays, posterior/G3 calls,
cohort decoding, repair, or frozen-source edits. All9 pinned SDK source hashes,
the checkpoint, manifest, runner, mixture, and active producer source were
verified; original files were checked again after inspection.

The first scratch script attempt had a syntax-only parenthesis error before
execution/imports. Its log is preserved separately; attempt02 corrected only
that new-script typo and produced this receipt. This does not change any
numerical source, inputs, or prior diagnostic. The earlier exploratory CPU
checkpoint inspection produced the same scalar endpoints; no model calls.

REPRODUCE
cd /home/it4i-bbenvegnen/workspace/PosteriorForge
PYTHONPATH=src OMP_NUM_THREADS=1 \
 /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python \
 <scratch-folder>/inspect_support.py

No raw arrays or checkpoint weights are included in this public package.
No GPU probe, policy change, retry, clipping, filtering, or salvage is performed
or authorized by this receipt.

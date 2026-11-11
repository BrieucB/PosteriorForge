# Additive reference CDF query refinement

`reference_grid.refine_reference_grid(banks_root, case_index, reference_root,
output_root, max_cdf_step=0.005)` creates a separate immutable reference table
from the completed parent reference. It does not alter the parent, observation,
model, posterior sample, reference nodes, weights, or fixed53projection directions.
This numerical refinement is required when the original shared513point query grid
cannot support a conservative0.10 CDF acceptance decision for narrow posteriors.

The operation requires a current single-node Slurm job with a held submitted GPU
reservation in the shared200hour budget ledger. By default the ledger is
`$PF_SCRATCH/budget/ledger.json`; `budget_ledger` or `PF_BUDGET_LEDGER` can name it
explicitly. All retained-node CDF evaluation uses float64 JAX on the allocated
GPU. The deadline is at most two hours; failures, timeouts, retries and CPU-side
artifact authentication consume the same allocation budget.

The parent manifest and table hashes, science/observation binding, fixed direction
hash, combined node hash/moments, and all eight last-two-power scrambled node banks
are authenticated first. The seed/power of each bank is verified against the
frozen SeedSequence derivation. Parent numerical and independent checks must
already qualify. The original513points over[-8,8] seed an evaluated-query cache.

At every round, each current interval whose maximum CDF increment over all53
directions exceeds0.005 is bisected. Every selected midpoint is evaluated once;
selection uses only the retained reference CDFs. No candidate posterior values
enter this algorithm. New points must satisfy both final-four-scramble spread
and previous-power agreement<=0.005. End-point tail masses must be<=0.005, and
all final CDF increments must be<=0.005 before the new table is qualified.
Exhausted round/query/deadline limits remain inconclusive. A newly detected
precision failure requires separate reference refinement, never a relaxed gate.

The output manifest records parent manifest/distribution SHA256, all retained node
manifest hashes, runtime/source identity, budget reservation, complete per-round
midpoint history, and the declared grid contract. Every added query is an exact
dyadic midpoint in an original interval. A verifier can replay each round using
subcolumns of the final saved CDF table to authenticate that the selected
intervals depend only on the fixed reference threshold. The NPZ retains the
parent's physical mean/SD and moment qualification exactly, with updated CDF
precision. The usual candidate CDF/bias/SD acceptance limits remain unchanged.

# Case 0 fixed numerical screen: v1 failure retained

Built with PriorLabs-TabPFN.

The complete all-21-stage screen for seed **20261111 / case 0** did **not** pass.
Its literal status remains `prior_coordinate_pair_structural_or_hard_failure`.
The actual recorded flags distinguish the outcome: all 21 hard-validity guard
sets passed; the sole numerical tolerance failure is the member/query below.
All final-mixture and inverse-CDF comparisons passed. This does not waive the
predeclared per-member criterion.

| Recorded field | Exact value |
| --- | --- |
| Stage index (zero-based) | 14 |
| Prior coordinate / physical slot | `z_d0_m3` / 13 (`d0_m3`, mechanical_3) |
| Route | `frozen88_resident_cache` |
| Query / recorded shape | N=1024 / `[1024,29]` |
| Training / stress shapes | `[4096,29]` / `[8,29]` |
| Member index (zero-based) | 5 |
| Member maximum CDF difference | `2.0325183868408203e-5` |
| Unchanged CDF limit | `2e-5` |
| Member total variation / limit | `2.1338464489609188e-5` / `1e-4` (passed) |
| Member normalization error / limit | `0` / `1e-5` (passed) |
| N=1024 final-mixture CDF difference | `6.335889978004161e-6` (passed) |

N=1 and the eight stress rows passed at this stage. All other 20 stages passed
all matching criteria. `all21-outcomes.json` retains every recorded stage/member
outcome and scalar metric; `failure-details.json` preserves the exact sole
failing member metrics, query shape and thresholds. No thresholds, numerical
predicates or status labels are changed. No cause, cache bug, or causal link to
old physical posterior failures is inferred by curation.

The recorded source and global RNG guards passed; all CDF/precision aliases
were restored. The one context records **1,008 actual corrected-CDF calls**.
There is no recorded runtime exception. The screen generated **zero posterior
vectors** and performed zero reference integrations. No v1 posterior job,
joint-global update, physical accuracy assessment, retry or registry activation
followed this failure. The separate CPU23 preflight remains within its codec/
input/reference scope; it is not a numerical or posterior pass for these pairs.

Slurm process state was `COMPLETED`; this means the screen process finished,
not that its numerical criterion passed. Root's reconciled **main allocation**
was **440 GPU-seconds, one GPU, zero restarts, no requeue**. Batch/extern step
rows are retained but excluded from the cost sum. The original reservation was
1,200 GPU-seconds. Recorded screen wall time was433.2478167640511 seconds.
Root's post-reconciliation ledger records103,706 actual GPU-seconds
(**28.8072222222/200 GPU-hours**), zero reserved,125 entries. Curation did not
modify or reconcile that ledger.

`manifest.json.gz` decompresses byte-for-byte to the original19,190,368-byte
manifest with SHA-256
`fbed99d9fb256bb72a49cb03bebb820d805ac2cbafaa4ebac4deb08bf3160734`.
Allocation receipt SHA-256 is
`9ee6c03d940410597cce304fc65593612b2891d81745561ed5c1d7c7082401e8`.
Raw Slurm accounting and available exact stdout/stderr streams are included;
stream scope/availability and hashes are recorded in summary.json. No NPZ,
checkpoint, predictive-array or sample archives are copied into Git. No NN,
model, simulator, posterior, reference integration, diagnostic experiment or
GPU call is performed by this metadata curation. V1 source/proofs and the prior
c810 failure remain unchanged; no later v2 result is included here.

# V2 case 0 fixed-conditional numerical screen

Built with PriorLabs-TabPFN.

The fresh numerical screen for **seed 20261111 / case 0** passed **all 21 stages**,
all 8 members, N=1, N=1024 and eight stress rows. Its literal status is
`prior_coordinate_pair_v2_screen_pass`. The execution policy is original all-eight
FP32 at stages **6, 10 and 14**, and frozen cached FP32 at the other **18** stages.
All hard guards, final-mixture and inverse-CDF comparisons passed. Native stage14
records maximum member CDF difference **0** across the declared queries.

The largest recorded member CDF difference is `1.3589859008789062e-05` (stage 16, query 1024, member 0; zero-based indices), below the unchanged `2e-5` limit. The largest recorded final-mixture CDF difference is `3.5332062232606987e-06`. `maxima-summary.json` preserves the exact maxima/locations for member and final
CDF, total variation, normalization and probability error; `all21-outcomes.json`
retains every recorded member/shape/route/guard result. These summaries select
recorded scalar diagnostics and do not rerun NN outputs or probability arrays.

The one restored context records **1,008 actual corrected-CDF calls**. Source,
global RNG and alias/SDK restoration guards passed. There is no recorded runtime
exception. The screen generated **zero posterior vectors** and performed zero
reference integrations. This is a **finite-prefix matching execution proof only**,
not posterior accuracy, a 50,000-vector cohort, global numerical equivalence,
population calibration or qualification of another seed/case. Pair402/CPU6f35
remain input ancestry only. Any later conditional posterior and one physical G3
update have separate sources, guards, artifacts and assessment.

Root reconciled job **5411500** as Slurm `COMPLETED`, **433 allocated GPU-seconds**,
one GPU, zero restarts and no requeue. Batch/extern child steps are retained but
not added to cost. The original reservation was 1,200 GPU-seconds; recorded screen
wall time is **426.4734170259908 seconds**. The exact allocation receipt records
104,139 total actual GPU-seconds (**28.9275/200 GPU-hours**), zero reserved and
126 entries at reconciliation. Curation does not mutate that ledger.

No speed gain or isolated route-cost effect is claimed. Root reported v1's
440-second screen on acn05 and v2's 433-second screen on acn12; these different-node
observations are accounting records, not a controlled performance comparison.
The original v1 raw `fbed99d9...` remains failed and unchanged; no v1/case2 pass is
transferred into v2 or a homogeneous panel.

`manifest.json.gz` decompresses byte-for-byte to the original **18,486,492-byte**
manifest SHA-256
`aa8aff311d8f7fab6d774337510bb5f66abed309300c995267e98fa7f3e2a88e`.
Allocation receipt SHA-256 is
`4e2b07d53a6c8d482d2080b1ced540ccb3a5a13d1d9de145d94370598c0c6a1b`.
Exact available stdout/stderr and raw allocation accounting are preserved, with
hashes in summary.json. No NPZ, checkpoint, predictive-array or sample archives
are copied to Git. No source edits, model/NN reruns, worlds, posterior draws,
reference integrations, GPU, registry or G3 actions occur during curation.

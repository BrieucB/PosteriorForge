# V2 case 0 posterior: launch metadata only

Built with PriorLabs-TabPFN.

Root submitted **one 50,000-complete-vector request**, seed **20261111 / case 0**,
context 4,096, repetition 0, as job **5411530**. The source commit is
`ff1921e88a3c3436208861ae506bef44871906c9`; recorded
[CI run 43354323971](https://github.com/BrieucB/PosteriorForge/actions/runs/43354323971)
concluded success. The numerical policy is FP32 original all-eight execution at
stages **6, 10 and 14**, with frozen cached FP32 at the other **18** stages. The
same-pair v2 screen was separately closed with SHA-256
`aa8aff311d8f7fab6d774337510bb5f66abed309300c995267e98fa7f3e2a88e`.
That finite-prefix screen is not a posterior accuracy assessment.

The original launch files are copied byte-for-byte. Preflight SHA-256 is
`213bbce411d7bdc7d2cdee7a9e6e90df3513021255eb2a3dd7e988feebf9c94f`;
submission SHA-256 is
`fcc3e890fb49595339d64907ec95b6c09dd8874730d1a047209755d67d5888f4`.
They record one GPU, `--no-requeue`, a **1,200-GPU-second reservation**, and
reservation `34337f2890be48a0b069ed1efc0e3ff7`. Original `state=submitted` and
`actual_gpu_seconds=0` are launch bookkeeping, **not terminal accounting or a
completed-cohort claim**. No running output manifest was inspected or copied.

Root's actual pre-submission guard authenticated 155 committed source entries,
with the producer's own source checked separately. It recorded no Torch import.
Source-validation wall time was **0.5596202770248055 seconds**, included within
**2.7991984598338604 seconds** total CPU validation including imports. Both were
measured **before reservation**; the source subset must not be added again to the
total, and these are not GPU-request timings or an optimization benchmark.

The exact five untracked checkout names and their original scope are retained
in both raw preflight and summary. They are a read-only checkout-name list,
**not a live-process module inventory** and not a clean-tree claim. The old
completed screen in the queue snapshot is preserved; Root recorded zero active
or uncertain project allocations before this single submission.

This package contains no numerical result, full posterior, physical gate,
final allocation cost, G3 update, sample array or NPZ. A valid closed request may
later permit one separate physical joint-global G3 update and unchanged
assessment; none occurred during launch curation. Old v1 failures remain
unchanged and give no v2 or other-pair qualification credit. Curation performs
no source edits, model calls, reference integrations, GPU submissions, ledger or
registry changes, G3, or commits. `SHA256SUMS` excludes itself.

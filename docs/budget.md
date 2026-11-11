# Compute budget and launch policy

Every PosteriorForge allocation uses one durable private ledger, normally
`$PF_SCRATCH/budget/ledger.json`. The immutable cap is 200 allocated GPU-hours.
Reference calculation, qualification, failed jobs, retries, profiling and final
confirmation all consume this same cap. Queue waiting consumes no allocation.

`BudgetLedger.submit` is the only submission path. It reserves
`GPU count × requested walltime` under a filesystem lock, saves and synchronizes
the ledger, then calls `sbatch` once. Same-node GPU counts are restricted to
1, 2 or 4 on Karolina qgpu using account eu-26-17. Script resource directives
are rejected so allocation resources agree with the reservation. Launch scripts
run offline after dependency and checkpoint preparation.

`reconcile_all()` reads Slurm allocation records using `sacct -X`, excludes job
steps, and counts elapsed seconds times allocated GPUs. Generic and typed GPU
TRES are alternatives, so `gres/gpu=1,gres/gpu:a100=1` counts one GPU. Terminal
failed/cancelled/timed-out jobs are charged exactly like successful jobs. Missing
accounting and nonterminal jobs retain their entire reservation. An unexpected
accounting overrun is retained truthfully and prevents additional submissions.

If `sbatch` times out or its response is ambiguous, the reservation remains held.
Never retry blindly. Recover the submitted allocation ID with the unique `pf-`
job name and attach it with `attach_job`, then reconcile after termination.
Only a reservation whose submission was never attempted can be released with
`release_unsubmitted`. This conservative behavior survives interruption between
Slurm acceptance and local receipt of its job ID.

The ledger contains timestamps, commands, job IDs, original reservations,
terminal Slurm accounting rows and cumulative actual/reserved/remaining cost.
Keep the mutable ledger outside Git; publish a sanitized snapshot with portable
result evidence. Only PosteriorForge jobs are inspected or reconciled.

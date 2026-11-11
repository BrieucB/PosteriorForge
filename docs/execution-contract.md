# Execution contract

All source work, installs, scientific calculations, and jobs run on Karolina.
Checkout: `/home/it4i-bbenvegnen/workspace/PosteriorForge`.
Private scratch: `/scratch/project/eu-26-17/eubrieucb/posteriorforge`.
Immutable upstream inputs are separate checkouts. Existing BayesArena worktrees,
MesoUQ campaigns, AWS resources, and unrelated jobs are outside this project's
ownership. Files and GitHub are the only tracking systems.

Milestones proceed in order:

1. Qualification: public numerical closure, source parity, hashed environments,
   offline checkpoint, CPU CI, real GPU checkpoint and float64 algebra checks.
2. Scientific feasibility: independent retained references for eight held-out
   cases, then all 24 baseline 4096-context runs of 50,000 accepted vectors.
3. Optimization: only after feasibility passes, complete baseline 72-run matrix,
   profile untouched upstream with PyTorch CPU/CUDA traces, sweep structural
   reuse and microbatches 1024/4096/10000 on declared case 0; preserve all 21
   conditional stages; validate absolute and upstream-relative metrics.
4. Multi-GPU: only after scientific acceptance, one full replica per GPU,
   independent RNG, split 50,000 full rows across 1/2/4 same-node GPUs. Case 0,
   each accepted context size, three timing repeats, honest allocated cost.
5. Final audit: public-only installation, provenance, complete evidence and
   recruiter-readable report with figures, raw checksums and retrieval commands.

The budget is a hard 200 allocated GPU-hours, inclusive of references, failures,
retries, qualification, profiling and confirmation. Reservations cover requested
GPU count times walltime before submission; actual Slurm accounting releases
unused reservations after terminal jobs. Concurrent reservations count against
the cap. The planning envelopes 10/70/90/30 are transferable, not extra budgets.
Queue time consumes no allocation. No automatic polling service is created.

A failed 4096-context scientific gate is an authorized terminal feasibility
outcome. Publish whether failure is model error or reference uncertainty; leave
optimization and multi-GPU deferred for a user decision. Exhausted or inconclusive
reference refinement cannot pass. Do not fine-tune, change checkpoint, expand
scenarios, or enlarge the experiment to rescue a failure.

For inference, raw observations and fixed parameter order are used, without
context filtering. `append_simulations` is called once with a complete bank.
`max_iter_rejection=None` is mandatory: the upstream finite timeout appends
unchecked candidates. A job walltime terminates incomplete work as failed.
No posterior sample is clipped. Upstream leakage-uncorrected `log_prob` is never
reported as calibrated accepted-sample density.

Record cold start, context preparation, warm/total request latency, complete
samples/sec, allocated/reserved memory, stage timings, rejection, transfers and
synchronization. Include setup in eight-case amortization; report medians and
ranges for three timing repeats. Unrun measurements remain explicitly unrun.

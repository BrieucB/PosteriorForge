# Native-routing v2 case 0 screen: launch metadata only

Built with PriorLabs-TabPFN.

This package preserves Root's exact closed launch snapshots for job **5411500**,
seed **20261111 / case 0**, source commit
`dcc084e6f8b3a9490290e31027560973f766bf6a`. The authorized numerical screen uses
FP32 original all-eight execution at stages **6, 10 and 14**, and frozen cached
execution at the other **18** stages. It tests N=1, N=1024 and eight stress rows
for all 21 stages/all 8 members. The PairSpec count of 50,000 describes the
separately conditional posterior benchmark, not outputs from this screen.

[Recorded CI run 43351578320](https://github.com/BrieucB/PosteriorForge/actions/runs/43351578320)
concluded success. The exact Root preflight records the committed 155-source
inventory check passing, with inventory SHA-256
`edae96e0a5aa7f8a84b3d699636bf041f30fdf8dd4ff263c47fce2f68f6374ae`.
Root reported that this actual source check did not import Torch. Curation only
copies the recorded metadata and does not repeat a source guard or API audit.

Exact raw preflight SHA-256:
`7da236aedca2635f9f5854b59c0c8d739c12269d550376c33cef51b79c612725`.
Exact raw submission SHA-256:
`88093d10de6df860334a7e51175f65ddc77f325aee1b34a05e4502160a8cf783`.
Both are copied byte-for-byte. They record one GPU, a 1,200-second reservation,
`--no-requeue`, and reservation `a2b1da54e2194f3bb00be186cb7428fa`.
The original `state=submitted` and `actual_gpu_seconds=0` are preserved as
launch-time bookkeeping; they are **not final accounting** or a success result.
No running screen manifest was inspected or copied for this package.

Authorization is only this fresh v2 first-pair screen. No posterior reservation
follows unless the actual v2 numerical screen passes. No other pair, FP16 run,
optimization, profiling or scaling is authorized here. Pair402/CPU6f35 are data
ancestry only; the old case2/v1 numerical pass and v1 case0 failure do not supply
a v2 pass or homogeneous-panel credit. The old artifacts remain unchanged.

The exact checkout names listed as untracked at Root's preflight are retained:

- `scripts/diagnose_host_reuse.py`
- `scripts/evaluate_ss15_campaign.py`
- `src/posteriorforge/execution_tail_observer.py`
- `tests/test_execution_tail_observer.py`
- `tests/test_host_reuse_diagnostic.py`

This is a **read-only checkout-name listing, not a live-process module
inventory**. It neither proves a clean working tree nor identifies which modules
the GPU process loads. The raw queue snapshot may include the completed v1 job;
Root's recorded active-or-uncertain project allocation count was zero before
this one submission, not a claim about all cluster users or processes.

No final numerical outcome, completed posterior cohort, physical assessment or
terminal Slurm cost is included. No models, draws, reference integrations,
NPZ/sample arrays, ledger mutation, GPU submission, code/source edits or commits
are performed by metadata curation. `SHA256SUMS` excludes itself.

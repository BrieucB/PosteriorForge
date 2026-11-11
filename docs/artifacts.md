# Evidence and raw artifact retrieval

PosteriorForge keeps immutable inputs, sample arrays, reference node banks,
profiler traces and scheduler logs outside Git. Compact JSON/CSV evidence and
figures can be committed after inspection. Model weights are never committed.
The default private project root on Karolina is
`/scratch/project/eu-26-17/eubrieucb/posteriorforge`.

## Regenerate a portable report

Run this on Karolina in the qualified environment, after creating a gate artifact
from an explicit banks/reference/run root:

```bash
posteriorforge report \
  --gate "$PF_SCRATCH/results/feasibility/gate.json" \
  --ledger "$PF_SCRATCH/budget/ledger.json" \
  --output "$PF_SCRATCH/results/feasibility/report-attempt01"
```

The output directory must be fresh. `report.md` embeds every generated figure
inline. Reference precision and budget figures use retained state even when the
campaign is incomplete. Posterior, latency and memory figures are generated only
when their actual numeric evidence exists. Missing measurements are described
explicitly; no unmeasured speedup, timing percentile or scaling curve is drawn.

`gate.json` records the scientific verdict and per-run assessments. `runs.csv`
contains posterior metrics and measured complete-request timings/memory;
`references.csv` lists retained reference precision and state. `budget.json` and
`budget.csv` contain sanitized cumulative allocation and reservation records.
`checksums.json` records SHA256 values for each output file and input snapshots.
Private absolute submission commands are omitted from the public budget snapshot.

## Locate and verify raw evidence

Banks use NPZ arrays with JSON manifests recording array descriptors, parameter
order, seeds, scientific fingerprint and payload checksums. Complete posterior
runs use `samples.npz` plus `manifest.json`; the run manifest records the sample
payload SHA256. References retain separately authenticated scramble/node banks
and a `distribution.npz` CDF table. The reference manifest binds the held-out
observation and scientific fingerprint to those checksums. The scientific gate
records each selected run/reference manifest SHA256.

First inspect a published artifact index or the selected gate JSON to identify
the exact run directory. Do not choose a latest directory by modification time.
Verify the complete SHA256 chain before analyzing or sharing downloaded arrays.
For example, from a machine with the authorized SSH alias:

```bash
ssh -o BatchMode=yes -o ForwardAgent=no karolina \
  sha256sum /scratch/project/eu-26-17/eubrieucb/posteriorforge/results/EXACT-RUN/manifest.json
scp -o BatchMode=yes -o ForwardAgent=no \
  karolina:/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/EXACT-RUN/manifest.json .
scp -o BatchMode=yes -o ForwardAgent=no \
  karolina:/scratch/project/eu-26-17/eubrieucb/posteriorforge/results/EXACT-RUN/samples.npz .
sha256sum manifest.json samples.npz
```

`EXACT-RUN` is an intentionally unresolved example, not an artifact promised to
exist. Use the manifest hash retained by the gate and compare the sample checksum
against `samples_sha256` inside that exact manifest. On macOS, `shasum -a 256`
is the equivalent checksum command. Copying artifacts for inspection does not
require recomputing scientific results outside Karolina.

## Public downloads

The [reference-v1 release](https://github.com/BrieucB/PosteriorForge/releases/tag/reference-v1)
contains the analytical banks and all retained references, with server-verified
SHA256. The repository's [artifact index](../evidence/artifacts.json) records the
exact asset URL and digest. No private project access is needed to download it.

```bash
gh release download reference-v1 --repo BrieucB/PosteriorForge \
  --pattern posteriorforge-reference-v1.tar.gz --pattern SHA256SUMS.references \
  --pattern posteriorforge-reference-grids-v1.tar.gz --pattern SHA256SUMS.reference-grids
sha256sum -c SHA256SUMS.references
sha256sum -c SHA256SUMS.reference-grids
tar -xzf posteriorforge-reference-v1.tar.gz
tar -xzf posteriorforge-reference-grids-v1.tar.gz
```

The reference release preserves the original 513-point tables. Deterministic
query refinement is a separate artifact and preserves the retained nodes, physical
moments and independent checks. Always select the refined reference root used
by the final gate when reproducing its verdict.

Private SCRATCH paths require project access and are not public download URLs.
They are an optional access route for the project owner. Scientific source,
locks, configurations and public release assets support reproduction without
private repositories or private scratch access.

## Plot posterior agreement

After the complete gate, use its authenticated run paths and the selected
refined reference root:

```bash
posteriorforge plot-posteriors \
  --gate gate.json --runs upstream-v1 --references reference-grids-v1 \
  --output posterior-figures
```

The figure shows empirical CDFs, not leakage-uncorrected model log densities.
It uses the fixed first context seed and labels the worst marginal selected
for each case. Its JSON sidecar records selection, checksums and input bindings.
All scientific figures are generated on Karolina for this campaign.

Built with PriorLabs-TabPFN.

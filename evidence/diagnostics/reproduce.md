# Reproduce the failure investigation

Built with PriorLabs-TabPFN.

The original scientific release is immutable at `89404ac9e43bf9fb72d3e51a7113dbd207871221`.
This directory is a later diagnostic investigation. No corrected posterior or
candidate statistical inference is included. Read [report.md](report.md) before
interpreting probabilities, timings or the failed conditional-PIT attempt.

## Retrieve authenticated evidence

Use Python 3.11 and the original hashed environments in `environments/`.
Run scientific calculations on Karolina for this campaign. Download release
assets into a fresh working directory; the model checkpoint is not in any archive.

```bash
gh release download diagnostics-20261111 --repo BrieucB/PosteriorForge \
  --pattern posteriorforge-diagnostics-20261111.tar.gz \
  --pattern runtime.trace.json.gz --pattern SHA256SUMS.diagnostics
sha256sum -c SHA256SUMS.diagnostics
tar -xzf posteriorforge-diagnostics-20261111.tar.gz
```

The archive contains original partial runtime evidence, both complete checkpoint
capture manifests and NPZ payloads, failed-PIT evidence, Slurm logs, and sanitized
budget evidence. The 476 MB trace is separate; its uncompressed size is 7.40 GB.
Keep it compressed for bounded-memory replay. The [artifact index](artifacts.json)
records exact release URLs, sizes and digests. Public bank/reference/posterior
archives are listed in [the original artifact guide](../../docs/artifacts.md).

## Replay compact model evidence without a GPU/model call

Use the verified inference environment for the pinned Torch/TabPFN sampler CPU
microbenchmark. The model summaries themselves use NumPy/Matplotlib and saved
captures; they never feed a corrected probability back into inference.

```bash
export PYTHONPATH="$PWD/src"
python scripts/summarize_model_diagnostics.py \
  --bar diagnostics-20261111/bar-sampler \
  --translation diagnostics-20261111/ensemble-translation \
  --output work/replayed-model-summary
python scripts/diagnose_runtime.py sampler-cpu \
  --capture-dir diagnostics-20261111/bar-sampler \
  --output work/replayed-sampler-cpu.json
python scripts/diagnose_runtime.py trace-summary \
  --trace runtime.trace.json.gz --output work/replayed-trace-summary.json
```

The two-pass trace parser detects truncation, streams individual events, and
unions overlapping device intervals instead of summing them as elapsed time.
Timing rows in a rerun of the sampler microbenchmark may differ; saved request
and trace measurements remain the reported evidence. A warm-single timing does
not exist. The original runtime manifest remains `running` because only its
postprocessing child was terminated after complete raw trace export; the
controlled-stop sidecar and recovered report explain that state explicitly.

To regenerate the recovered runtime plot/report from these saved files:

```bash
python scripts/diagnose_runtime.py report \
  --profile-dir diagnostics-20261111/runtime-profile \
  --baseline-summary evidence/diagnostics/runtime-baseline-summary.json \
  --trace-summary evidence/diagnostics/runtime-trace-summary.json \
  --controlled-stop diagnostics-20261111/runtime-controlled-stop.json \
  --sampler evidence/diagnostics/runtime-sampler-cpu.json \
  --output work/replayed-runtime-report
```

The science/representation helpers document their required bank, run and
reference arguments in `--help`; they analyze the authenticated original public
arrays, not newly simulated worlds. `science-*-checksums.json` retains the
independent audit's original input/output bindings. `checksums.json` authenticates
the final compact diagnostic files.

## GPU collection is separate from replay

No additional GPU work is pending or authorized by these replay commands.
The original runtime collector executed at commit `c122261`; the bar collector
at `b3d799c`; the ensemble collector at `cdc9874`. Their manifests record the
execution revision, settings, bindings and source hashes. The final helper has
bounded-memory replay and improved CPU-tested failure metadata; the improved
noise-conditional precision path was not rerun on real rows. The failed PIT
attempt is retained as inconclusive and is excluded from scientific conclusions.

Any future collection must use the existing `BudgetLedger.submit` and Slurm
launcher on Karolina, with a new explicitly bounded reservation. The investigation
has ended at 2,271 additional allocated GPU-seconds, below its 7,200-second ceiling.
It does not authorize a reconstruction repair, statistical candidate, full sweep,
optimization or multi-GPU run. Those remain proposed next experiments.

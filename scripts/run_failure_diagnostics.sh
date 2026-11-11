#!/usr/bin/env bash
# Diagnostic measurement only; launch through the existing budget ledger.
set -uo pipefail
: "${SLURM_JOB_ID:?Accounted Slurm allocation required}"
: "${PF_SCRATCH:?Private project scratch required}"
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir="$PF_SCRATCH/results/diagnostics-20261111"
mkdir -p "$run_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PF_SCRATCH/cache/matplotlib-diagnostics"
cd "$repo_dir"
runtime_exit=0
"$PF_SCRATCH/envs/inference/bin/python" scripts/diagnose_runtime.py profile \
  --banks "$PF_SCRATCH/results/banks-v1" \
  --checkpoint "$PF_SCRATCH/cache/model/tabpfn-v2-regressor.ckpt" \
  --output "$run_dir/runtime-profile" --profile-count 10000 \
  --ledger "$PF_SCRATCH/budget/ledger.json" \
  > "$run_dir/runtime-profile.stdout.json" 2> "$run_dir/runtime-profile.stderr.log" || runtime_exit=$?
noise_exit=0
"$PF_SCRATCH/envs/inference/bin/python" scripts/diagnose_noise_conditionals.py \
  --banks "$PF_SCRATCH/results/banks-v1" --runs "$PF_SCRATCH/results/upstream-v1" \
  --output "$run_dir/noise-conditionals" --cases 0 --seeds 20261111,20261112,20261113 \
  --ledger "$PF_SCRATCH/budget/ledger.json" \
  > "$run_dir/noise-conditionals.stdout.json" 2> "$run_dir/noise-conditionals.stderr.log" || noise_exit=$?
"$PF_SCRATCH/envs/inference/bin/python" - "$run_dir" "$runtime_exit" "$noise_exit" <<'PY'
import sys
from pathlib import Path
from posteriorforge.artifacts import write_json
write_json(Path(sys.argv[1])/'combined-exits.json', {
    'schema':'posteriorforge.diagnostic.combined.v1',
    'diagnostic_only':True,'runtime_exit':int(sys.argv[2]),
    'conditional_pit_exit':int(sys.argv[3]),'automatic_retry':False})
PY
if [[ "$runtime_exit" -ne 0 || "$noise_exit" -ne 0 ]]; then exit 1; fi

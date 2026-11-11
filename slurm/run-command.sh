#!/usr/bin/env bash
# Invoke only through BudgetLedger.submit; resource options are supplied there.
set -euo pipefail
: "${SLURM_JOB_ID:?This launcher requires an accounted Slurm allocation}"
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != 1 ]]; then
  echo "PosteriorForge requires a single-node allocation" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: run-command.sh /absolute/environment/bin/python [arguments...]" >&2
  exit 2
fi
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export HF_HOME="${HF_HOME:-$PF_SCRATCH/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PF_SCRATCH/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PF_SCRATCH/cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TABPFN_DISABLE_TELEMETRY=1
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export TMPDIR="$PF_SCRATCH/work/job-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec "$@"

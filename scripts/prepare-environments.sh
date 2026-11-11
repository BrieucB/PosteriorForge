#!/usr/bin/env bash
# Run on a network-enabled preparation host. Compute jobs stay offline.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch_dir=${PF_SCRATCH:?Set PF_SCRATCH to a private directory outside the checkout}
bootstrap_python=${PF_BOOTSTRAP_PYTHON:-/usr/bin/python3.11}
case "$(realpath -m "$scratch_dir")/" in "$repo_dir/"*) echo 'Scratch must be outside checkout' >&2; exit 2;; esac
mkdir -p "$scratch_dir/envs" "$scratch_dir/cache/uv"
chmod 700 "$scratch_dir"
export UV_CACHE_DIR="$scratch_dir/cache/uv"
"$bootstrap_python" -m venv "$scratch_dir/envs/bootstrap"
"$scratch_dir/envs/bootstrap/bin/python" -m pip install --require-hashes -r "$repo_dir/environments/bootstrap.lock"
uv_bin="$scratch_dir/envs/bootstrap/bin/uv"
for name in inference reference; do
    "$bootstrap_python" -m venv "$scratch_dir/envs/$name"
    "$uv_bin" pip sync --python "$scratch_dir/envs/$name/bin/python" --require-hashes "$repo_dir/environments/$name.lock"
    "$uv_bin" pip check --python "$scratch_dir/envs/$name/bin/python"
done

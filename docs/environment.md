# Reproducible environments

Built with PriorLabs-TabPFN.

Two separate Linux x86_64 CPython 3.11 environments prevent incompatible SBI and
reference requirements from being mixed. `environments/inference.lock` fixes every
resolved transitive package and distribution SHA256, including Torch 2.6.0,
TabPFN 2.0.9, SBI 0.23.0, NumPy 1.26.4, and SciPy 1.12.0. The Torch wheel uses
CUDA 12.4. `environments/reference.lock` fixes NumPy 2.2.6, SciPy 1.15.3,
JAX/JAXlib/CUDA12 plugin 0.8.3 and their dependencies. The reference environment
must pass float64 GPU-versus-CPU algebra qualification in an allocated GPU job.
A dependency check or import check alone is not GPU qualification.

On Karolina, prepare on a network-enabled login node using the verified system
Python 3.11, with all environments and caches in private scratch:

```bash
export PF_SCRATCH=/path/to/private/posteriorforge
export PF_BOOTSTRAP_PYTHON=/usr/bin/python3.11
bash scripts/prepare-environments.sh
"$PF_SCRATCH/envs/inference/bin/python" -m pip install --no-deps --no-build-isolation .
"$PF_SCRATCH/envs/reference/bin/python" -m pip install --no-deps --no-build-isolation .
/usr/bin/python3.11 scripts/prepare-model.py --cache-dir "$PF_SCRATCH/cache/model"
```

`environments/bootstrap.lock` pins the uv resolver with hashes. The preparation
script synchronizes the two complete dependency sets and checks installed
metadata. No BayesArena or MesoUQ dependency, checkout, credential, or package
index is needed. Do not install the inference and reference extras together.
The runtime environments use pinned dependencies; package-only CPU CI needs
neither a checkpoint nor CUDA downloads.

The baseline regressor checkpoint is `tabpfn-v2-regressor.ckpt` from
[Prior-Labs/TabPFN-v2-reg at immutable revision
02a44705ff7790b50cded404bc59adb94da456a1](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/tree/02a44705ff7790b50cded404bc59adb94da456a1).
Its 44,390,977 bytes must have SHA256
`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.
Preparation verifies both size and hash before atomically accepting the model.
Weights and downloaded caches remain outside Git. Both upstream implementation
and model license notices are packaged in `posteriorforge/upstream` and model
attribution is recorded in `provenance/model-tabpfn-v2.json`. Use an explicit
existing checkpoint path and offline environment flags in compute jobs. A hash
failure is an error; a missing checkpoint is not permission to fetch on compute.

The six original NPE-PFN Python files are byte-identical to upstream commit
`29c1e900a89781ee00677df6d2415a6664e59c37`; their hashes are in
`provenance/upstream-npe-pfn.json`. The wrapper loads `npe_pfn.npe_pfn` directly
without executing upstream's package `__init__`. That initializer eagerly imports
an unrelated TSNPE training path; the released SBI 0.23.0 wheel **and source
archive** omit `sbi.samplers.score`, which that training import requires. The
namespace wrapper avoids this packaging defect while preserving all sampling,
regression, support, and rejection source bytes and all requested version pins.
The TSNPE training path is outside this project's scope and is not exposed.

`environments/frozen-baseline.json` records the exact regressor settings and
pinned internal preprocessing defaults. The same regression weights are reused
for all 21 autoregressive stages. Raw observations are supplied, context
filtering is disabled, the ensemble has eight estimators, random_state is zero,
precision is explicitly `torch.float32`, fit mode is `fit_preprocessors`, and
memory saving remains `auto`. One sklearn worker avoids oversubscribing allocated
CPUs. No prior-support timeout may return invalid candidates: baseline calls use
`max_iter_rejection=None`; an external job timeout is an incomplete failed run.
Uncorrected upstream log densities are not used as densities of accepted samples.

The digest-pinned `containers/inference.Dockerfile` installs the inference lock
and public package. It intentionally contains no model weights. The recipe still
requires a real image build and GPU smoke check before any container-qualified
claim. Host Python/venv qualification does not qualify that image. Mount the
verified model read-only and pass its path to the CLI. Slurm jobs use prebuilt
venvs and cached weights offline, with the host driver supplied by Karolina.

To intentionally update dependencies, edit the corresponding `.in` file, use
uv 0.13.7 `pip compile --python /usr/bin/python3.11 --generate-hashes`, inspect the
lock diff, synchronize a new environment, and rerun CPU and allocated GPU
qualification. Do not silently resolve or upgrade packages inside a compute job.

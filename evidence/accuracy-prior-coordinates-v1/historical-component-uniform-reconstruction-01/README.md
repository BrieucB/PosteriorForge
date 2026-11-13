# Authenticated historical component uniforms: CPU reconstruction

Built with PriorLabs-TabPFN.

This newly authorized diagnostic reconstructs **only the component-uniform
stream** for failed v2 case0 manifest `92755b78...`. It is RNG reconstruction,
**not a captured ICDF-input witness**, sampler/model execution or a recovered
posterior. No member, bin, logits, probability, CDF or interpolation fraction is
reconstructed or inferred.

The frozen source chain authenticates the original 3175593578 private CPU
`torch.Generator`, one `torch.rand(51200, dtype=torch.float32, device='cpu')`
vector before each stage's branch/head retries, assignment by physical row,
and retained row uniform across retries. The bound production factory is the
canonical source-checked 23cc factory; 4037's reverse AST changes only observation
width. Actual observer records confirm one complete 51,200-row wave and all 21
sequential returns. **Every one of the 21 before/after state-hash pairs matches**
the reconstruction, and the final state also matches the actual failure footer.

Only the two requested scalars are retained:

| Stage | Row | Reconstructed original FP32 U | Exact little-endian bytes |
|---:|---:|---:|---|
|15|4355|0.9999980926513672|`e0ff7f3f`|
|19|41765|0.9999994039535522|`f6ff7f3f`|

Both are strictly below one. They do not prove that the rounded terminal CDF
was below U, that the sample exceeded its actual canonical border, or which
member/bin was accepted. Those actual historical arrays/assignments remain
unavailable. No new Float64 uniforms or alternate random stream is substituted.

The successful replay performed **21 actual `torch.rand` calls**, generating
**1,075,200 uniform values**, with one local generator construction, one local
`manual_seed` and 42 private state-hash reads. These are explicitly nonzero RNG
calls, not new posterior/world/reference/G3 draws. Only the two authorized
values are extracted; full regenerated uniform vectors are not archived.
Default NumPy/Python/TorchCPU RNG hashes match before/after with **no restoration
or global manual_seed**. CUDA remains uninitialized. The exact recorded runtime
is Python 3.11.13, NumPy 1.26.4, Torch package 2.6.0/module 2.6.0+cu124.

Attempt 01 is preserved as an initial **zero-RNG** preflight stop: the diagnostic
compared package metadata 2.6.0 with module version 2.6.0+cu124. The separate
attempt 02 uses the original producer's importlib.metadata convention, preserving
both raw attempts/scripts/logs. No historical state/shape/call-chain mismatch was
ignored or relaxed, and no failed request/source/array was altered.

Raw successful diagnostic SHA-256:
`b6c3e08ce2897d5fead7d129dbbf3e2cae8cd37fa7986903899fb68471d8ac24`.
Source-before/after hashes, all 42 state comparisons, counts, exact source-chain
checks and runtime/default-state evidence are retained in the raw JSON. No
CUDA/GPU allocation, neural call, finite-bar inverse, codec, simulator, reference
integration, sampler recovery, policy change, ledger/registry write or G3 is
performed. Scripts ran only on Karolina in the original inference environment.
`SHA256SUMS` excludes itself.

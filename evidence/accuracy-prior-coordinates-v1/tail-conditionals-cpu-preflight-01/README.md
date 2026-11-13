# Two-prefix capture: focused CPU checks only

Built with PriorLabs-TabPFN.

The exact new standalone capture source and its focused tests pass inference
**13/13**, reference **11 passed / 2 missing-Torch skips**. Tiny deterministic
CPU logits exercise whole-batch inverse observer parity; no neural model or
posterior draw runs. The actual Torch-free input preflight authenticates the two
declared N1024 windows and reconstructed uniforms in 0.9481471329927444 seconds.
Exact raw preflight and logs are preserved. NPZ input arrays remain on Karolina.

The final source is intended only for Root's one conditional capture at most
300 GPU-seconds, not a production repair or full posterior. No GPU reservation,
submission, model/world/reference/G3 work or frozen-source change is performed
by this CPU receipt. Publication/CI is not claimed as a launch prerequisite.
Source/test hashes are recorded after the test runs; no nonexistent independent
pre-run hash snapshot is claimed. `SHA256SUMS` excludes itself.

Serialized-code fingerprint context diagnosis
Built with PriorLabs-TabPFN.

The first strict consumer stopped before any G3 draw or output directory:
execution_policy.derivation.run_code_sha256 was the ONLY differing field.
Both values were strings containing 64 lowercase hexadecimal characters.
The immutable successful production62aa recorded8c8ffeb9...; the retained
CPU fixture5238d0 (same actual run codeobject, explicit mocked neural/CUDA
boundaries) recorded exactly the same8c8ffeb9.... Three fresh independent
processes instead returned45b0ba56.... All other execution-policy fields,
the a30e producer source and pinned runtime source inventory were identical.

In an independent read-only process, merely retaining one reference to the
SAME live run code object's co_consts changed marshal SHA25645b0ba56... to
65be2546.... Deleting that reference restored45b0ba56.... The code object
identity and source stayed unchanged; run/main/inference were never invoked.
Thus raw marshal serialization is context sensitive, not a canonical source
content hash. This witness does not reconstruct every original active-frame
reference count. Definition-only module-name tests both yielded the fresh
hash: module name alone is not established as the cause.

diagnostic.json preserves exact values, types, paths/checksums, fresh-process
observations, reversible witness, limitations and the narrow proposed
consumer scope. The failed-consumer log is exact, not rewritten. At capture
the requested G3 output path did not exist. No model, GPU, G3, uniform/normal
draw, reference integration or numerical metric was generated here.

The narrow consumer correction must retain both serialized-code values as
separate telemetry, require every other metadata field exactly (including
types), and retain recorded147-Git/source-file/reverse-AST/live-closure guards.
Neither the producer/manifest nor a numerical guard is changed or waived.

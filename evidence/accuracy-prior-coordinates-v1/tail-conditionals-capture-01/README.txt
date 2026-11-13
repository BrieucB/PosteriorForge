COUNTERFACTUAL CONDITIONAL CAPTURE — METADATA PACKAGE

This package preserves the exact raw manifest and file descriptors from one
bounded counterfactual FP32 conditional/ICDF capture, job 5411602. It contains
no dense arrays, fitted-object pickles, or full CUDA trace. Built with
PriorLabs-TabPFN.

SCOPE
Two fixed saved-prefix windows were evaluated: ordered stage 15, raw rows
[4096,5120), focal row 4355 (offset 259); and stage 19, rows [40960,41984),
focal row 41765 (offset 805). Both use the authenticated seed20261111/case0
training bank, fixed SS15 inputs, latent labels and masks. The original arm
is fresh stock FP32 fit_preprocessors all8, outside a precision Session;
the candidate arm uses the frozen0199 cached FP32 path at these two stages.

All8 heads have N1024 captures in both arms. The predeclared trigger additionally
caused all8 N1 captures for stage15 in both arms. There are48 selected-head query
capture files total; the actual CDF footer records48 calls. The exact raw
manifest retains every original comparison/trigger/status field. This public
summary performs no numerical interpretation; Science's separate offline
analysis owns that report.

These are NEW COUNTERFACTUAL observations. The contiguous1024 windows are not
authenticated historical member-grouped batches. No historical selected
member, bin, probabilities, logits, or ICDF result is claimed to have been
recreated. Component uniforms were independently reconstructed from historical
private-stream states and supplied as fixed values; no new uniform stream was
drawn by the capture. Additional declared fixed U values and the u=1 guard
stress are diagnostic inputs, not posterior samples.

EXACT BYTES / DESCRIPTORS
capture-manifest.json.gz decompresses byte-for-byte to original manifest SHA:
cf11be3c542eb11c7ed9423845eb1f80fb046df59a2b85254cb9f20d25663461.
The immutable raw SHA sidecar and gzip/raw descriptors are included.
raw-artifact-descriptors.json records each of48 NPZ files' exact file SHA,
disk size, and all actual array dtype/shape/SHA descriptors copied from the raw
manifest. Every file SHA was verified by streaming its bytes; no NPZ was loaded.
Fitted configuration/target-transform pickle file hashes and trace descriptors
are preserved, with those raw files kept on HPC. Small raw Slurm stdout/stderr
are included exactly, together with submission, sacct, allocation receipt,
root launch preflight, and exact input CPU preflight records.

ACTUAL CLOSURE / COST
The raw success footer records source identity unchanged and exact default
NumPy/Python/Torch CPU/CUDA RNG equality; corrected CDF executed48 times and the
alias/SDK source restored. Both candidate precision Sessions restored their
aliases/flags. The report honestly scopes internal preprocessing RNG calls as
not instrumented, rather than claiming zero internal calls.
Capture elapsed time:48.31360623938963s, including tracing/serialization.
This is not a completed posterior request or clean performance latency.

Slurm job5411602 completed with1 GPU,53 allocated GPU-seconds, zero restarts,
and explicit --no-requeue, within a300-second reservation. The exact allocation
receipt is SHA f2779102404a8110599ac5f988d93899268250a7ce5e87344adfe60bd6b24716.
Root reconciled project actual104974 GPU-seconds and zero reserved after this
allocation. Raw submitted and reconciled records are distinct and retained;
the submitted snapshot's zero actual cost is not the final accounting.

NO POSTERIOR / RECOVERY CLAIM
The original failed request92755b... and all earlier diagnostic packages remain
unchanged. This capture produced zero posterior vectors, G3 updates, reference
integrations, or FP16 activations. It changes no production source, codec,
sampler, prior, feature, threshold, or policy and performs no repair, clipping,
filtering, salvage, or retry posterior. The full trace is SHA
21869db939e39d81326679fd1ac7addd9867a6bd1718acaa087ca3561c0b7629
and remains on HPC; its elapsed CUDA spans are not exclusive kernel unions.
Any normalized reference analysis and numerical conclusions belong to the
separate subsequent Science report, not this curation package.

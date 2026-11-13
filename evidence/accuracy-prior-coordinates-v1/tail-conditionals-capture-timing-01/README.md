# Two-prefix conditional capture: saved-trace timing

Built with PriorLabs-TabPFN.

The closed capture `cf11be3c...` completed both required **N1024** windows, all
eight original and cached FP32 members, at stages **15** and **19**. Stage 15
triggered and completed N1 for both arms; stage 19 did not trigger N1. Every new
prediction is **counterfactual reevaluation**, not recovered historical member or
batch output. Exact full capture metadata, source, timing records, stdout/stderr
and the offline analysis source/result are preserved. All density arrays and
the **192,909,213-byte raw trace** remain on Karolina outside Git.

The offline summary verifies raw trace SHA-256
`21869db939e39d81326679fd1ac7addd9867a6bd1718acaa087ca3561c0b7629`
and uses the existing pinned 1852 streaming reader and 328 launch-correlation
routine in new private namespaces, following 4707's host-only annotation rule.
Only owner-thread host `pf-tail` annotations are mapped. GPU range duplicates
remain distinct. One **virtual entire-profiler extent** is supplied to the
legacy request-oriented reader without changing raw trace bytes; its field
`profiled_request_trace_span_seconds` means that virtual extent here, not an
actual request boundary or a clean posterior benchmark.

For the **entire 40.230329280761715-second profiler extent**:

- GPU kernel interval union: **5.505771322265625 seconds**.
- GPU copy interval union: **0.6474881254882813 seconds**.
- GPU kernel/copy/**memset** active union: **6.156533650390625 seconds**. The legacy
  reader key mentions kernel/copy but its source includes memset too.
- Host CUDA synchronization interval union: **0.6035097109375 seconds**.
- Zero launch-correlation conflicts; no unmatched kernel events. Unmatched copy
  work remains explicit: **3,052 events**, **0.08254865137860179-second union**.

These intervals **overlap and must not be added**. Per-stage/arm kernel/copy
unions and actual operator/kernel names are in `timing-summary.json`. The trace
observes 1,824 `fmha_cutlassF_f32_aligned_64x64_rf_sm80` GPU attention kernels;
operator counts include generic/nested CPU capability operators and do not
independently prove GPU FlashAttention or exclusive backend-call counts.

Owner neural host-annotation union is **16.602075525559485 seconds**. The
remaining **23.62825375520223 seconds** is a combined unbracketed host extent:
ICDF/observer, weights/normalizer/target guards, capture serialization/I-O,
Python gaps, waits and cleanup. It **cannot** be split into separate ICDF versus
capture/I-O wall costs from this instrumentation. Recorded host call totals are
original fit 0.866454563s, candidate prepare 4.793064173s, original N1024 5.152354252s,
candidate N1024 3.300670138s, original N1 1.204370801s and candidate N1 1.291433793s;
these include waits/transfers and are not CPU compute. CUDA-event spans are
explicitly not exclusive kernel times.

The capture recorded **48.31360623938963 seconds** elapsed and
**1.5191079978831112 seconds** trace export. Offline parsing took
**16.84746124688536 seconds**, with Torch absent. The elapsed-versus-profile
remainder includes unpartitioned startup/guards/export; no isolated startup or
clean-request speedup is inferred. Root owns terminal Slurm accounting; this
package does not reinterpret the launch reservation as actual cost. No 50,000
extrapolation, production optimization or posterior accuracy claim is made.

Curation/offline parsing executes no model, sampler, ICDF, codec, new RNG, world,
reference integration, GPU, G3, policy change or ledger/registry write. Frozen
capture/helper/tests remain unchanged. The successful finite screen and prior
failed posterior remain separately scoped. `SHA256SUMS` excludes itself.

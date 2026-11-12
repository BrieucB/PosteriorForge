Failed prior-coordinate pilot: Float64 FIFO integration boundary
Built with PriorLabs-TabPFN.

This is an integration failure, not a codec failure or a scientific accuracy
assessment. Job 5411061 requested 50,000 posterior vectors for seed 20261111,
case 2. It saved no samples.npz and no completed posterior cohort. No physical
joint-global Gibbs update or physical accuracy assessment followed this failure.

The original all 21 actual stage-return records show one Float32[51200,21]
latent wave. The complete wave, including the unused FIFO tail, passed strict
decode to Float64[51200,21]. Its values remain archived on Karolina; this public
package contains metadata, hashes and the exact original manifest only.

The immutable FIFO8fb dtype guard then raised:
  ValueError: unmodified CPU FP32 vectors required; no implicit casting
The provider had generated 51,200 rows, while FIFO bookkeeping correctly
remained at generated 0 and delivered 0 because its dtype guard had not accepted
the provider return. The original API requested 50,000 and the buffer was not
finalized. Source inspection places this exception before support delivery;
the normal original-support counter footer was not recorded.

The saved corrected-CDF adapter reports 1,138 calls and restored original alias
and unchanged SDK sources. These observed fields are retained as such.
The normal-success execution journal, final default CPU/CUDA/Python/NumPy RNG
hashes, request timing, GPU memory peaks and success source footer were NOT
recorded. The sampling_counters, actual_prediction_checks, expected CDF/head
and prepare counts, and generated_proposal_wave_count success footer fields
are also absent. They must not be inferred from stage completion, cleanup source, or
the separate diagnostic-only NumPy/Python RNG observation. The original
completed_stage_indices field remains empty; actual stage-return coverage is
separately recorded as 0..20. The machine status remains failed.

An independent science-agent read-only archive audit reported CLEAR within
this failure-custody scope. It did not supply the missing success observations.
No new model calls, posterior draws or reference integrations were used to
assemble this package. The recorded Git revision is b25365a1275b5e9a4d549612a81674ad9002d15f;
the original git_dirty field is TRUE. A clean checkout is not claimed.

allocation-receipt.json and raw Slurm-accounting.psv preserve the reconciled
FAILED job: 1 GPU, 752 allocated GPU-seconds, 0 restarts, no requeue, original
1,200 GPU-second reservation. The read-only ledger observation is 102,497
actual GPU-seconds (28.4713888889/200 GPU-hours), 0 reserved, 123 entries.
The recorded wrapper interval was 748.5380248269 seconds. stage-timings.json
preserves all 21 original stage intervals: preparation sum 84.5931307292s;
predict-and-sample sum 644.3417515010s. Fit/promotion subregions can overlap
preparation. These are failed-execution timings, not a completed posterior
request latency or throughput; its normal-success timing footer is absent.

manifest.json.gz decompresses byte-for-byte to the original 3,559,587-byte
manifest with SHA256 c81001f878bc1cd04d144c4d98a22e2ab5a089698b58978e10773c33654bd488.
summary.json retains the observed/absent distinctions and hashes of the three
HPC-only array archives. Raw sample/proposal arrays are not included.

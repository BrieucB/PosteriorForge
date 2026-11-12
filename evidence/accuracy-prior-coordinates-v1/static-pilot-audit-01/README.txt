Static audit of the failed prior-coordinate pilot
Built with PriorLabs-TabPFN.

This source-only audit concerns immutable failed manifest c81001f878bc1cd04d144c4d98a22e2ab5a089698b58978e10773c33654bd488.
It adds no model, profiler, GPU, posterior, simulator or reference work.

grouped-member-audit.json pins exact source SHA256 values and one-based line
ranges. The original 23cc sampler still groups rows by assigned member before
single-component predictions. Stage 0 constructs all 8 singleton heads once.
The cached e71 path filters executor lists before prediction; native 6/10 use
the selected-preprocessing-member stock path with exactly 1 corrected CDF
translation per call. All 8 members remain in the prepared ensemble. There is
no unintended full8 query path in the inspected sources.

The saved stage-event histograms imply 1 shared callback plus 1130 selected-head
callbacks (1131 callbacks), 1138 physical head calls and 1138 recorded corrected
CDF calls. These are derived from EXISTING recorded events, not recovered
sampling_counters, execution_counters or actual_prediction_checks footers.
The latter were never saved for the failed request. No kernel trace was run.

Native stages 6/10 account for 81.6859244760s of 644.3417515010s recorded total
predict-and-sample time (12.6774%). They cannot explain the whole runtime.
The failed wrapper interval was 748.5380248269s. No successful 50,000-vector
latency or throughput is claimed. Inclusive preparation subregions are not
double-counted.

git-dirty-evidence.json preserves the original git_dirty=True observation
and recorded b25365a1275b5e9a4d549612a81674ad9002d15f revision. The original
manifest did not record dirty filenames. POSTHOC-git-status.txt is explicitly
a later checkout snapshot, not an invented exhaustive runtime inventory.
The five older candidates supplied by the coordinator are listed only with
their status in that later snapshot. A bounded scan of the original job's
stdout/stderr is identified by exact hashes and recorded separately.

The original failed request remains failed. Its missing final RNG/source/
execution/sampler guards are not inferred. The corrected source closure must
be committed and pinned before a later separately authorized production run.

# Seed 20261111, case 3: model_failure

Built with PriorLabs-TabPFN. This is the frozen **hybrid NPE-PFN/Gibbs** method:
sigma-first local-block neural conditionals, followed by one exact joint refresh
of all three globals. All 50,000 complete physical vectors are assessed.

The case-specific qualified reference and unchanged conservative limits give
marginal CDF maximum 0.231336352, joint CDF maximum
0.126480968, and mean-bias maximum
0.585035025 reference SD.
Failed marginal coordinates: mu_a, a_m1, b_m1, d0_m1, sigma_m1, a_m2, b_m2.
Failed mean coordinates: mu_a, mu_b, a_m1, b_m1, d0_m1, sigma_m1, a_m2, b_m2.
Failed joint CDF indices: 10, 15.
Failed SD-ratio coordinates: sigma_m1.
The full SD-ratio interval range is [0.751748870, 1.199239471].
Before the exact global refresh, the parent status was model_failure, with
marginal CDF maximum 0.231336352, joint CDF maximum
0.144119069, and mean-bias maximum 0.585035025.
Its failed mean coordinates were mu_a, mu_b, mu_d0, a_m1, b_m1, d0_m1, sigma_m1, a_m2, b_m2;
failed joint indices were 10, 15, 19.
Both before and after summaries are preserved; the exact global kernel is not
claimed to improve every finite-cohort criterion.

The predeclared focus uses indices [0,4,11,12]; every coordinate remains in the
full density plot and exact assessment. No failure is omitted.

![focus density](joint-global-density-focus.png)

![density](joint-global-density.png)

The JSON assessment, source figure bytes and checksums are unchanged copies.
The two gzip archives decompress to the exact original producer manifests;
their raw SHA256 sidecars verify that identity. Payload hashes are retained,
but raw sample arrays are not included.

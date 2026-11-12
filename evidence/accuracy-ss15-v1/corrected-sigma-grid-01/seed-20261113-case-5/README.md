# Seed 20261113, case 5: model_failure

Built with PriorLabs-TabPFN. This is the frozen **hybrid NPE-PFN/Gibbs** method:
sigma-first local-block neural conditionals, followed by one exact joint refresh
of all three globals. All 50,000 complete physical vectors are assessed.

The case-specific qualified reference and unchanged conservative limits give
marginal CDF maximum 0.191410711, joint CDF maximum
0.126484988, and mean-bias maximum
0.425237116 reference SD.
Failed marginal coordinates: a_m1, b_m1, a_m2, d0_m2, sigma_m2.
Failed mean coordinates: a_m1, b_m1, a_m2, d0_m2, sigma_m2.
Failed joint CDF indices: 5, 13.
Failed SD-ratio coordinates: a_m1, b_m1, d0_m1, sigma_m2.
The full SD-ratio interval range is [0.751292640, 1.352833815].
Before the exact global refresh, the parent status was model_failure, with
marginal CDF maximum 0.191410711, joint CDF maximum
0.124602505, and mean-bias maximum 0.425237116.
Its failed mean coordinates were mu_a, mu_b, a_m1, b_m1, a_m2, d0_m2, sigma_m2;
failed joint indices were 5, 13.
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

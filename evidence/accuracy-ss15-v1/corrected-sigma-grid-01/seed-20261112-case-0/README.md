# Seed 20261112, case 0: model_failure

Built with PriorLabs-TabPFN. This is the frozen **hybrid NPE-PFN/Gibbs** method:
sigma-first local-block neural conditionals, followed by one exact joint refresh
of all three globals. All 50,000 complete physical vectors are assessed.

The case-specific qualified reference and unchanged conservative limits give
marginal CDF maximum 0.112156079, joint CDF maximum
0.056999913, and mean-bias maximum
0.244627919 reference SD.
Failed marginal coordinates: b_m3.
Failed mean coordinates: a_m3, b_m3.
Failed joint CDF indices: none.
Failed SD-ratio coordinates: none.
The full SD-ratio interval range is [0.896784914, 1.050701877].
Before the exact global refresh, the parent status was model_failure, with
marginal CDF maximum 0.112156079, joint CDF maximum
0.070524870, and mean-bias maximum 0.244627919.
Its failed mean coordinates were mu_a, mu_d0, a_m3, b_m3;
failed joint indices were none.
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

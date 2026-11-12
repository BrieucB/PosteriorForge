# Seed 20261111, case 4: model_failure

Built with PriorLabs-TabPFN. This is the frozen **hybrid NPE-PFN/Gibbs** method:
sigma-first local-block neural conditionals, followed by one exact joint refresh
of all three globals. All 50,000 complete physical vectors are assessed.

The case-specific qualified reference and unchanged conservative limits give
marginal CDF maximum 0.098715089, joint CDF maximum
0.074393863, and mean-bias maximum
0.205064535 reference SD.
Failed marginal coordinates: none.
Failed mean coordinates: sigma_m2.
Failed joint CDF indices: none.
Failed SD-ratio coordinates: none.
The full SD-ratio interval range is [0.915038698, 1.027220162].
Before the exact global refresh, the parent status was model_failure, with
marginal CDF maximum 0.098715089, joint CDF maximum
0.066356505, and mean-bias maximum 0.205064535.
Its failed mean coordinates were sigma_m2;
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

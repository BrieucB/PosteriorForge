# Seed 20261111, case 2: model_failure

Built with PriorLabs-TabPFN. This is the frozen **hybrid NPE-PFN/Gibbs** method:
sigma-first local-block neural conditionals, followed by one exact joint refresh
of all three globals. All 50,000 complete physical vectors are assessed.

The case-specific qualified reference and unchanged conservative limits give
marginal CDF maximum 0.116391369, joint CDF maximum
0.091391515, and mean-bias maximum
0.272693301 reference SD.
Failed marginal coordinates: a_m2, b_m2.
Failed mean coordinates: a_m2, b_m2.
Failed joint CDF indices: none.
Failed SD-ratio coordinates: none.
The full SD-ratio interval range is [0.930752221, 1.092127066].
The predeclared focus uses indices [0,4,11,12]; every coordinate remains in the
full density plot and exact assessment. No failure is omitted.

![focus density](joint-global-density-focus.png)

![density](joint-global-density.png)

The JSON assessment, source figure bytes and checksums are unchanged copies.
The two gzip archives decompress to the exact original producer manifests;
their raw SHA256 sidecars verify that identity. Payload hashes are retained,
but no raw samples or private CV material are included.

# Post-pilot cost addendum

Root supplied the closed corrected production manifest `62aa8bf7...`, allocation
receipt `eb8f8f03...`, and reconciled job 5411248 accounting: **769 allocated GPU-
seconds, one GPU, zero restarts**, no requeue. The recorded production request
returned 50,000 physical Float64 vectors. Source/default-RNG and complete21 guards
are recorded as passing. This is a technical completion update; physical quality
assessment and any wider-panel execution decision remain separate.

Replace only the old 752-second failed-pilot proxy in the original planning cost:
23 × (426 + 769)= **27,485 GPU-seconds = 7.6347222222 GPU-hours**. It remains a proxy,
not a promise that other contexts/cases finish, pass numerical screens or meet
physical gates. The conservative unchanged 23-pair reservation plan is still
**55,200 GPU-seconds = 15.3333333333 GPU-hours** with separate 1,200-second one-GPU
caps for each screen and posterior request. No reservation or ledger write is
made by this addendum; no grid job is submitted or authorized by it.

Root's reported ledger after reconciliation is 103,266 actual GPU-seconds,
**28.685/200 GPU-hours, zero reserved**, 124 entries. The original plan README
SHA 0bea4231... and bindings/cost JSON SHA dfd31d91... remain byte-for-byte
unchanged. All finite-prefix, fixed-policy, failure/inconclusive-retention,
actual-pair custody and non-population-calibration caveats still apply. The old
c810 request remains failed; its forensic pass is not production credit.

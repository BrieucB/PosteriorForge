# V2 case 0 production request: strict inverse failure

Built with PriorLabs-TabPFN.

The single request **seed 20261111 / case 0**, context 4,096, repetition 0,
job **5411530**, ended with literal status **`failed`** and `CoordinateError`:
`decode: endpoint, out-of-support, or saturated noise; no clipping`.
All 21 neural stage calls returned and one **51,200 × 21 Float32 latent wave**
was preserved before decode. Strict full-wave physical inversion failed before
the provider returned to the Float64 FIFO or original support selection.
**No physical 50,000-vector cohort, sample file, valid decoded full-wave archive,
physical assessment or G3 update exists for this request.** No salvage or retry
is performed by this package.

The source policy remains FP32 original all-eight execution at stages 6, 10 and
14 and cached FP32 at the other 18. The separately passed same-pair finite screen
`aa8aff31...` remains passed as a finite-prefix numerical screen. It did not
establish that all generated tail values would satisfy the strict codec.

The raw manifest's bounded CPU diagnostic records all 51,200 rows examined,
zero raw nonfinite cells, 51,198 rows valid and two finite failing rows. Recorded
localizations are row **4,355**, causal stage **15**, physical slot **16**
(`sigma_a1`), latent `13.876729011535645`; and row **41,765**, stage **19**,
slot **20** (`sigma_a3`), latent `205.2462615966797` (zero-based indices).
These are **post-failure codec-only diagnostic recomputations**, not captured
PFN probabilities, accepted head/bin/uniform witnesses or internal state from
the first failing inverse. Full recorded bounds, raw row/scalar bytes and
last-valid-prefix diagnostics are preserved without rerunning them.

`archive-descriptors.json` preserves the complete latent-wave and stage-return
archive descriptors and exact outside-Git NPZ hashes. **All actual arrays stay
on Karolina outside Git**, including the unused lookahead tail. No row is
clipped, repaired, discarded, replaced or accepted here.

The new observed failure footer is preserved separately from missing normal
success fields. Its actual post-exception observations record restored stage
aliases, source closure pass, unchanged default RNG comparison, **1,141 actual
corrected-CDF calls** with original alias/SDK restoration, and original support
counters zero. FIFO reports failed, not finalized, attempted wave 51,200,
generated/delivered bookkeeping zero. Its zero generation count reflects the
provider failing before return, not absence of neural computation. The top-level
`completed_stage_indices=[]` remains literal; the distinct stage-return journal
records returns 0 through 20. Likewise the top-level requirement declaration
`all_generated_rows_strictly_decoded_including_unused_FIFO_tail=True` is preserved
as a policy declaration, **not a success result**: the actual decode record is
False. No absent success timing, count, RNG, source or memory footer is fabricated.

Root reconciled Slurm **FAILED**, **782 allocated GPU-seconds**, one GPU, zero
restarts, no requeue; `.batch`/`.extern` child rows are retained and never added.
Wrapper wall time is **777.6754412082955 seconds**; no successful request time is
available. The exact allocation receipt records **104,921 GPU-seconds**
(**29.14472222222222/200 GPU-hours**), zero reserved, at reconciliation. These are
cost records, not performance or accuracy claims.

`manifest.json.gz` decompresses exactly to raw manifest SHA-256
`92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce`
(**16,490,005 bytes**). Exact allocation SHA-256 is
`af5ec09a4a78156c4c51443e353df8e3fa55756dfe1235a5702a91d9166eb6a7`.
The closed stderr preserves the original traceback. Curation selects recorded
metadata only, performs no array/scientific recomputation, source edits, model
calls, draws, reference integrations, GPU or ledger/registry/G3 actions, and
commits nothing. Earlier failed `c810...` and numerical-failure `fbed...` records
remain unchanged. `SHA256SUMS` excludes itself.

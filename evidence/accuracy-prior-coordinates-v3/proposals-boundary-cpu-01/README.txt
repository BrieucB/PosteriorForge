V3 decoder: narrow boundary and numerical-scope supplement
Built with PriorLabs-TabPFN.

The decoder module9216e76c and ALL inverse/validation mathematics are UNCHANGED.
Only one focused engineering regression was added. Final tests37/37 PASS in each
locked environment, no skips; exact sources/commands/stdout are in receipt.json.

Exact versus numerical scope
The exact rational object is the solution of the original physical-error
inequality. Validation PHYSICAL coordinates are rounded inward to that interval.
Their CDF/logSF and input log_ndtr values are then evaluated by ordinary SciPy
Float64 functions. These are NOT certified directed probability bounds.
Consequently the implementation and23-scalar high-precision evidence do NOT
establish a universal, all-input certified physical-error theorem. No probability
or physical tolerance has been added or relaxed, and original proposal bytes are
never altered by validation probes.

Near-budget engineering cases
For central noise latent z=0, each physical budget boundary and its two adjacent
Float64 cells on either side were tested (10 cases/environment). All4 exact
outside-budget cells were rejected in both environments. No false acceptance
was observed. One last-inside lower-side cell was conservatively rejected due to
the inward physical probe discretization; the next inside cell passed. This
conservatism is explicit and does not widen the original error budget. These
cases are regression evidence, not a universal certified numerical-CDF claim.

No original frozen d8/module/formula/global was edited. No density, NN/model
constructor, posterior/cohort, GPU, G3 or RNG draw occurred. Earlier CPU evidence
proposals-cpu-01 stays unchanged; this supplement carries the final37-test source
and additional scope detail. Zero neural-qualified pairs remain.

# Paired-bank provider interface

`posteriorforge.banks.PairedBankProvider.get_bank(seed=...,count=...)` returns a
`PairedBank` with float64 `theta[n,21]`, `obs[n,33]`, and a provenance manifest.
Each row is one complete joint-prior world and its corresponding observation.
Providers must preserve this pairing, the exact parameter/observation order,
full inference-prior support, noise definition, seed derivation, and source/cost
metadata. Context sizes are prefixes of complete4096-world banks.

`AnalyticalBankProvider` is the implemented baseline. A future learned surrogate
or C/CUDA physical provider belongs behind this interface with its own scientific
fingerprint and independent validation. Such adapters are intentionally outside
this benchmark. Inference receives only held-out observations; held-out theta
labels exist for diagnostics and must never enter contexts or query features.

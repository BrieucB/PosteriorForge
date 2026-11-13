# Fixed 23-pair CPU preflight evidence

Built with PriorLabs-TabPFN.

All 23 predefined remaining seed/case pairs passed in both locked CPU
environments. The combined receipt is SHA-256
`6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193`.
The separate existing case 2 regression also passed actual raw/model data,
qualification seed and all 21 mask/target/query byte comparisons against 5ec.

This is **codec/domain/input/reference-custody evidence only**. No PFN/model
calls, new worlds, posterior draws, reference integrations, GPU submissions or
ledger writes occurred. It does not numerically qualify another pair's neural
execution or establish posterior/population calibration. The only next GPU
candidate authorized by the preregistration is the first lexicographic pair,
seed 20261111 / case 0, subject to a separate fresh all-21 matching screen.

Every input prefix is exactly the original pure pre-G3 parent's first 1,024 rows
in saved order, authenticated against completion receipt 613. No old physical
gate failure excluded a fixture. No row, case or endpoint was replaced, filtered,
clipped or repaired. Existing qualified references/refined grids were checked
with the frozen metadata-and-payload-SHA binder and matched each original pure
parent; their values were not used to fit or choose anything.

Float32 model inputs and all 21 masked training/target/query bindings match in
both environments for all 23 pairs. Float64 latent prefix bytes differ for all
23 pairs between NumPy/SciPy versions; both actual provenance records and strict
roundtrip reports are retained. Float64 cross-version byte identity is not
claimed or required. Default NumPy/Python RNG and all source hashes were
preserved; Torch was not imported.

Inference: Python 3.11.13, NumPy 1.26.4, SciPy 1.12.0; wall 106.083643729 s.
Reference: Python 3.11.13, NumPy 2.2.6, SciPy 1.15.3; wall 107.718125647 s.
`summary.json` records phase wall sums, source pins and exact archived-byte
hashes. `phase-and-outcome-log.json` is explicitly derived from closed manifest
records; no independent raw stdout capture exists or is fabricated.

The three gzip files decompress byte-for-byte to the original environment
manifests and combined receipt. These are metadata archives only; no NPZ or raw
sample/proposal arrays are included. Existing arrays remain on Karolina. The
Root-owned source commit is 7833bca. Focused CPU binding/reference negative tests
were 10/10 passing in each locked environment before execution (test source
`e9c53e6b...`); this package does not rerun them. `SHA256SUMS` excludes itself.

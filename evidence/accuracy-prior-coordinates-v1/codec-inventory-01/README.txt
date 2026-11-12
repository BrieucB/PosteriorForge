Prior-coordinate CPU codec receipt — Built with PriorLabs-TabPFN

This is a codec/domain preflight, not a backend numerical pass or a posterior
accuracy result. Both locked environments passed all original 3x4096 training
rows and the predeclared FIRST1024 saved case-2 physical rows. No endpoints,
nonfinite values, saturation or declared-roundtrip failures occurred in these
specific inputs. No row was clipped, repaired, filtered, or replaced.

Float32 latent and ordered-model descriptors agree across environments.
Float64 latent byte hashes DIFFER between SciPy1.15.3 and1.12.0; those versions
must not be described as bit-identical Float64 transforms. Both obey the same
frozen bounds. The full exact per-environment inventories retain per-coordinate
errors, descriptors and observed default NumPy/Python RNG-state hashes.

Physical errors use known-prior units, separately for Float64 and final
Float32 casting. The 8sqrt(eps64) latent bound is a declared conditioning/
roundtrip admissibility limit, not an ulp-Float64 accuracy guarantee.
Any future model latent with nonfinite values, physical endpoint saturation,
or failed roundtrip aborts the whole request; no endpoint atom or new rejection
rule is silently introduced.

Each gzip archive decompresses to its exact original manifest and raw SHA256
sidecar. CPU-receipt.json carries the source/prior/order/tolerance descriptors,
fixed-prefix custody and both-environment summaries. No raw arrays are copied.
The original observations, scientific problem and active scientific grid are
unchanged. No model, simulator, posterior request, reference metric or GPU call
was made by this qualification.

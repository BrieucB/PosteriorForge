Built with PriorLabs-TabPFN.

READ-ONLY MATHEMATICAL DIAGNOSIS; NOT RECOVERY OR A NEW POLICY
Failed request92755 and latent archive1ba89 remain unchanged. No physical
sample/cohort, G3, model/sampler/RNG draw, reference integration or frozen
source/codec/policy edit occurs. Metadata/scalar script only; no NPZ in Git.

d8 already uses log_ndtr(-z), logaddexp with the bounded-prior tail, then
ndtri_exp. It never forms naive Phi(positive z). At both actual inputs the
log addition collapses to the same Float64 base. The frozen inverse gives
-2.0000000000000004 and final physical scalar0.5: the STRICT predicate fails.

Using actual binary source constants and stable 180-digit tail-distance math,
the REAL quantiles are .5-9.49141115336e-44 and .5-1.18436262033e-9150. Both
are valid interior REAL numbers. Half the Float64 predecessor gap is
2.77555756156e-17. Correct Float64 rounding gives0.5 for THESE inverses;
this does not mean no other interior floats exist. The closest is
.49999999999999994. Substituting it is not correct rounding and cannot
preserve latent roundtrip: frozen encodez8.40512847 is far from13.8767/205.2463.
No endpoint/predecessor replacement is performed.

The gap solves int_0^t exp(bu-u^2/2)du=M*Q(z)/phi(b), gap=s*t. Rescaling
to[0,1] avoids subtracting a10^-9150 distance from.5. 180-digit mpmath is
not certified interval arithmetic. Independently, Mills Q(z)<phi(z)/z
and integral>=t give gap<s*exp((b^2-z^2)/2)/z, still many orders below the
rounding midpoint. Exact binary physical constants and rounded a=-2,b=2/
log_mass are recorded separately; normalization differences cannot alter
the conclusion. Precision bits141/30394 concern local separation only,
not sufficient full codec roundtrip accuracy. Original reportd56e2a is
preserved exactly; reproduced gap values agree beyond65 significant digits.

Reproduce from the checkout, in the INFERENCE environment:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /scratch/project/eu-26-17/eubrieucb/posteriorforge/envs/inference/bin/python evidence/accuracy-prior-coordinates-v1/noise-inverse-math-diagnostic-01/reproduce.py --parent /scratch/project/eu-26-17/eubrieucb/posteriorforge/results/accuracy-prior-coordinates-v1/native-pilot-v2-seed20261111-case0-01 --output /a/FRESH/path/scalars.json
Source/input SHAs are checked. Output is scalar JSON only.

See summary.json for ranked future decisions and limits. LogCDF-only or more
accurate intermediates followed by Float64 output cannot repair this contract.
Latent-tail provenance remains unidentified: actual selected logits/u/bin/
fraction were not used/reconstructed. Any sampler/representation/precision/
domain policy change requires an explicit parent decision and new evidence.
Nothing is implemented; no cohort is recovered.

# V2 failed case 0: source inspection and limits

Built with PriorLabs-TabPFN.

This note reads frozen source and already recorded scalar diagnostics from raw
manifest `92755b78...`. It does **not** execute a sampler, model, inverse codec,
new RNG draw, reference integration, recovery or G3. It makes **no cause claim**.
Exact source hashes and line-numbered excerpts are in `source-inspection.json`.

The actual inverse is finite-uniform-bar interpolation: FP32 softmax → cumulative
sum → searchsorted → clamped **bin index** → `left + width*(u-before)/p`.
The interpolation fraction itself is not clamped and the final CDF is not forced
to one (`execution_mixture.py:224–252`). This is the pinned SDK inherited
`BarDistribution.icdf` structure. Although `FullSupportBarDistribution` uses
half-normal formulas in its density, its `sample` calls that inherited inverse;
a half-normal tail draw is not the active sampled computation.

Two distinct source-compatible mechanisms must remain separate. Ordinary
interpolation in a wide finite edge bar may yield a large finite latent. Also,
if the rounded terminal CDF falls below the consumed uniform, clamping the
search result to the final bin does not ensure `(u-before)/p <= 1`; a small positive
denominator can magnify the excess into a finite extrapolation. This is a
conditional algebraic observation, **not proof it occurred**. The FP64 raw-mass
envelope guard does not control each final FP32 softmax/cumsum/interpolation.

Strict noise inversion already uses log tails, then requires the physical
value to be strictly between support endpoints. Finite input therefore does not
guarantee a representable interior physical output. The archived failure says
that endpoint/support test failed, not that the latent input was nonfinite.
Separate high-precision work can quantify representability; this note does not
repeat it or infer a specific posterior tail probability.

For the two recorded finite failures (row 4355/stage15/sigma_a1, latent
13.876729011535645; row41765/stage19/sigma_a3, latent205.2462615966797), the actual
accepted **member, uniform, logits/probabilities, bin, terminal CDF, CDF-before,
selected mass and interpolation fraction were not saved**. Aggregate head counts,
seed/state hashes and returned values cannot by themselves identify those
intermediates. We do not regenerate the random stream, rerun predictions or
substitute saved fixed-conditional outputs. A finite-prefix screen passing does
not cover every generated row or prove tail stability. No policy, source,
threshold, support, dtype, rejection or acceptance rule is changed.

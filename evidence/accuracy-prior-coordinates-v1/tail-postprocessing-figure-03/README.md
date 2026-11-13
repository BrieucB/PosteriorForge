# Final A/B/C comparison figure: caption/layout correction

Built with PriorLabs-TabPFN.

This final figure supersedes the e937e117 draft, which remains separately
preserved with its exact original renderer on Karolina. It shortens the two
left-axis labels to **Latent quantile z** and adds the requested explicit caveat:
**B replay p sums above 1; the source CDF clamp removes its far tail; its RT
passes do not validate a policy.** Legend/footer spacing was adjusted for
readability. The final PNG was visually inspected with no label/footer overlap.

Exact closed primary report remains
`cbc8e44e62780d48e594845bd63a9c72faa76eaf7a1459d9e4dbb2210e989a5a`.
The entire `plotted_rows` record is identical to the first render. This is only
a caption/layout revision; no numerical value, status, threshold, model,
postprocessing calculation or primary artifact changes.

It displays all 32 N1024 stage/arm/member rows (stages 15/19, original/candidate,
all eight members), at each stage's own authenticated reconstructed uniform. A is
exact-normalized inversion of captured final CPU F32 weights; B uses a labelled
CPU F32 pretranslation softmax replay and F64 postprocessing; C uses the same
already-adjusted F32 logits and F64 softmax/postprocessing. No second temperature
division is applied. Actual GPU pretranslation softmax was not captured.
Markers/counts copy the primary noise-only RT32/RT64 outcomes, not a new gate or
physical 21-vector/cohort claim. The numerical primary report is authoritative.

No model, preprocessing, codec, RNG draw, GPU, world, reference integration,
G3 or new numerical diagnosis runs. Rendering occurred on Karolina/reference
CPU; local viewing only inspected its PNG. The exact final renderer is
`scripts/plot_prior_tail_postprocessing_v1.py`; metadata binds its source SHA.
`SHA256SUMS` excludes itself.

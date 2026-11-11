# Explicit finite-CDF arithmetic variant

Built with PriorLabs-TabPFN.

`cdf_arithmetic_v1` is a separate scientific variant. It changes only the
parentheses in pinned TabPFN2.0.9's finite ensemble-CDF interpolation. It does not
change endpoint-tail sampling, checkpoint, raw observations, target order,
preprocessing, ensembles, precision, support rejection or context filtering.
The original default CLI and all released evidence remain unchanged.

The adapter guards the SDK version, utility/regressor source hashes and function
identities. It clones the original translator code with private globals referring
to the corrected helper, never edits site-packages, and restores the exact alias
on exit. Source activation consumes no random draws. A call counter verifies that
all8ensemble translations in every21stage proposal use the adapter.

`posteriorforge.inference_cdf` calls the byte-identical original inference runner
with its exact original arguments and RNG derivation. A scoped metadata writer
labels initial/error/final manifests as the new variant. Completion is withheld
until adapter/writer restoration and source checks pass. Adapter/import setup is
measured separately from the original request timer; outer total includes it.

## Collection and assessment

Run collection only through the existing BudgetLedger and Slurm launcher, in the
qualified inference environment. This example names placeholders deliberately;
use the authenticated original banks/references and a fresh variant output path.

```bash
python -m posteriorforge.inference_cdf \
  --banks BANKS --output FRESH_VARIANT_RUN --context-seed 20261111 --case 0 \
  --context-size 4096 --count 50000 --microbatch 10000 --repetition 0 \
  --checkpoint ORIGINAL_CHECKPOINT --ledger EXISTING_PRIVATE_LEDGER \
  --reference ORIGINAL_CASE0_REFERENCE --reference-grid ORIGINAL_CASE0_REFINED_GRID
```

The case-0 pilot uses exactly the original physical-coordinate criteria and
qualified reference; no reference is regenerated. Evaluate saved complete data
on Karolina with the reference environment:

```bash
PYTHONPATH=src python scripts/evaluate_cdf_pilot.py \
  --root PRIVATE_POSTERIORFORGE_SCRATCH \
  --pilot-manifest FRESH_VARIANT_RUN/manifest.json --output FRESH_ASSESSMENT
```

The helper authenticates the original gate, bank/query/reference chain, exact
adapter identity/activation/call coverage, sample payload, physical parameter
order and original53CDFwitnesses. It reuses `assess_samples` unchanged and renders
before/after densities, all21marginalCDFs and all32joint-projectionCDFs. A single
pilot pass only permits expansion to all24original4096-context combinations.
Failure or unresolved numerical intervals stop additional GPU work. Neither this
adapter's mathematical correctness nor a pilot pass is a full scientific pass.

## Conditional bottleneck work

Only after scientific acceptance can context reuse, batching, precision,
compilation or GPU replication be tested. The original trace identifies repeated
GPU attention as the dominant measured request cost; it does not prove a repaired
variant's breakdown, arithmetic/bandwidth limitation or a speedup.

The first proposed structural experiment reuses each stage/ensemble's prepared
context across posterior batches while preserving its model, transforms, feature
order, masks and autoregressive query prefixes. Measure cache memory and transfers
before selecting residency: a cache API that stores KV on CPU may replace compute
with large transfers. Keep stage-specific state isolated and compare distributions
against this corrected **unoptimized** control plus the unchanged absolute gate.
Precision/compile and independent1/2/4GPUworkers remain separate conditional tests.

"""Declarative, bounded campaign expansion; planning never submits jobs."""
from pathlib import Path
import json
from .artifacts import sha256_file
from .banks import CONTEXT_SEEDS, CONTEXT_SIZES, HELDOUT_SEED


def campaign_plan(config, *, banks, references, runs):
    value = json.loads(Path(config).read_text())
    required = (
        (value.get("schema"), "posteriorforge.campaign.v1"),
        (value["context"]["seeds"], list(CONTEXT_SEEDS)),
        (value["context"]["complete_bank_size"], 4096),
        (value["context"]["nested_prefixes"], list(CONTEXT_SIZES)),
        (value["context"]["prior"], "full_joint"),
        (value["heldout"]["seed"], HELDOUT_SEED),
        (value["heldout"]["count"], 8),
        (value["posterior"]["complete_accepted_count"], 50000),
        (value["posterior"]["upstream_chunk_size"], 10000),
        (value["posterior"]["clip_samples"], False),
        (value["budget"]["hard_allocated_gpu_hours"], 200),
    )
    if any(actual != expected for actual, expected in required):
        raise ValueError("configuration changes the accepted scientific campaign")
    reference_jobs = [{"case": case, "banks": str(banks), "output": str(Path(references)/f"case-{case}")}
                      for case in range(8)]
    inference_jobs = [{"context_seed": seed, "case": case, "context_size": size,
                      "count": 50000, "microbatch": 10000, "banks": str(banks),
                      "output": str(Path(runs)/f"context-{size}"/f"seed-{seed}-case-{case}")}
                     for size in CONTEXT_SIZES for seed in CONTEXT_SEEDS for case in range(8)]
    return {"schema": "posteriorforge.campaign_plan.v1", "config_sha256": sha256_file(config),
            "submits_jobs": False, "references": reference_jobs,
            "first_gate_runs": [run for run in inference_jobs if run["context_size"] == 4096],
            "after_gate_baseline_runs": [run for run in inference_jobs if run["context_size"] != 4096],
            "optimization_requires_all_24_first_gate_runs_pass": True}

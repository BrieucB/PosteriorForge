"""Complete-matrix scientific gate with authenticated bank/reference binding."""
from pathlib import Path
import json
import numpy as np
from .artifacts import sha256_file, write_json
from .banks import CONTEXT_SEEDS, load_heldout
from .inference import support_mask
from .science.analytical_benchmark_v2 import array_descriptor, build_problem
from .reference import load_reference_distribution
from .validation import assess_samples, make_reference_input


def evaluate_gate(*, banks, references, runs, output):
    result = {"schema": "posteriorforge.scientific_gate.v1", "status": "incomplete",
              "passed": False, "expected_runs": 24, "context_size": 4096,
              "expected_count": 50000, "runs": [], "references": [],
              "scope": "eight_case_diagnostic_screen_not_global_calibration_or_SBC",
              "optimization_authorized_by_gate": False}
    refs = {}
    for case in range(8):
        root = Path(references) / f"case-{case}"
        path = root / "manifest.json"
        if not path.exists():
            result["references"].append({"case": case, "state": "missing"})
            continue
        manifest = json.loads(path.read_text())
        result["references"].append({"case": case, "state": manifest["state"],
                                     "cdf_precision": manifest.get("cdf_precision"),
                                     "manifest_sha256": sha256_file(path)})
        if "distribution" not in manifest:
            continue
        arrays, manifest = load_reference_distribution(root)
        query = load_heldout(banks, case).obs
        binding = manifest["binding"]
        if (binding["scientific_fingerprint"] != build_problem().scientific_fingerprint
                or binding["observation_sha256"] != array_descriptor(query)["sha256"]
                or binding["parameter_order"] != list(build_problem().parameter_order)):
            raise ValueError("reference case/science binding mismatch")
        refs[case] = make_reference_input(arrays, manifest)
    discovered = {}
    for path in sorted(Path(runs).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "posteriorforge.inference.v1": continue
        if manifest.get("context_size") != 4096 or manifest.get("requested_count") != 50000: continue
        if manifest.get("variant") != "upstream_unmodified": continue
        key = (manifest["context_seed"], manifest["case"])
        if key in discovered: raise ValueError("ambiguous duplicate scientific run; select an explicit run root")
        discovered[key] = (path, manifest)
    for seed in CONTEXT_SEEDS:
        for case in range(8):
            entry = {"context_seed": seed, "case": case, "status": "missing", "passed": False}
            pair = discovered.get((seed, case))
            if pair:
                path, manifest = pair
                entry.update(manifest_sha256=sha256_file(path), manifest_file=str(path.relative_to(runs)))
                if manifest["status"] != "complete":
                    entry["status"] = "execution_failure"
                elif case not in refs:
                    entry["status"] = "reference_inconclusive"
                else:
                    query = load_heldout(banks, case).obs
                    if manifest["query"] != array_descriptor(query):
                        raise ValueError("inference query differs from held-out observation")
                    payload = path.parent / "samples.npz"
                    if sha256_file(payload) != manifest["samples_sha256"]:
                        raise ValueError("inference sample payload hash mismatch")
                    with np.load(payload, allow_pickle=False) as data:
                        values = data["samples"].copy()
                    if array_descriptor(values) != manifest["samples"]:
                        raise ValueError("inference sample array hash mismatch")
                    entry["assessment"] = assess_samples(values, refs[case], support_mask=support_mask(values))
                    entry.update(status=entry["assessment"]["status"], passed=entry["assessment"]["passed"])
                    entry["timings"] = manifest["timings"]
                    entry["memory"] = manifest["memory"]
            result["runs"].append(entry)
    statuses = [entry["status"] for entry in result["runs"]]
    if all(entry["passed"] for entry in result["runs"]):
        result.update(status="pass", passed=True, optimization_authorized_by_gate=True)
    elif "missing" in statuses:
        result["status"] = "incomplete"
    elif any(status in ("model_failure", "invalid_samples") for status in statuses):
        result["status"] = "model_failure"
    elif any(status == "execution_failure" for status in statuses):
        result["status"] = "execution_failure"
    else:
        result["status"] = "reference_or_cdf_inconclusive"
    write_json(output, result)
    return result

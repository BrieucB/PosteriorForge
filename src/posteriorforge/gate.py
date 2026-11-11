"""Complete-matrix scientific gate with authenticated bank/reference binding."""
from pathlib import Path
import json
import numpy as np
from .artifacts import sha256_file, write_json
from .banks import CONTEXT_SEEDS, load_heldout, load_context, derive_seed
from .inference import support_mask, frozen_regressor_settings, upstream_source_hashes, UPSTREAM_REVISION, CHECKPOINT_SHA256
from .science.analytical_benchmark_v2 import array_descriptor, build_problem, unflatten_observations
from .reference import load_reference_distribution
from .science.analytical_benchmark_reference_v2 import FullHierarchyReference, refinement_witnesses
from .validation import assess_samples, make_reference_input


def verify_run_binding(manifest, context, query, seed, case):
    expected = {"scientific_fingerprint": build_problem().scientific_fingerprint,
                "parameter_order": list(build_problem().parameter_order),
                "context_theta": array_descriptor(context.theta),
                "context_obs": array_descriptor(context.obs),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "upstream_revision": UPSTREAM_REVISION,
                "upstream_files": upstream_source_hashes(),
                "context_filter": "no_filtering", "max_iter_rejection": None,
                "scientific_run": True, "microbatch": 10000,
                "seed": derive_seed(seed, "posterior", case=case),
                "repetition": 0, "rank": 0, "world_size": 1}
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"inference contract mismatch: {field}")
    actual_settings = dict(manifest["regressor_settings"])
    actual_settings.pop("model_path")
    expected_settings = frozen_regressor_settings("unused.ckpt")
    expected_settings.pop("model_path")
    if actual_settings != expected_settings:
        raise ValueError("frozen regressor settings mismatch")
    if manifest["query"] != array_descriptor(query):
        raise ValueError("inference query differs from held-out observation")



def verify_refined_grid(arrays, manifest, parent_arrays, parent_manifest, parent_sha):
    if manifest.get("parent_manifest_sha256") != parent_sha or manifest.get("binding") != parent_manifest["binding"]:
        raise ValueError("refined reference parent binding mismatch")
    if any(not np.array_equal(arrays[name], parent_arrays[name]) for name in ("directions", "mean", "sd")):
        raise ValueError("grid refinement changed directions or physical moments")
    if manifest.get("moment_precision") != parent_manifest["moment_precision"]:
        raise ValueError("grid refinement changed moment qualification")
    contract = manifest["grid_contract"]
    if (contract["schema"] != "posteriorforge.reference_grid_contract.v1"
            or contract["reference_only_selection"] is not True
            or contract["maximum_allowed_cdf_step"] != .005
            or contract["directions_sha256"] != array_descriptor(arrays["directions"])["sha256"]):
        raise ValueError("refined grid numerical contract mismatch")
    current = np.linspace(-8., 8., 513)
    final_queries = arrays["queries"]
    for entry in manifest["grid_history"]:
        if array_descriptor(current)["sha256"] != entry["queries_sha256"]:
            raise ValueError("grid query history hash mismatch")
        positions = np.searchsorted(final_queries, current)
        if np.any(positions >= len(final_queries)) or not np.array_equal(final_queries[positions], current):
            raise ValueError("refinement discarded retained queries")
        cdfs = arrays["cdfs"][:, positions]
        selected = np.flatnonzero(np.diff(cdfs, axis=1).max(axis=0) > .005)
        if selected.tolist() != entry["selected_interval_indices"]:
            raise ValueError("grid selection was not fixed-reference deterministic")
        additions = np.asarray(entry["added_queries"])
        expected = current[selected] + np.diff(current)[selected] / 2
        if len(additions):
            if not np.array_equal(additions, expected):
                raise ValueError("grid additions were not deterministic midpoints")
            current = np.sort(np.r_[current, additions])
    if not np.array_equal(current, final_queries):
        raise ValueError("grid final queries differ from retained history")
    if manifest.get("state") == "verified":
        if (manifest["qualification"].get("grid_gate_passed") is not True
                or np.diff(arrays["cdfs"], axis=1).max() > .005
                or manifest["cdf_precision"] > .005):
            raise ValueError("refined grid incorrectly claims qualification")


def evaluate_gate(*, banks, references, runs, output, reference_grids=None):
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
        definition = FullHierarchyReference(build_problem(), unflatten_observations(query))
        expected_directions, _ = refinement_witnesses(definition)
        if (not np.array_equal(arrays["directions"], expected_directions)
                or not np.array_equal(arrays["queries"], np.linspace(-8., 8., 513))):
            raise ValueError("fixed joint-projection or dense query definition mismatch")
        if reference_grids is not None:
            grid_root = Path(reference_grids) / f"case-{case}"
            grid_path = grid_root / "manifest.json"
            if not grid_path.exists():
                result["references"][-1]["grid_state"] = "missing"
                continue
            grid_arrays, grid_manifest = load_reference_distribution(grid_root)
            verify_refined_grid(grid_arrays, grid_manifest, arrays, manifest, sha256_file(path))
            result["references"][-1].update(grid_state=grid_manifest["state"],
                grid_manifest_sha256=sha256_file(grid_path),
                grid_point_count=len(grid_arrays["queries"]),
                maximum_cdf_step=grid_manifest["grid_contract"]["maximum_cdf_step"],
                cdf_precision=grid_manifest["cdf_precision"])
            arrays, manifest = grid_arrays, grid_manifest
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
                    context = load_context(banks, seed, 4096)
                    verify_run_binding(manifest, context, query, seed, case)
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
        result["status"] = "numerical_inconclusive"
    write_json(output, result)
    return result

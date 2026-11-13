"""Deterministic read-only saved-array diagnosis; no recovered physical cohort.

No random API/state capture, sampler, model, GPU, simulation or reference call.
Frozen per-noise codec helpers inspect unchanged contiguous coordinates; all rows
remain represented. No full generated 21D physical matrix is assembled or saved.
"""
from pathlib import Path
import argparse, hashlib, inspect, json, platform, sys
import numpy as np
import scipy
from posteriorforge import prior_coordinates_v1 as codec
from posteriorforge import prior_coordinate_pair_v1 as pairs
from posteriorforge.banks import load_context
from posteriorforge.science.analytical_benchmark_v2 import array_descriptor

MANIFEST_SHA = "92755b788fc894d4ca16157e573bd1b5c9d8a4468b71cf184fa16303bced7bce"
LATENT_SHA = "1ba89c1a4e53c5895ad1623b5a396e4086154c3dc830f8036da388f234d90349"
RETURNS_SHA = "6989b4930fcb8dda2a144811c7988530dec120b8e6c1bfd62b269f47261a14af"
CODEC_SHA = "d8ce6c49e7b65c6ee55c59b7243d4ce943cbea28c92ebed0329798074aaa70c5"
PAIR_SHA = "402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
QUANTILES = (0., .001, .01, .05, .5, .95, .99, .999, 1.)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(values):
    assert np.isfinite(values).all()
    q = np.quantile(values, QUANTILES, method="linear")
    return {"count": len(values), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
            "quantile_method": "linear", "quantiles": {str(a): float(b) for a, b in zip(QUANTILES, q)}}

def source_span(function):
    lines, start = inspect.getsourcelines(function)
    return {"file": str(Path(inspect.getsourcefile(function)).resolve()),
            "first_line": start, "last_line": start + len(lines) - 1}

def noise_coordinate_diagnosis(z, physical_slot, geo):
    """Exact d8 noise+roundtrip checks on every unchanged coordinate, no cohort."""
    spec = geo["noise"][physical_slot]
    status = np.zeros(len(z), dtype=np.uint8)
    failures = []
    calls = {"contiguous_blocks": 0, "noise_decode": 0, "noise_encode": 0,
             "latent_check": 0, "physical_check": 0}
    stage = codec.INVERSE_ORDER[physical_slot]
    def decode(v):
        calls["noise_decode"] += 1
        return codec._noise_decode(v, spec)
    def block(low, high):
        calls["contiguous_blocks"] += 1
        # Exact widening performed by frozen decode_ordered_prefix line233.
        # Values and row ordering are unchanged; no synthetic replacement input.
        v = np.ascontiguousarray(z[low:high], dtype=np.float64)
        try:
            physical = decode(v)
            calls["noise_encode"] += 1
            replay = codec._noise_encode(physical, spec)
            calls["latent_check"] += 1
            codec._latent_check(replay, v, codec.F32_LATENT_ADMISSIBILITY)
            again = decode(replay)
            calls["physical_check"] += 1
            codec._physical_check(again[:, None], physical[:, None], [physical_slot],
                                  codec.F64_PHYSICAL_UNITS, geo)
        except codec.CoordinateError as error:
            if high - low > 1:
                middle = low + (high-low)//2
                block(low, middle)
                block(middle, high)
                return
            status[low] = 2
            trace, attempted = [], None
            tb = error.__traceback__
            while tb is not None:
                frame = tb.tb_frame
                if Path(frame.f_code.co_filename).resolve() == Path(codec.__file__).resolve():
                    trace.append({"function": frame.f_code.co_name, "line": tb.tb_lineno})
                    if frame.f_code is codec._noise_decode.__code__ and "output" in frame.f_locals:
                        output = frame.f_locals["output"]
                        if output.shape == (1,):
                            attempted = float(output[0])
                tb = tb.tb_next
            failures.append({"row_index_zero_based": low, "ordered_stage": stage,
                "physical_slot": physical_slot, "physical_parameter": codec.PHYSICAL_ORDER[physical_slot],
                "latent_name": "z_"+codec.PHYSICAL_ORDER[physical_slot],
                "actual_model_Float32_latent": float(z[low]),
                "actual_model_scalar_bytes_hex": z[low:low+1].tobytes().hex(),
                "exception_type": type(error).__name__, "message": str(error),
                "frozen_codec_trace": trace, "frozen_scalar_output_before_rejection": attempted,
                "row_partition": "first50000" if low < 50000 else "unused1200tail"})
            return
        status[low:high] = 1
    block(0, len(z))
    assert np.all(status != 0)
    return {"stage": stage, "physical_slot": physical_slot, "name": codec.PHYSICAL_ORDER[physical_slot],
            "count": len(z), "passed_coordinates": int(np.count_nonzero(status == 1)),
            "failed_coordinates": int(np.count_nonzero(status == 2)),
            "all_rows_examined": True, "diagnostic_helper_calls": calls,
            "coordinate_status_sha256": hashlib.sha256(status.tobytes()).hexdigest(),
            "failures": failures}

def metadata_border_inventory(value, path=""):
    result = []
    if isinstance(value, dict):
        if "name" in value and "border" in str(value["name"]).lower():
            result.append({"path": path, "name": value["name"],
                "keys": sorted(value), "actual_values_present": isinstance(value.get("values"), list)})
        for key, child in value.items():
            here = path+"."+key
            if "border" in key.lower():
                if isinstance(child, str):
                    result.append({"path": here, "value": child})
                elif isinstance(child, dict):
                    result.append({"path": here, "keys": sorted(child),
                                   "actual_values_present": isinstance(child.get("values"), list)})
                elif isinstance(child, (float, int, bool)):
                    result.append({"path": here, "value": child})
                else:
                    result.append({"path": here, "type": type(child).__name__})
            result.extend(metadata_border_inventory(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(metadata_border_inventory(child, path+"["+str(index)+"]"))
    return result

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--repo", type=Path, required=True)
    cli.add_argument("--scratch-root", type=Path, required=True)
    args = cli.parse_args()
    repo, root = args.repo.resolve(), args.scratch_root.resolve()
    phase = root/"results/accuracy-prior-coordinates-v1"
    folder = phase/"native-pilot-v2-seed20261111-case0-01"
    manifest, latent_file, returns_file = (folder/"manifest.json",
        folder/"latent-proposal-waves.npz", folder/"latent-stage-returns.npz")
    assert sha(manifest) == MANIFEST_SHA and sha(latent_file) == LATENT_SHA and sha(returns_file) == RETURNS_SHA
    raw_manifest = manifest.read_bytes()
    d = json.loads(raw_manifest)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    assert versions == d["failure_forensics"]["runtime_versions"]
    assert codec.source_sha256() == CODEC_SHA and pairs.source_sha256() == PAIR_SHA
    codec.require_sources();pairs.require_sources()
    sources = {str(Path(codec.__file__).resolve()): CODEC_SHA,
        str(Path(pairs.__file__).resolve()): PAIR_SHA,
        str(repo/"src/posteriorforge/performance_runner.py"): "23ccad8b5bbcdd7a48340b349d515ca870763359cfeb71333e02fc2b6a8aca3b",
        str(repo/"src/posteriorforge/execution_ss15_sampling.py"): "4037d48846ef3678ce7b647c1a326606ea967d396b3ce34bb1998241a131f6de"}
    assert all(sha(path) == digest for path, digest in sources.items())
    with np.load(latent_file, allow_pickle=False) as saved:
        assert saved.files == ["wave_0000"]
        latent = saved["wave_0000"]
    assert latent.dtype == np.float32 and latent.shape == (51200,21)
    latent.setflags(write=False)
    descriptor = array_descriptor(latent)
    assert descriptor == d["failed_decode_wave_descriptor"] == d["failure_forensics"]["input"]
    correspondence = []
    with np.load(returns_file, allow_pickle=False) as saved:
        expected_keys = ["stage_"+str(j).zfill(4) for j in range(21)] + ["filled_stage_mask", "ordered_latent_prefix"]
        assert set(saved.files) == set(expected_keys)
        filled, prefix = saved["filled_stage_mask"], saved["ordered_latent_prefix"]
        assert filled.dtype == np.bool_ and filled.shape == (21,) and filled.all()
        assert prefix.dtype == latent.dtype and prefix.shape == latent.shape and prefix.tobytes() == latent.tobytes()
        for stage in range(21):
            actual = saved["stage_"+str(stage).zfill(4)]
            assert actual.dtype == np.float32 and actual.shape == (51200,)
            assert actual.tobytes() == latent[:,stage].tobytes()
            record = d["stage_draw_observation"]["actual_stage_return_events"][stage]
            assert array_descriptor(actual) == record["actual_returned_draws"]
            correspondence.append({"stage": stage, "physical_slot": codec.ORDER[stage],
                "dtype": actual.dtype.str, "shape": list(actual.shape),
                "stage_return_sha256": array_descriptor(actual)["sha256"],
                "returned_bytes_equal_original_assembly_column": True,
                "additional_cast_performed": False})
    assert np.isfinite(latent).all()
    # Existing ORIGINAL physical input arrays only; no simulation or model calls.
    owner = pairs.PairSpec(20261111,0)
    context = load_context(root/"results/banks-v1",20261111,4096)
    assert array_descriptor(context.theta) == d["raw_scientific_bindings"]["context_theta"]
    original_theta_sha = array_descriptor(context.theta)["sha256"]
    fixture, custody = pairs.load_prefix(owner,
        completion_receipt=root/"results/accuracy-ss15-v1/sigma-native-grid-completion-01/completion.json",
        parent=root/"results/accuracy-ss15-v1/sigma-native-pilot-01")
    assert custody == d["matched_execution_screen"]["fixed_prefix_custody"]
    fixture_sha = array_descriptor(fixture)["sha256"]
    proof = json.loads((phase/"pair-conditionals-v2-seed20261111-case0-01/manifest.json").read_bytes())
    assert sha(phase/"pair-conditionals-v2-seed20261111-case0-01/manifest.json") == d["matching_conditional_qualification_sha256"]
    geo = codec._geometry()
    analyses, distributions = [], []
    for physical_slot in codec.NOISE_INDICES:
        stage = codec.INVERSE_ORDER[physical_slot]
        spec = geo["noise"][physical_slot]
        generated = latent[:,stage]
        analysis = noise_coordinate_diagnosis(generated, physical_slot, geo)
        analyses.append(analysis)
        # Independent noise encoding is exactly the relevant codec channel;
        # this is the declared one-final-model-cast, not changed input values.
        training_model = codec._noise_encode(context.theta[:,physical_slot], spec).astype(np.float32)
        fixture_model = codec._noise_encode(fixture[:,physical_slot], spec).astype(np.float32)
        assert array_descriptor(training_model) == proof["stage_results"][stage]["mask_inputs"]["actual_latent_target"]
        distributions.append({"stage": stage, "physical_slot": physical_slot,
            "latent_name": "z_"+codec.PHYSICAL_ORDER[physical_slot],
            "generated_all51200": stats(generated), "generated_first50000": stats(generated[:50000]),
            "generated_unused1200tail": stats(generated[50000:]),
            "training_prior4096_model_Float32": stats(training_model),
            "training_noise_model_descriptor": array_descriptor(training_model),
            "training_descriptor_matches_actual_qualified_target": True,
            "original_pure_FIRST1024_fixture_model_Float32": stats(fixture_model),
            "fixture_noise_model_descriptor": array_descriptor(fixture_model)})
    failures = sorted([r for a in analyses for r in a["failures"]],
                      key=lambda x:(x["row_index_zero_based"],x["ordered_stage"]))
    byrow = {}
    for row in failures:
        byrow.setdefault(row["row_index_zero_based"],[]).append(row)
    recorded = d["failure_forensics"]["failures"]
    assert sorted(byrow) == sorted(r["row_index"] for r in recorded)
    comparisons = []
    for old in recorded:
        row_index = old["row_index"]
        first = min(byrow[row_index],key=lambda x:x["ordered_stage"])
        assert first["ordered_stage"] == old["first_failing_causal_stage"]
        assert first["physical_slot"] == old["physical_slot"]
        assert first["actual_model_Float32_latent"] == old["latent_value_at_stage"]
        assert first["actual_model_scalar_bytes_hex"] == old["latent_scalar_bytes_hex"]
        assert latent[row_index].tobytes().hex() == old["raw_latent_row_bytes_hex"]
        comparisons.append({"row_index": row_index, "recorded_first_fail_matches_actual_saved_scalar": True,
                            "recorded_entire_row_bytes_match_actual_NPZ": True,
                            "all_noise_failures_for_this_row": len(byrow[row_index])})
    partitions = {}
    for name,start,end in (("all51200",0,51200),("first50000",0,50000),("unused1200tail",50000,51200)):
        row_count = sum(start<=r<end for r in byrow)
        cell_count = sum(start<=r["row_index_zero_based"]<end for r in failures)
        partitions[name] = {"proposal_rows": end-start, "failing_rows": row_count,
            "row_failure_fraction": row_count/(end-start), "row_failure_percent":100*row_count/(end-start),
            "noise_scalar_coordinates":6*(end-start), "failing_noise_coordinates":cell_count,
            "noise_coordinate_failure_fraction":cell_count/(6*(end-start))}
    journal = d["observed_failure_footer"]["observations"]["actual_engine_journal"]["value"]
    borders = {str(j):metadata_border_inventory(journal["completed_stage_scopes"][j]["model"])
               for j in (15,19)}
    missing = [k for k in ("execution_counters","sampling_counters","actual_prediction_checks",
        "expected_patched_call_count","expected_stage_prepare_count","generated_proposal_wave_count",
        "default_rng_after","default_rng_preserved","sources_unchanged","rejection","timings","memory",
        "samples","samples_file","samples_sha256","decoded_wave_archive") if k not in d]
    assert array_descriptor(latent)==descriptor and array_descriptor(context.theta)["sha256"]==original_theta_sha
    assert array_descriptor(fixture)["sha256"]==fixture_sha
    assert manifest.read_bytes()==raw_manifest and sha(latent_file)==LATENT_SHA and sha(returns_file)==RETURNS_SHA
    assert all(sha(path)==digest for path,digest in sources.items()) and "torch" not in sys.modules
    report={"schema":"posteriorforge.readonly_noise_latent_array_diagnosis.v1",
        "status":"diagnosis_complete_request_remains_failed","manifest_sha256":MANIFEST_SHA,
        "latent_archive":{"sha256":LATENT_SHA,"key":"wave_0000","role":"full assembled ordered latent proposal wave; not physical samples","descriptor":descriptor},
        "stage_return_archive":{"sha256":RETURNS_SHA,"role":"actual21 returned Float32 vectors, filled mask and duplicate ordered latent prefix","filled_mask_dtype":"bool","prefix_dtype":"Float32"},
        "original_assembly_convention":{"source_sha256":sources[str(repo/"src/posteriorforge/performance_runner.py")],
            "source_path":"src/posteriorforge/performance_runner.py","lines":[470,472],
            "convention":"sample=np.asarray(sample,dtype=np.float32); values[:,stage]=sample",
            "all_actual_returns_already_Float32_no_extra_cast_for_check":True},
        "stage_return_correspondence":correspondence,"sources_before_and_after":sources,"runtime_versions":versions,
        "diagnostic_code_sha256":sha(__file__),"noise_coordinate_analyses":analyses,
        "all_offending_noise_cells":failures,"unique_offending_rows":sorted(byrow),"partitions":partitions,
        "existing_first_failure_record_checks":comparisons,
        "noise_distributions":distributions,"ordinary_quantile_grid":list(QUANTILES),
        "distribution_scope":"Generated values are conditional on heldoutcase0 and generated prefixes; training4096 are unconditional prior values; FIRST1024 is old pure-model prefix fixture, not a new cohort or qualified reference posterior. Range/quantile differences alone do not establish a cause or calibration failure.",
        "fixture_custody":custody,"training_physical_theta_descriptor":array_descriptor(context.theta),
        "diagnostic_semantics":{"all_six_noise_slots_and_all51200_rows_accounted":True,
            "exact_frozen_noise_decode_encode_and_declared_F32_latent_F64_physical_checks":True,
            "recursive_subdivision_only_for_failure_localization_not_row_filtering":True,
            "all_generated_21D_physical_matrix_assembled_or_saved":False,
            "unchanged_input_values_widened_exactly_as_codec_line233":True,
            "modified_clipped_repaired_filtered_resampled_inputs":False,
            "RNG_state_capture_or_sampling_model_world_reference_G3_calls":0},
        "exact_exception_scope":{"require_raise_line":43,"strict_noise_condition_lines":[109,111],
            "noise_decode_calls_strict_line":137,"causal_decode_calls_noise_line":176,
            "full_prefix_decode_line":234,"latent_roundtrip_check_line":237,
            "codec_path":str(Path(codec.__file__).resolve()),"codec_sha256":CODEC_SHA},
        "saved_model_border_metadata":borders,
        "actual_selected_head_uniform_logits_bin_CDF_fraction_available":False,
        "border_cause_scope":"Saved summaries report border dtypes, not the actual prediction-specific canonical border arrays/minmax; no actual-border overshoot or ICDF-cause attribution is made.",
        "normal_success_fields_missing":missing,
        "available_failure_guard_scope":{"source":"existing observed_failure_footer, not new model/RNG capture",
            "all21_returned":len(d["stage_events"])==21,"all_stage_aliases_restored":journal["all_stage_aliases_restored"],
            "CDF_calls":d["compatibility_adapter"]["patched_call_count"],
            "CDF_alias_restored":d["compatibility_adapter"]["original_alias_restored"],
            "source_closure_observed_passed":d["observed_failure_footer"]["observations"]["actual_source_closure_after_exception"]["value"]["passed"],
            "postexception_default_RNG_observed_equal":d["observed_failure_footer"]["observations"]["actual_default_RNG_post_exception"]["value"]["equal_when_available"],
            "original_support_counters":d["observed_failure_footer"]["observations"]["actual_support_counters"]["value"],
            "FIFO_delivered":d["proposal_buffer"]["original_api_delivered_count"],"FIFO_finalized":d["proposal_buffer"]["finalized"]},
        "no_recovered_cohort_or_accuracy_assessment":True,"attribution":"Built with PriorLabs-TabPFN"}
    destination=Path(__file__).with_name("diagnostic.json")
    destination.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"diagnostic_sha256":sha(destination),"offenders":failures,
        "partitions":partitions,"all_noise_cells":sum(a["count"] for a in analyses),
        "helper_calls":sum(a["diagnostic_helper_calls"]["contiguous_blocks"] for a in analyses),
        "Torch_imported":False},indent=2))

if __name__=="__main__":
    main()

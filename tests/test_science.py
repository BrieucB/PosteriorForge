"""CPU math/provenance tests: no model downloads and no cluster required."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from posteriorforge.science import analytical_benchmark_v2 as B
from posteriorforge.science import analytical_benchmark_reference_v2 as R

FINGERPRINT = "216a240a9fb347348f433804b78eeebb6e040332db6d2fd40b164a7ed74b7966"


def test_science_fingerprint_and_shapes():
    problem = B.build_problem()
    assert problem.scientific_fingerprint == FINGERPRINT
    assert len(problem.parameter_order) == 21
    assert sum(dataset.observation_size for dataset in problem.datasets) == 33
    assert problem.parameter_order[6] == "sigma_m1"
    assert problem.parameter_order[-1] == "sigma_a3"
    assert problem.prior.bounds == (.01, .5)


def test_vendored_closure_hashes_match_immutable_source():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "provenance/science-source.json").read_text())
    assert len(provenance["files"]) == 6
    for entry in provenance["files"]:
        actual = sha256((root / entry["target"]).read_bytes()).hexdigest()
        assert actual == entry["target_sha256"] == entry["source_sha256"]


def test_observation_order_and_prior_draws():
    theta, observation = B.draw_worlds(19, role="posteriorforge.unit", seed=8)
    problem = B.build_problem()
    flat = B.flatten_observations(observation)
    assert theta.shape == (19, 21) and theta.dtype == np.float64
    assert flat.shape == (19, 33) and flat.dtype == np.float64
    assert problem.contains(theta).all()
    rebuilt = B.unflatten_observations(flat)
    for name in B.DATASET_IDS:
        np.testing.assert_array_equal(observation.values[name], rebuilt.values[name])
    np.testing.assert_array_equal(flat[:, :10], observation.values["mechanical_1"])
    np.testing.assert_array_equal(flat[:, 30:31], observation.values["acoustic_1"])


def test_frozen_stream_random_access_and_central_selection():
    full, obs = B.draw_worlds(16, role="posteriorforge.stream", seed=93, start=503)
    chunks = [B.draw_worlds(8, role="posteriorforge.stream", seed=93, start=i) for i in (503, 511)]
    np.testing.assert_array_equal(full, np.concatenate([p[0] for p in chunks]))
    np.testing.assert_array_equal(B.flatten_observations(obs),
                                 np.concatenate([B.flatten_observations(p[1]) for p in chunks]))
    central, _ = B.draw_worlds(3, role="posteriorforge.heldout", seed=20261114, central=True)
    assert B.central_world_mask(central).all()


def test_small_conditional_math_matches_independent_observation_space():
    # Five tiny Gaussian algebra checks are unit work; no Sobol reference is run.
    problem = B.build_problem()
    _, obs = B.draw_worlds(1, role="posteriorforge.reference-unit", seed=11)
    reference = R.FullHierarchyReference(problem, obs)
    assert reference.verify_cpu()["passed"]
    assert reference.residual_dof == [7, 7, 7, 0, 0, 0]
    assert len(reference.names) == 15
    directions, queries = R.refinement_witnesses(reference)
    assert directions.shape == (53, 21) and queries.shape == (129,)
    np.testing.assert_array_equal(directions[21:] != 0, np.ones((32, 21), dtype=bool))

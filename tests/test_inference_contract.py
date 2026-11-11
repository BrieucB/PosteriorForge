import numpy as np
import pytest
from posteriorforge.inference import validate_query, support_mask, frozen_regressor_settings
from posteriorforge.science.analytical_benchmark_v2 import build_problem


def test_query_rejects_labels_and_wrong_observation_shape():
    obs = np.arange(33, dtype=float)
    accepted = validate_query(obs)
    assert accepted.shape == (1, 33)
    obs[:] = -1
    assert accepted[0, 32] == 32
    for candidate in (np.zeros(21), np.zeros(54), np.zeros((8, 33)), np.full(33, np.nan)):
        with pytest.raises(ValueError):
            validate_query(candidate)


def test_complete_row_support_checks_every_noise_coordinate():
    problem = build_problem()
    theta = problem.sample_prior(5, np.random.default_rng(123))
    assert support_mask(theta).all()
    for name in problem.prior.noise_names:
        broken = theta.copy()
        broken[0, problem.parameter_order.index(name)] = .5001
        assert not support_mask(broken)[0]
    theta[1, 0] = np.inf
    assert not support_mask(theta)[1]
    with pytest.raises(ValueError):
        support_mask(theta[:, :-1])


def test_frozen_configuration():
    settings = frozen_regressor_settings('test.ckpt')
    assert settings['inference_precision'] == 'torch.float32'
    assert settings['n_estimators'] == 8
    assert settings['fit_mode'] == 'fit_preprocessors'
    assert settings['random_state'] == 0

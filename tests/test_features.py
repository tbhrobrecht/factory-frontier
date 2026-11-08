"""Feature and action-mask checks for student function approximators."""

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.wrappers import encode_features, mask_action_values, masked_policy_probabilities


@pytest.mark.parametrize("preset", ["function_approximation", "ppo"])
def test_features_encode_categories_and_preserve_visibility(preset):
    env = FactoryFrontier(preset)
    observation, _ = env.reset(seed=0)
    features = encode_features(observation)
    assert features.dtype == np.float32 and features.shape == (4165,)
    assert np.isfinite(features).all()
    channels = features[:16 * 16 * 13].reshape(16, 16, 13)
    visible = observation["visibility"].astype(bool)
    assert not channels[~visible].any()
    np.testing.assert_array_equal(channels[visible].argmax(axis=1), observation["grid"][visible])
    np.testing.assert_array_equal(channels.sum(axis=2), observation["visibility"])
    original = features.copy()
    observation["factory"][3] += 0.01
    assert not np.array_equal(encode_features(observation), original)
    np.testing.assert_array_equal(features, original)


def test_value_mask_excludes_illegal_outputs_without_mutating_them():
    values, mask = np.arange(15, dtype=np.float32), np.zeros(15, dtype=np.int8)
    mask[Action.NORTH] = 1
    masked = mask_action_values(values, mask)
    assert masked.argmax() == Action.NORTH and np.isneginf(masked[mask == 0]).all()
    np.testing.assert_array_equal(values, np.arange(15))
    assert np.isneginf(mask_action_values(values, np.zeros_like(mask))).all()
    with pytest.raises(ValueError, match="matching"):
        mask_action_values(values[:2], mask)


@pytest.mark.parametrize("preset,slip,deadline", [
    ("function_approximation", 0.0, 60), ("policy_methods", 0.08, 72),
    ("ppo", 0.05, 112),
])
def test_approximation_presets_expose_jumps_without_unfinished_builds(preset, slip, deadline):
    env = FactoryFrontier(preset)
    observation, _ = env.reset(seed=0)
    assert env.config.slip_probability == slip
    assert env.config.max_steps == env.config.width * env.config.boundary_interval == deadline
    assert not observation["action_mask"][Action.BUILD_SCOUT:Action.REMOVE_WALL + 1].any()
    assert observation["action_mask"][Action.JUMP_EAST]
    env.step(Action.JUMP_EAST)
    assert env.state.factory(0).position == (env.config.height - 1, 2)
    assert env.state.factory(0).energy == env.config.initial_energy - 2 + env.state.gathered[0]


def test_masked_policy_is_stable_and_excludes_illegal_actions():
    logits, mask = np.array([10000.0, 10001.0, 1e6]), np.array([1, 1, 0])
    probabilities = masked_policy_probabilities(logits, mask)
    np.testing.assert_allclose(probabilities, [1 / (1 + np.e), np.e / (1 + np.e), 0])
    assert probabilities.sum() == pytest.approx(1)
    assert masked_policy_probabilities(logits, mask, temperature=2).max() < probabilities.max()
    np.testing.assert_array_equal(logits, [10000, 10001, 1e6])
    with pytest.raises(ValueError, match="legal actions"):
        masked_policy_probabilities(logits, np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        masked_policy_probabilities(logits, mask, temperature=0)

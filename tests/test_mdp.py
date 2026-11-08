"""Focused regressions for the shared MDP and DP transition model."""

from copy import deepcopy

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier


@pytest.mark.parametrize("preset,slip", [("mdp", 0.2), ("dynamic_programming", 0.1)])
def test_model_probabilities_and_illegal_actions(preset, slip):
    env = FactoryFrontier(preset)
    env.reset(seed=0)
    model = env.transition_model()
    assert env.config.slip_probability == slip
    assert len(model) == 25
    for actions in model.values():
        assert set(actions) == {int(action) for action in Action}
        for outcomes in actions.values():
            assert all(p > 0 for p, *_ in outcomes)
            assert sum(p for p, *_ in outcomes) == pytest.approx(1.0)
    north = {position: p for p, position, _, _ in model[(4, 0)][int(Action.NORTH)]}
    assert north == pytest.approx({(3, 0): 1 - slip, (4, 1): slip / 3, (4, 0): 2 * slip / 3})
    for action in (Action.SOUTH, Action.WEST, Action.BUILD_SCOUT, Action.JUMP_NORTH):
        assert model[(4, 0)][int(action)] == [(1.0, (4, 0), -0.01, False)]
        env.step(action)
        assert env.state.factory(0).position == (4, 0)
    for outcomes in model[(0, 4)].values():
        assert outcomes == [(1.0, (0, 4), 0.0, True)]


@pytest.mark.parametrize("preset", ["mdp", "dynamic_programming"])
def test_model_preserves_episode_and_goal_reward(preset):
    env = FactoryFrontier(preset, reward_weights={"time": -0.5, "win": 7.0})
    env.reset(seed=0)
    model = env.transition_model()
    env.step(Action.NORTH)
    before = deepcopy(env.state)
    rng = deepcopy(env.np_random.bit_generator.state)
    assert env.transition_model() == model
    assert env.state == before
    assert env.np_random.bit_generator.state == rng
    env.state.factory(0).position = (0, 3)
    env.np_random = np.random.default_rng(0)  # First draw does not slip.
    _, reward, terminated, truncated, _ = env.step(Action.EAST)
    assert reward == 6.5 and terminated and not truncated
    assert any(destination == (0, 4) and reward == 6.5 and terminal
               for _, destination, reward, terminal in model[(0, 3)][int(Action.EAST)])
    assert env.transition_model() == model
    with pytest.raises(RuntimeError, match="finished"):
        env.step(Action.WAIT)


@pytest.mark.parametrize("preset", ["mdp", "dynamic_programming"])
def test_time_limit_is_truncation_not_model_termination(preset):
    env = FactoryFrontier(preset)
    env.reset(seed=0)
    assert env.config.max_steps == 50
    assert env.transition_model()[(4, 0)][int(Action.WAIT)] == [(1.0, (4, 0), -0.01, False)]
    for step in range(50):
        _, _, terminated, truncated, info = env.step(Action.WAIT)
        assert not terminated
        assert truncated == (step == 49)
        assert info["winner"] is None


@pytest.mark.parametrize("overrides", [
    {"danger_enabled": True}, {"resources_enabled": True},
    {"opponent_enabled": True}, {"movement_cost": 0.1},
    {"enabled_actions": (Action.WAIT, Action.MINE)},
])
def test_position_model_rejects_non_position_dynamics(overrides):
    env = FactoryFrontier("mdp", **overrides)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="movement-only"):
        env.transition_model()

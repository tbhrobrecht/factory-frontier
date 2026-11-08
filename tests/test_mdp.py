"""Check the planning model against actual environment transitions."""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.core.actions import MOVEMENT_ACTIONS


@pytest.mark.parametrize("preset", ["mdp", "dynamic_programming"])
@pytest.mark.parametrize("slip", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("enabled", [(Action.WAIT, *MOVEMENT_ACTIONS), (Action.WAIT, Action.NORTH)])
def test_model_matches_every_step_branch(monkeypatch, preset, slip, enabled):
    env = FactoryFrontier(preset, slip_probability=slip, enabled_actions=enabled)
    env.reset(seed=7)
    # Exercise blocked requested moves and slips into walls as well as edges.
    env.state.walls.add((2, 2))
    initial = env.state.copy()
    model = env.transition_model()
    assert (2, 2) not in model
    for position, actions in model.items():
        assert set(actions) == {int(action) for action in Action}
        for action in Action:
            outcomes = actions[int(action)]
            assert sum(p for p, *_ in outcomes) == pytest.approx(1.0)
            assert all(p > 0 for p, *_ in outcomes)
            if position == (0, 4):
                assert outcomes == [(1.0, position, 0.0, True)]
                continue
            actual = defaultdict(float)
            branches = (
                [(direction, 1 - slip if direction == action else slip / 3)
                 for direction in MOVEMENT_ACTIONS]
                if action in MOVEMENT_ACTIONS else [(action, 1.0)]
            )
            for direction, probability in branches:
                if probability == 0:
                    continue
                # Enumerate the sampler's possible results, retaining the real
                # engine's legality checks, movement, rewards and termination.
                monkeypatch.setattr(
                    "factory_frontier.core.rules.sample_movement",
                    lambda requested, probability, rng, direction=direction:
                        direction if requested in MOVEMENT_ACTIONS else requested,
                )
                env.state = initial.copy()
                env.state.factory(0).position = position
                env._episode_done = False
                observation, reward, terminated, truncated, info = env.step(action)
                assert not truncated
                assert env.observation_space.contains(observation)
                actual[(env.state.factory(0).position, reward, terminated)] += probability
            predicted = {(destination, reward, terminal): p for p, destination, reward, terminal in outcomes}
            assert predicted == pytest.approx(dict(actual))


def test_mdp_illegal_actions_wait_without_slipping():
    env = FactoryFrontier("mdp")
    env.reset(seed=0)
    actions = env.transition_model()[(4, 0)]
    for action in (Action.SOUTH, Action.WEST, Action.BUILD_SCOUT, Action.JUMP_NORTH):
        assert actions[int(action)] == [(1.0, (4, 0), -0.01, False)]
    north = {position: p for p, position, _, _ in actions[int(Action.NORTH)]}
    assert north == pytest.approx({(3, 0): 0.8, (4, 1): 0.2 / 3, (4, 0): 0.4 / 3})


def test_model_is_pure_and_independent_of_episode_progress():
    env = FactoryFrontier("mdp")
    env.reset(seed=10)
    model = env.transition_model()
    env.step(Action.NORTH)
    before = deepcopy(env.state)
    rng = deepcopy(env.np_random.bit_generator.state)
    assert env.transition_model() == model
    assert env.state == before
    assert env.np_random.bit_generator.state == rng
    # Model remains queryable after reaching the goal.
    env.state.factory(0).position = (0, 3)
    env.np_random = np.random.default_rng(0)  # First draw does not slip.
    env.step(Action.EAST)
    assert env.state.terminated
    assert env.transition_model() == model
    with pytest.raises(RuntimeError, match="finished"):
        env.step(Action.WAIT)


def test_mdp_time_limit_is_truncation_not_model_termination():
    env = FactoryFrontier("mdp", max_steps=1)
    env.reset(seed=0)
    assert env.transition_model()[(4, 0)][int(Action.WAIT)] == [(1.0, (4, 0), -0.01, False)]
    _, reward, terminated, truncated, info = env.step(Action.WAIT)
    assert reward == -0.01
    assert not terminated and truncated
    assert info["winner"] is None


def test_mdp_goal_reward_uses_configured_weights_once():
    env = FactoryFrontier("mdp", slip_probability=0.0, reward_weights={"time": -0.5, "win": 7.0})
    env.reset(seed=0)
    model = env.transition_model()
    assert model[(0, 3)][int(Action.EAST)] == [(1.0, (0, 4), 6.5, True)]
    assert model[(0, 4)][int(Action.WAIT)] == [(1.0, (0, 4), 0.0, True)]


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

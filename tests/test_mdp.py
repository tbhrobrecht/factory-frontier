"""Focused regressions for the shared MDP and DP transition model."""

from copy import deepcopy

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.core.actions import DELTAS, MOVEMENT_ACTIONS
from factory_frontier.core.rules import action_mask, sample_movement


@pytest.mark.parametrize("preset", ["mdp", "dynamic_programming"])
def test_discount_is_configurable_and_does_not_discount_rewards(preset):
    assert FactoryFrontier(preset).config.gamma == 0.99
    env = FactoryFrontier(preset, gamma=0.5)
    env.reset(seed=0)
    model = env.transition_model()
    reference = FactoryFrontier(preset)
    reference.reset(seed=0)
    assert model == reference.transition_model()
    assert env.config.with_overrides(gamma=0.0).gamma == 0.0
    for _ in range(3):
        assert env.step(Action.WAIT)[1] == -0.01
    env.state.factory(0).position = (0, 3)
    env.np_random = np.random.default_rng(0)  # The east move does not slip.
    _, reward, terminated, truncated, _ = env.step(Action.EAST)
    assert reward == pytest.approx(9.99)
    assert terminated and not truncated


@pytest.mark.parametrize("gamma", [-0.01, 1.0, 1.01, float("nan"), float("inf"), True, "0.99"])
def test_discount_validation(gamma):
    with pytest.raises(ValueError, match="gamma"):
        FactoryFrontier("mdp", gamma=gamma)


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


def test_wait_and_default_rewards_for_every_state_action():
    env = FactoryFrontier("mdp")
    env.reset(seed=0)
    goal = (0, 4)
    for position, actions in env.transition_model().items():
        assert actions[int(Action.WAIT)] == [
            (1.0, position, 0.0 if position == goal else -0.01, position == goal)
        ]
        for outcomes in actions.values():
            for _, destination, reward, terminal in outcomes:
                assert terminal == (destination == goal)
                expected = 0.0 if position == goal else 9.99 if terminal else -0.01
                assert reward == pytest.approx(expected)


@pytest.mark.parametrize("with_wall", [False, True])
def test_model_matches_every_simulator_branch(monkeypatch, with_wall):
    """Enumerate slip branches, including walls, boundaries and invalid requests."""
    env = FactoryFrontier("mdp")
    env.reset(seed=0)
    if with_wall:
        env.state.walls.add((2, 2))
    model = env.transition_model()
    template = env.state.copy()
    goal = (0, 4)
    for position, actions in model.items():
        if position == goal:
            continue  # The live API forbids steps after termination.
        for action in Action:
            source = template.copy()
            source.factory(0).position = position
            legal = bool(action_mask(source, env.config)[int(action)])
            branches = (
                [(direction, 0.8 if direction == action else 0.2 / 3)
                 for direction in MOVEMENT_ACTIONS]
                if legal and action in MOVEMENT_ACTIONS else [(Action.WAIT, 1.0)]
            )
            observed = {}
            for realized, probability in branches:
                env.state = source.copy()
                env._episode_done = False
                monkeypatch.setattr(
                    "factory_frontier.core.rules.sample_movement",
                    lambda requested, slip, rng, result=realized: result,
                )
                _, reward, terminal, truncated, _ = env.step(action)
                assert not truncated
                destination = env.state.factory(0).position
                # Independently verify the realized direction's spatial result.
                delta = DELTAS.get(realized, (0, 0))
                target = (position[0] + delta[0], position[1] + delta[1])
                expected = target if source.in_bounds(target) and target not in source.walls else position
                assert destination == expected
                key = (destination, reward, terminal)
                observed[key] = observed.get(key, 0.0) + probability
            predicted = {(destination, reward, terminal): p
                         for p, destination, reward, terminal in actions[int(action)]}
            assert observed == pytest.approx(predicted)


def test_simulator_samples_the_eighty_twenty_distribution():
    rng = np.random.default_rng(1234)
    samples = [sample_movement(Action.NORTH, 0.2, rng) for _ in range(30_000)]
    for direction in MOVEMENT_ACTIONS:
        expected = 0.8 if direction == Action.NORTH else 0.2 / 3
        assert samples.count(direction) / len(samples) == pytest.approx(expected, abs=0.01)
    before = deepcopy(rng.bit_generator.state)
    assert sample_movement(Action.WAIT, 0.2, rng) == Action.WAIT
    assert rng.bit_generator.state == before


def test_position_dynamics_do_not_depend_on_history_or_cutoff():
    early, late = FactoryFrontier("mdp"), FactoryFrontier("mdp")
    early.reset(seed=7)
    late.reset(seed=7)
    model = early.transition_model()
    for _ in range(49):
        late.step(Action.WAIT)
    # Same position/action and RNG draw, despite different episode histories.
    early.np_random = np.random.default_rng(11)
    late.np_random = np.random.default_rng(11)
    first = early.step(Action.NORTH)
    last = late.step(Action.NORTH)
    assert early.state.factory(0).position == late.state.factory(0).position
    assert first[1:3] == last[1:3]  # Reward and MDP termination are unchanged.
    assert not first[3] and last[3]  # Only the simulation cutoff differs.
    assert early.transition_model() == late.transition_model() == model
    with pytest.raises(RuntimeError, match="finished"):
        late.step(Action.WAIT)


def test_goal_entry_at_cutoff_terminates_once():
    env = FactoryFrontier("mdp")
    env.reset(seed=0)
    model = env.transition_model()
    for _ in range(49):
        env.step(Action.WAIT)
    env.state.factory(0).position = (0, 3)
    env.np_random = np.random.default_rng(0)
    observation, reward, terminated, truncated, info = env.step(Action.EAST)
    assert reward == pytest.approx(9.99)
    assert terminated and not truncated and info["step"] == 50
    assert not observation["action_mask"].any()
    assert env.transition_model() == model
    assert all(outcomes == [(1.0, (0, 4), 0.0, True)]
               for outcomes in model[(0, 4)].values())
    with pytest.raises(RuntimeError, match="finished"):
        env.step(Action.WAIT)

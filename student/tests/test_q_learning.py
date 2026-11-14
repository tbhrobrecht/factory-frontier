"""Regressions for fixed-map training, replay and checkpoint evaluation."""

from copy import deepcopy

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from student.agent import StudentAgent
from student.train import greedy_evaluation, train


@pytest.mark.parametrize("terminated,truncated,expected", [
    (False, False, 2.98), (False, True, 2.98), (True, False, 1.0),
])
def test_targets_mask_illegal_actions_and_bootstrap_only_without_termination(terminated, truncated, expected):
    env = FactoryFrontier("intro")
    observation, _ = env.reset(seed=0)
    next_observation, *_ = env.step(Action.NORTH)
    agent = StudentAgent(alpha=1.0, replay_episode=False)
    agent.reset(observation)
    next_values = agent.action_values(next_observation)
    next_values.fill(999.0)
    next_values[next_observation["action_mask"] != 0] = 2.0
    agent.update(observation, Action.NORTH, 1.0, next_observation, terminated, truncated)
    assert agent.action_values(observation)[Action.NORTH] == pytest.approx(expected)


def test_backward_replay_propagates_terminal_reward_without_double_counting_episodes():
    env = FactoryFrontier("intro")
    first, _ = env.reset(seed=0)
    second, *_ = env.step(Action.NORTH)
    final, *_ = env.step(Action.NORTH)
    agent = StudentAgent(alpha=1.0)
    agent.reset(first)
    agent.update(first, Action.NORTH, 0.0, second, False, False)
    assert agent.action_values(first)[Action.NORTH] == 0.0
    agent.update(second, Action.NORTH, 10.0, final, True, False)
    assert agent.action_values(first)[Action.NORTH] == pytest.approx(9.9)
    assert agent.action_values(second)[Action.NORTH] == 10.0
    assert agent.episodes == 1
    assert not agent.trajectory
    agent.reset(first)
    assert agent.action_values(first)[Action.NORTH] == pytest.approx(9.9)


def test_checkpoint_round_trip_and_greedy_evaluation_do_not_change_learning_state(tmp_path):
    env = FactoryFrontier("model_free_control")
    observation, _ = env.reset(seed=0)
    agent = StudentAgent(seed=7)
    agent.action_values(observation)[Action.EAST] = 3.0
    agent.episodes = 12
    path = tmp_path / "agent.json.gz"
    agent.save(path, config="model_free_control", map_seed=0)
    loaded = StudentAgent.load(path, seed=7)
    assert loaded.epsilon == 0.0
    assert loaded.episodes == 12
    assert loaded.checkpoint_metadata["map_seed"] == 0
    np.testing.assert_array_equal(loaded.action_values(observation), agent.action_values(observation))
    assert loaded.act(observation) == Action.EAST
    q_before = {key: values.copy() for key, values in agent.q.items()}
    rng_before = deepcopy(agent.rng.bit_generator.state)
    greedy_evaluation(agent, "model_free_control", (0, 0), seed=4)
    assert agent.epsilon == 0.1 and agent.episodes == 12
    assert agent.rng.bit_generator.state == rng_before
    assert set(agent.q) == set(q_before)
    for key, values in q_before.items():
        np.testing.assert_array_equal(agent.q[key], values)


@pytest.mark.parametrize("vary_maps,expected_seeds", [(False, [7, 7, 7]), (True, [7, 8, 9])])
def test_training_defaults_to_fixed_map_and_allows_varying_maps(monkeypatch, tmp_path, vary_maps, expected_seeds):
    import student.train as training

    seen = []
    original = training.run_episode

    def record(env, agent, seed):
        seen.append(seed)
        return original(env, agent, seed)

    monkeypatch.setattr(training, "run_episode", record)
    checkpoint = tmp_path / "q.json.gz"
    summary = train("model_free_control", episodes=3, seed=7, vary_maps=vary_maps,
                    checkpoint=checkpoint, eval_episodes=2)
    assert seen == expected_seeds
    assert summary.map_seed == (None if vary_maps else 7)
    assert summary.evaluation_episodes == 2
    assert checkpoint.exists()
    assert StudentAgent.load(checkpoint).episodes == 3

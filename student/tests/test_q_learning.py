"""Public baseline and runner regressions, independent of student solutions."""

from copy import deepcopy

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.agents.tabular_q_agent import TabularQAgent
from student.train import greedy_evaluation, train


@pytest.mark.parametrize("terminated,truncated,expected", [
    (False, False, 2.98), (False, True, 2.98), (True, False, 1.0),
])
def test_targets_mask_illegal_actions_and_bootstrap_only_without_termination(terminated, truncated, expected):
    env = FactoryFrontier("intro")
    observation, _ = env.reset(seed=0)
    next_observation, *_ = env.step(Action.NORTH)
    agent = TabularQAgent(learning_rate=1.0)
    agent.reset(observation)
    next_values = agent.q.setdefault(agent.state_key(next_observation), np.zeros(len(Action)))
    next_values.fill(999.0)
    next_values[next_observation["action_mask"] != 0] = 2.0
    agent.update(observation, Action.NORTH, 1.0, next_observation, terminated, truncated)
    assert agent.q[agent.state_key(observation)][Action.NORTH] == pytest.approx(expected)


def test_baseline_checkpoint_round_trip_and_evaluation_restore_rng(tmp_path):
    env = FactoryFrontier("model_free_control")
    observation, _ = env.reset(seed=0)
    agent = TabularQAgent(seed=7)
    key = agent.state_key(observation)
    agent.q[key] = np.zeros(len(Action))
    agent.q[key][Action.EAST] = 3.0
    path = tmp_path / "agent.json"
    agent.save(path)
    loaded = TabularQAgent.load(path)
    assert loaded.epsilon == 0.0
    np.testing.assert_array_equal(loaded.q[key], agent.q[key])
    assert loaded.act(observation) == Action.EAST
    rng_before = deepcopy(agent.rng.bit_generator.state)
    greedy_evaluation(agent, "model_free_control", (0, 0), seed=4)
    assert agent.epsilon == 0.1
    assert agent.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("vary_maps,expected_seeds", [(False, [7, 7, 7]), (True, [7, 8, 9])])
def test_training_defaults_to_fixed_map_and_allows_varying_maps(monkeypatch, tmp_path, vary_maps, expected_seeds):
    import student.train as training

    seen = []
    original = training.run_episode

    def record(env, agent, seed):
        seen.append(seed)
        return original(env, agent, seed)

    monkeypatch.setattr(training, "StudentAgent", TabularQAgent)
    monkeypatch.setattr(training, "run_episode", record)
    checkpoint = tmp_path / "q.json"
    summary = train("model_free_control", episodes=3, seed=7, vary_maps=vary_maps,
                    checkpoint=checkpoint, eval_episodes=2)
    assert seen == expected_seeds
    assert summary.map_seed == (None if vary_maps else 7)
    assert summary.evaluation_episodes == 2
    assert checkpoint.exists()
    assert TabularQAgent.load(checkpoint).q

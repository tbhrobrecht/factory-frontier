from __future__ import annotations

import numpy as np

from factory_frontier import FactoryFrontier
from factory_frontier.agents import ModelAdapter, RandomAgent, RuleBasedAgent
from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.competition import load_replay, play_match, render_replay, round_robin


def test_model_adapter_masks_illegal_actions() -> None:
    observation, _ = FactoryFrontier("intro").reset(seed=0)
    values = np.arange(len(observation["action_mask"]), dtype=float)
    action = ModelAdapter(lambda _: values).act(observation)
    assert observation["action_mask"][action] == 1


def test_tabular_save_load_uses_canonical_action_width(tmp_path) -> None:
    env = FactoryFrontier("intro")
    observation, _ = env.reset(seed=0)
    agent = TabularQAgent(seed=0)
    agent.q[agent.state_key(observation)] = np.arange(env.action_space.n, dtype=float)
    path = tmp_path / "agent.json"
    agent.save(path)
    loaded = TabularQAgent.load(path)
    np.testing.assert_array_equal(loaded.q[agent.state_key(observation)], agent.q[agent.state_key(observation)])


def test_competition_is_reproducible() -> None:
    first = play_match(RandomAgent(seed=11), RuleBasedAgent(), seed=99)
    second = play_match(RandomAgent(seed=11), RuleBasedAgent(), seed=99)
    assert first == second


def test_round_robin_plays_both_seats() -> None:
    board, matches = round_robin(
        {"a": RandomAgent(seed=1), "b": RandomAgent(seed=2)}, seeds=(7,)
    )
    assert len(matches) == 2
    assert all(entry.played == 2 for entry in board)
    assert all(entry.wins + entry.losses + entry.draws == 2 for entry in board)


def test_saved_replay_can_be_rendered(tmp_path) -> None:
    path = tmp_path / "replay.json"
    result = play_match(RandomAgent(seed=3), RuleBasedAgent(), seed=5, replay_path=path)
    loaded = load_replay(path)
    frames = render_replay(loaded)
    assert loaded == result
    assert len(frames) == result.steps + 1

"""Competition fairness, terminal rewards, privacy and learning regressions."""

import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.agents import Agent
from factory_frontier.competition import load_replay, play_match, render_replay
from factory_frontier.configs import get_config


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_paired_maps_and_mining_are_fair(seed):
    env = FactoryFrontier("competition", slip_probability=0)
    env.reset(seed=seed)
    mirror = lambda cells: {(15 - r, 15 - c) for r, c in cells}
    for cells in (env.state.walls, set(env.state.resources), env.state.mining_nodes):
        assert cells == mirror(cells)
    assert len(env.state.resources) == 24 and len(env.state.walls) == 30
    assert {(15, 3), (0, 12)} <= env.state.mining_nodes
    assert not any(c in (7, 8) for r, c in env.state.walls)
    env.state.factory(0).position = (15, 3)
    env.state.resources.pop((15, 3))  # Already collected on arrival.
    energy = env.state.factory(0).energy
    result = env.step_multi({0: Action.MINE, 1: Action.WAIT})
    assert env.state.factory(0).energy == energy + 1
    assert env.state.gathered[0] == 0 and result[0][1] == -0.1


@pytest.mark.parametrize("first,second,wall", [
    (Action.EAST, Action.WEST, False), (Action.BUILD_SCOUT, Action.BUILD_SCOUT, False),
    (Action.BUILD_SCOUT, Action.WEST, False), (Action.REMOVE_WALL, Action.REMOVE_WALL, True),
])
def test_simultaneous_conflicts_do_not_depend_on_dictionary_order(first, second, wall):
    states = []
    for order in ((0, 1), (1, 0)):
        env = FactoryFrontier("competition", enabled_actions=tuple(Action), slip_probability=0, danger_enabled=False)
        env.reset(seed=0)
        env.state.walls = {(5, 5)} if wall else set()
        env.state.resources.clear()
        env.state.mining_nodes.clear()
        env.state.factory(0).position, env.state.factory(1).position = (5, 4), (5, 6)
        actions = (first, second)
        env.step_multi({owner: actions[owner] for owner in order})
        states.append(env.state.copy())
    assert states[0] == states[1]
    assert len(states[0].entities) == 2
    assert [states[0].factory(owner).energy for owner in (0, 1)] == [16, 16]
    assert (5, 5) not in states[0].walls


def test_score_is_resource_first_and_terminal_with_win_loss_rewards():
    env = FactoryFrontier("competition", max_steps=1)
    env.reset(seed=0)
    env.state.gathered[0] = 1
    env.state.factory(0).energy, env.state.factory(1).energy = 1, 100
    output = env.step_multi({0: Action.WAIT, 1: Action.WAIT})
    assert output[0][2:4] == (True, False) and output[0][4]["winner"] == 0
    assert output[0][4]["reward_components"]["win"] == 10
    assert output[1][4]["reward_components"]["loss"] == -10
    assert not output[0][0]["action_mask"].any()
    env = FactoryFrontier("competition")
    env.reset(seed=0)
    for _ in range(8):
        output = env.step_multi({0: Action.WAIT, 1: Action.WAIT})
    assert env.state.winner is None and env.state.terminated
    assert output[0][1] == pytest.approx(-10.1)  # Double-death draws still penalize dying.


def test_partial_infos_exclude_opponent_events():
    env = FactoryFrontier("competition", slip_probability=0)
    env.reset(seed=0)
    output = env.step_multi({0: Action.EAST, 1: Action.WEST})
    for owner in (0, 1):
        assert output[owner][4]["events"]
        assert all(event["owner"] == owner for event in output[owner][4]["events"])
    assert {event["owner"] for event in env.state.event_log if event["event"] == "move"} == {0, 1}


def test_two_agent_training_and_custom_configuration_replay(tmp_path):
    class Learner(Agent):
        def reset(self, observation):
            self.flags = []

        def act(self, observation):
            return Action.WAIT

        def update(self, observation, action, reward, next_observation, terminated, truncated):
            self.flags.append((terminated, truncated))

    first, second = Learner(), Learner()
    config = get_config("competition").with_overrides(name="short-match", max_steps=1)
    result = play_match(first, second, config=config, training_owners=(0, 1), player_names=("a", "b"))
    assert first.flags == second.flags == [(True, False)]
    assert result.end_reason == "score" and result.players == ("a", "b")
    assert result.scores == ((1, 0, 16), (1, 0, 16))
    path = tmp_path / "match.json"
    result.save_replay(path)
    assert load_replay(path) == result
    assert len(render_replay(path)) == 2

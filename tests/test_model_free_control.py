"""Focused checks for the student control task's fuel and danger rules."""

from collections import deque
from copy import deepcopy

import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.core.actions import DELTAS, MOVEMENT_ACTIONS


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_generated_task_has_a_safe_refuelling_solution(seed):
    env = FactoryFrontier("model_free_control")
    observation, _ = env.reset(seed=seed)
    assert len(env.state.resources) == 5 and len(env.state.walls) == 5
    assert observation["grid"][7, 3] == 2 and observation["mining_nodes"][7, 3]
    for _ in range(3):
        observation, _, terminated, _, _ = env.step(Action.EAST)
        assert not terminated
    assert observation["grid"][7, 3] == 5 and observation["mining_nodes"][7, 3]
    queue, seen = deque([((7, 3), [])]), {(7, 3)}
    while queue:
        position, path = queue.popleft()
        if position == (0, 7):
            break
        for action in MOVEMENT_ACTIONS:
            dr, dc = DELTAS[action]
            target = (position[0] + dr, position[1] + dc)
            if env.state.in_bounds(target) and target not in env.state.walls and target not in seen:
                seen.add(target)
                queue.append((target, path + [action]))
    else:
        pytest.fail("Generated map has no route to the goal")
    assert len(path) == 11
    needed = len(path) - int(env.state.factory(0).energy)
    assert needed > 0  # Starting fuel alone cannot solve this task.
    for _ in range(needed):
        _, reward, terminated, _, _ = env.step(Action.MINE)
        assert reward == pytest.approx(-0.05) and not terminated
    for action in path:
        observation, _, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
    assert terminated and not truncated and info["winner"] == 0
    with pytest.raises(RuntimeError, match="hidden"):
        env.transition_model()


@pytest.mark.parametrize("step,position,boundary,action", [
    (5, (7, 1), 0, Action.WEST), (39, (1, 7), 6, Action.WAIT),
])
def test_danger_kills_on_entry_and_at_the_deadline(step, position, boundary, action):
    env = FactoryFrontier("model_free_control")
    env.reset(seed=0)
    env.state.step_count, env.state.danger_column = step, boundary
    env.state.factory(0).position = position
    _, _, terminated, truncated, info = env.step(action)
    assert terminated and not truncated and info["winner"] == 1


def test_empty_fuel_masks_moves_and_q_keys_distinguish_resource_states():
    env = FactoryFrontier("model_free_control", initial_energy=0)
    observation, _ = env.reset(seed=0)
    assert not any(observation["action_mask"][action] for action in MOVEMENT_ACTIONS)
    key = TabularQAgent.state_key(observation)
    for field in ("grid", "factory", "danger", "mining_nodes"):
        changed = deepcopy(observation)
        changed[field].flat[0] += 1
        assert TabularQAgent.state_key(changed) != key

"""Focused checks for the student control task's fuel and danger rules."""

from collections import deque
from copy import deepcopy

import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.core.actions import DELTAS, MOVEMENT_ACTIONS


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("preset", ["model_free_control", "function_approximation", "policy_methods", "ppo"])
def test_generated_task_has_a_safe_refuelling_solution(seed, preset):
    # Verify layout/fuel feasibility independently of sampled movement noise.
    env = FactoryFrontier(preset, slip_probability=0.0)
    observation, _ = env.reset(seed=seed)
    fuel, goal = (env.config.height - 1, 3), (0, env.config.width - 1)
    assert len(env.state.resources) == env.config.resource_count
    assert len(env.state.walls) == int(env.config.width * env.config.height * env.config.wall_density)
    assert observation["grid"][fuel] == 2 and observation["mining_nodes"][fuel]
    for _ in range(3):
        observation, _, terminated, _, _ = env.step(Action.EAST)
        assert not terminated
    assert observation["grid"][fuel] == 5 and observation["mining_nodes"][fuel]
    queue, seen = deque([(fuel, [])]), {fuel}
    while queue:
        position, path = queue.popleft()
        if position == goal:
            break
        for action in MOVEMENT_ACTIONS:
            dr, dc = DELTAS[action]
            target = (position[0] + dr, position[1] + dc)
            if env.state.in_bounds(target) and target not in env.state.walls and target not in seen:
                seen.add(target)
                queue.append((target, path + [action]))
    else:
        pytest.fail("Generated map has no route to the goal")
    assert len(path) == env.config.height + env.config.width - 5
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

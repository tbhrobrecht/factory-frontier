"""Room structure and observation-boundary regressions for the PPO task."""

from collections import Counter

import numpy as np

from factory_frontier import FactoryFrontier
from factory_frontier.core.entities import EntityType, make_entity
from factory_frontier.core.observations import encode_observation


def test_rooms_have_partition_walls_and_doorways():
    env = FactoryFrontier("ppo")
    env.reset(seed=7)
    rows = Counter(row for row, column in env.state.walls)
    columns = Counter(column for row, column in env.state.walls)
    # Each full-length partition has two open doors, one into each room pair.
    assert max(rows.values()) == env.config.width - 2
    assert max(columns.values()) == env.config.height - 2


def test_hidden_terrain_and_entities_do_not_leak_into_observations():
    env = FactoryFrontier("ppo")
    observation, _ = env.reset(seed=0)
    hidden = observation["visibility"] == 0
    assert hidden[:14, :14].any()
    assert not observation["grid"][hidden].any()
    assert not observation["mining_nodes"][hidden].any()
    env.state.walls.add((0, 0))
    env.state.walls.discard((0, 1))
    env.state.walls.discard((0, 2))
    env.state.resources[(0, 1)] = 1
    env.state.mining_nodes.add((0, 1))
    env.state.entities[2] = make_entity(2, EntityType.SCOUT, 0, (0, 2))
    changed = encode_observation(env.state, env.config)
    for field in observation:
        np.testing.assert_array_equal(changed[field], observation[field])
    env.state.factory(0).position = (0, 1)
    revealed = encode_observation(env.state, env.config)
    assert revealed["grid"][0, 0] == 1
    assert revealed["mining_nodes"][0, 1] == 1
    assert revealed["entity_mask"].sum() == 2

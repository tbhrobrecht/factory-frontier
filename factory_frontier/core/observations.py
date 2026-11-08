"""Stable, padded observation encoder."""

from __future__ import annotations

import numpy as np

from factory_frontier.configs.presets import CurriculumConfig
from factory_frontier.core.entities import EntityType
from factory_frontier.core.rules import action_mask
from factory_frontier.core.state import GameState


OBSERVATION_FIELDS = (
    "grid", "entities", "entity_mask", "factory", "opponent", "danger", "visibility",
    "action_mask", "map_shape", "mining_nodes",
)


def visibility_mask(state: GameState, config: CurriculumConfig, owner: int) -> np.ndarray:
    visible = np.zeros((config.max_grid_size, config.max_grid_size), dtype=np.int8)
    if not config.partial_observability:
        visible[: state.height, : state.width] = 1
        return visible
    factory = state.factory(owner)
    if factory is None:
        return visible
    row, column = factory.position
    radius = config.visibility_radius
    for r in range(max(0, row - radius), min(state.height, row + radius + 1)):
        for c in range(max(0, column - radius), min(state.width, column + radius + 1)):
            if abs(r - row) + abs(c - column) <= radius:
                visible[r, c] = 1
    return visible


def _factory_features(state: GameState, owner: int, max_steps: int) -> np.ndarray:
    entity = state.factory(owner)
    if entity is None:
        return np.zeros(8, dtype=np.float32)
    return np.asarray(
        [
            1.0,
            entity.position[0] / max(1, state.height - 1),
            entity.position[1] / max(1, state.width - 1),
            entity.energy / 100.0,
            entity.health / 3.0,
            entity.cooldown / 10.0,
            state.gathered[owner] / 100.0,
            state.step_count / max_steps,
        ],
        dtype=np.float32,
    )


def encode_observation(state: GameState, config: CurriculumConfig, owner: int = 0) -> dict[str, np.ndarray]:
    """Encode state to the same keys and shapes for every preset."""

    size = config.max_grid_size
    visible = visibility_mask(state, config, owner)
    grid = np.zeros((size, size), dtype=np.int16)
    for row, column in state.walls:
        grid[row, column] = 1
    for row, column in state.resources:
        grid[row, column] = 2
    for row, column in state.mining_nodes:
        if grid[row, column] == 0:
            grid[row, column] = 3
    for entity in state.entities.values():
        if entity.alive:
            grid[entity.position] = 4 + int(entity.kind) + (4 if entity.owner != owner else 0)
    grid[visible == 0] = 0
    mining_nodes = np.zeros((size, size), dtype=np.int8)
    for position in state.mining_nodes:
        mining_nodes[position] = visible[position]

    entities = np.zeros((config.max_entities, 8), dtype=np.float32)
    entity_mask = np.zeros(config.max_entities, dtype=np.int8)
    observable = [
        entity for entity in sorted(state.entities.values(), key=lambda item: item.id)
        if entity.alive and visible[entity.position]
    ]
    for index, entity in enumerate(observable[: config.max_entities]):
        entities[index] = [
            int(entity.kind) / len(EntityType),
            float(entity.owner == owner),
            entity.position[0] / max(1, state.height - 1),
            entity.position[1] / max(1, state.width - 1),
            entity.energy / 100.0,
            entity.health / 3.0,
            entity.cooldown / 10.0,
            1.0,
        ]
        entity_mask[index] = 1

    opponent_owner = 1 if owner == 0 else 0
    opponent = _factory_features(state, opponent_owner, config.max_steps)
    if not config.opponent_enabled or state.factory(opponent_owner) is None:
        opponent.fill(0)
    elif state.factory(opponent_owner) is not None and not visible[state.factory(opponent_owner).position]:
        opponent.fill(0)

    danger_active = float(config.danger_enabled)
    danger_position = max(-1, state.danger_column)
    factory = state.factory(owner)
    if factory is None:
        distance = 0.0
    elif owner == 0 or not config.opponent_enabled:
        distance = (factory.position[1] - danger_position) / max(1, state.width)
    else:
        distance = (state.width - 1 - danger_position - factory.position[1]) / max(1, state.width)
    danger = np.asarray(
        [danger_active, danger_position / max(1, state.width), distance, config.boundary_interval / 100.0],
        dtype=np.float32,
    )
    return {
        "grid": grid,
        "entities": entities,
        "entity_mask": entity_mask,
        "factory": _factory_features(state, owner, config.max_steps),
        "opponent": opponent,
        "danger": danger,
        "visibility": visible,
        "action_mask": action_mask(state, config, owner),
        "map_shape": np.asarray([state.height, state.width], dtype=np.int16),
        "mining_nodes": mining_nodes,
    }

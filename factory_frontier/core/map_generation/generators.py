"""Small deterministic procedural map generators."""

from __future__ import annotations

import numpy as np

from factory_frontier.configs.presets import CurriculumConfig


def _candidates(config: CurriculumConfig) -> list[tuple[int, int]]:
    reserved = {
        (config.height - 1, 0), (config.height - 2, 0), (config.height - 1, 1),
        (0, config.width - 1), (1, config.width - 1), (0, config.width - 2),
        (0, 0), (config.height - 1, config.width - 1),
    }
    return [
        (row, column)
        for row in range(config.height)
        for column in range(config.width)
        if (row, column) not in reserved
    ]


def generate_map(
    config: CurriculumConfig, rng: np.random.Generator
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Return walls, collectible resources and renewable mining nodes."""

    cells = _candidates(config)
    rng.shuffle(cells)
    wall_count = int(config.width * config.height * config.wall_density)
    if config.map_generator == "empty":
        wall_count = 0
    walls = set(cells[:wall_count])
    if config.map_generator == "symmetric":
        reflected = {(config.height - 1 - r, config.width - 1 - c) for r, c in walls}
        walls |= reflected
        # Never seal the two spawn corners.
        walls -= {(config.height - 1, 0), (0, config.width - 1)}
    available = [cell for cell in cells if cell not in walls]
    rng.shuffle(available)
    count = min(config.resource_count if config.resources_enabled else 0, len(available))
    if config.map_generator == "symmetric":
        selected: set[tuple[int, int]] = set()
        for cell in available:
            mirror = (config.height - 1 - cell[0], config.width - 1 - cell[1])
            if mirror not in walls:
                selected.update((cell, mirror))
            if len(selected) >= count:
                break
        resource_cells = sorted(selected)[:count]
    else:
        resource_cells = available[:count]
    resources = {cell: 1 for cell in resource_cells}
    nodes = set(resource_cells[::4])
    return walls, resources, nodes

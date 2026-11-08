"""Small deterministic procedural map generators."""

from __future__ import annotations

from collections import deque

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


def _room_layout(
    config: CurriculumConfig, rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Four rooms with open doors and a protected monotone refuelling route."""
    if min(config.width, config.height) < 8:
        raise ValueError("The rooms generator requires width and height of at least 8")
    row = int(rng.integers(3, config.height - 3))
    column = int(rng.integers(4, config.width - 3))
    lower_door = int(rng.integers(row + 1, config.height))
    right_door = int(rng.integers(column + 1, config.width))
    doors = {(row, int(rng.integers(0, column))), (row, right_door),
             (int(rng.integers(0, row)), column), (lower_door, column)}
    walls = {(row, c) for c in range(config.width)} | {(r, column) for r in range(config.height)}
    walls -= doors
    # Approach and depart through each doorway without walking along its wall.
    r, c = config.height - 1, 3
    route = {(r, c) for c in range(4)}
    waypoints = [(lower_door, column - 1), (lower_door, column + 1),
                 (row + 1, right_door), (row - 1, right_door), (0, config.width - 1)]
    for target_r, target_c in waypoints:
        directions = [(-1, 0)] * (r - target_r) + [(0, 1)] * (target_c - c)
        rng.shuffle(directions)
        for dr, dc in directions:
            r, c = r + dr, c + dc
            route.add((r, c))
    # Keep every door and its approaches clear when adding interior obstacles.
    protected = route | doors
    for r, c in doors:
        protected.update(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
    return walls, protected


def _symmetric_map(
    config: CurriculumConfig, rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Paired terrain and fuel, with escape lanes into a shared central arena."""
    width, height = config.width, config.height
    count = config.resource_count if config.resources_enabled else 0
    if width < 8 or height < 4 or width % 2 or height % 2 or count % 2:
        raise ValueError("Symmetric maps need even dimensions (width >= 8, height >= 4) and an even resource count")

    def mirror(cell):
        return height - 1 - cell[0], width - 1 - cell[1]

    def pair(cell):
        return tuple(sorted((cell, mirror(cell))))

    center = (width // 2 - 1, width // 2)
    protected = {(r, c) for r in range(height) for c in center}
    escape = {(height - 1, c) for c in range(width // 2)}
    protected |= escape | {mirror(cell) for cell in escape}
    pairs = [pair(cell) for cell in _candidates(config) if cell < mirror(cell)]
    rng.shuffle(pairs)
    wall_pairs = [p for p in pairs if not protected.intersection(p)]
    walls = {cell for p in wall_pairs[:max(0, int(width * height * config.wall_density) // 2)] for cell in p}
    # Resources are placed only in the component shared by both spawn corners.
    reachable, queue = {(height - 1, 0)}, deque([(height - 1, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = r + dr, c + dc
            if 0 <= cell[0] < height and 0 <= cell[1] < width and cell not in walls and cell not in reachable:
                reachable.add(cell)
                queue.append(cell)
    available = [p for p in pairs if set(p) <= reachable]
    priority = [pair((height - 1, min(3, center[0] - 1))), pair((height // 2 - 1, center[0]))]
    available = priority + [p for p in available if p not in priority]
    if count // 2 > len(available):
        raise ValueError("Not enough reachable mirrored cells for the requested resources")
    selected = available[:count // 2]
    resources = {cell: 1 for p in selected for cell in p}
    nodes = {cell for p in selected[::4] if p[0][1] not in center for cell in p}
    return walls, resources, nodes


def generate_map(
    config: CurriculumConfig, rng: np.random.Generator
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Return walls, collectible resources and renewable mining nodes."""

    if config.map_generator == "symmetric":
        return _symmetric_map(config, rng)
    cells = _candidates(config)
    route: set[tuple[int, int]] = set()
    room_walls: set[tuple[int, int]] = set()
    fuel = (config.height - 1, min(3, config.width - 1))
    if config.map_generator == "control":
        # A safe refuelling stop followed by a randomized monotone route. Keep
        # these cells free of walls so default tasks are solvable, not lotteries.
        row, column = config.height - 1, 0
        route.add((row, column))
        while column < fuel[1]:
            column += 1
            route.add((row, column))
        directions = [(0, 1)] * (config.width - 1 - column) + [(-1, 0)] * row
        rng.shuffle(directions)
        for dr, dc in directions:
            row, column = row + dr, column + dc
            route.add((row, column))
    elif config.map_generator == "rooms":
        room_walls, route = _room_layout(config, rng)
    rng.shuffle(cells)
    wall_count = int(config.width * config.height * config.wall_density)
    if config.map_generator == "empty":
        wall_count = 0
    candidates = [cell for cell in cells if cell not in route and cell not in room_walls]
    walls = room_walls | set(candidates[:max(0, wall_count - len(room_walls))])
    available = [cell for cell in cells if cell not in walls]
    rng.shuffle(available)
    count = min(config.resource_count if config.resources_enabled else 0, len(available))
    if config.map_generator in ("control", "rooms") and count:
        available = [fuel, *(cell for cell in available if cell != fuel)]
    resource_cells = available[:count]
    resources = {cell: 1 for cell in resource_cells}
    nodes = set(resource_cells[::4])
    return walls, resources, nodes

"""Dependency-free ASCII renderer."""

from __future__ import annotations

from factory_frontier.core.entities import EntityType
from factory_frontier.core.state import GameState


GLYPHS = {
    EntityType.FACTORY: "F",
    EntityType.SCOUT: "S",
    EntityType.WORKER: "W",
    EntityType.MINER: "M",
}


def render_ascii(state: GameState) -> str:
    canvas = [["." for _ in range(state.width)] for _ in range(state.height)]
    for row, column in state.mining_nodes:
        canvas[row][column] = "n"
    for row, column in state.resources:
        canvas[row][column] = "*"
    for row, column in state.walls:
        canvas[row][column] = "#"
    for entity in state.entities.values():
        if entity.alive:
            glyph = GLYPHS[entity.kind]
            canvas[entity.position[0]][entity.position[1]] = glyph if entity.owner == 0 else glyph.lower()
    return "\n".join(" ".join(row) for row in canvas)

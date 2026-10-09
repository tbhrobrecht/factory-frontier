"""Central mutable episode state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factory_frontier.core.entities import Entity, EntityType


@dataclass(slots=True)
class GameState:
    """All mutable game data. Rules mutate only this object."""

    width: int
    height: int
    walls: set[tuple[int, int]] = field(default_factory=set)
    resources: dict[tuple[int, int], int] = field(default_factory=dict)
    mining_nodes: set[tuple[int, int]] = field(default_factory=set)
    entities: dict[int, Entity] = field(default_factory=dict)
    step_count: int = 0
    danger_column: int = -1
    next_entity_id: int = 1
    gathered: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    energy_spent: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    event_log: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    winner: int | None = None

    def factory(self, owner: int = 0) -> Entity | None:
        return next(
            (e for e in self.entities.values() if e.owner == owner and e.kind == EntityType.FACTORY and e.alive),
            None,
        )

    def entity_at(self, position: tuple[int, int]) -> Entity | None:
        return next((e for e in self.entities.values() if e.alive and e.position == position), None)

    def in_bounds(self, position: tuple[int, int]) -> bool:
        row, column = position
        return 0 <= row < self.height and 0 <= column < self.width

    def copy(self) -> "GameState":
        return GameState(
            width=self.width,
            height=self.height,
            walls=set(self.walls),
            resources=dict(self.resources),
            mining_nodes=set(self.mining_nodes),
            entities={key: value.copy() for key, value in self.entities.items()},
            step_count=self.step_count,
            danger_column=self.danger_column,
            next_entity_id=self.next_entity_id,
            gathered=dict(self.gathered),
            energy_spent=dict(self.energy_spent),
            event_log=[dict(event) for event in self.event_log],
            terminated=self.terminated,
            winner=self.winner,
        )

    def log(self, event: str, **data: Any) -> None:
        self.event_log.append({"step": self.step_count, "event": event, **data})


def seeded_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


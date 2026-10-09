"""Data-only entities; behavior lives in rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum


class EntityType(IntEnum):
    FACTORY = 1
    SCOUT = 2
    WORKER = 3
    MINER = 4


@dataclass(slots=True)
class Entity:
    """A serializable engine entity."""

    id: int
    kind: EntityType
    owner: int
    position: tuple[int, int]
    energy: float = 0.0
    health: float = 1.0
    cooldown: int = 0
    alive: bool = True

    def copy(self) -> "Entity":
        return replace(self)


def make_entity(
    entity_id: int,
    kind: EntityType,
    owner: int,
    position: tuple[int, int],
    energy: float = 0.0,
) -> Entity:
    """Factory for built-in entity types, usable by custom rules."""

    health = 3.0 if kind == EntityType.FACTORY else 1.0
    return Entity(entity_id, kind, owner, position, energy=energy, health=health)


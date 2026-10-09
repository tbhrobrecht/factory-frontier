"""The immutable action vocabulary shared by every curriculum stage."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Canonical actions. Append new actions; never reorder existing members."""

    WAIT = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    JUMP_NORTH = 5
    JUMP_SOUTH = 6
    JUMP_EAST = 7
    JUMP_WEST = 8
    BUILD_SCOUT = 9
    BUILD_WORKER = 10
    BUILD_MINER = 11
    BUILD_WALL = 12
    REMOVE_WALL = 13
    MINE = 14


ACTION_COUNT = len(Action)
MOVEMENT_ACTIONS = (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)
JUMP_ACTIONS = (
    Action.JUMP_NORTH,
    Action.JUMP_SOUTH,
    Action.JUMP_EAST,
    Action.JUMP_WEST,
)

DELTAS: dict[Action, tuple[int, int]] = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
    Action.JUMP_NORTH: (-2, 0),
    Action.JUMP_SOUTH: (2, 0),
    Action.JUMP_EAST: (0, 2),
    Action.JUMP_WEST: (0, -2),
}


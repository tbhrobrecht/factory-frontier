"""A deliberately simple competition baseline that handles either seat."""

import numpy as np

from factory_frontier.agents.base import Agent, Observation
from factory_frontier.core.actions import Action


class SurvivalAgent(Agent):
    """Escape toward the central columns, then wait; no resource planning."""

    def act(self, observation: Observation) -> int:
        locations = np.argwhere(observation["grid"] == 5)
        if not len(locations):
            return int(Action.WAIT)
        row, column = locations[0]
        height, width = observation["map_shape"]
        mask = observation["action_mask"]
        if width // 2 - 1 <= column <= width // 2:
            return int(Action.WAIT)
        forward = (Action.JUMP_EAST, Action.EAST) if column < width // 2 else (Action.JUMP_WEST, Action.WEST)
        vertical = (Action.NORTH, Action.SOUTH) if row >= height // 2 else (Action.SOUTH, Action.NORTH)
        for action in (*forward, *vertical, Action.MINE, Action.WAIT):
            if mask[action]:
                return int(action)
        return int(Action.WAIT)

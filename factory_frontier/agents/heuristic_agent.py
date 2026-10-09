"""Intentionally simple goal-directed baseline."""

from __future__ import annotations

from factory_frontier.agents.base import Agent, Observation
from factory_frontier.core.actions import Action


class RuleBasedAgent(Agent):
    """Prefer north/east progress, then choose any legal action."""

    def act(self, observation: Observation) -> int:
        mask = observation["action_mask"]
        for action in (Action.NORTH, Action.EAST, Action.MINE, Action.SOUTH, Action.WEST, Action.WAIT):
            if mask[int(action)]:
                return int(action)
        legal = mask.nonzero()[0]
        return int(legal[0]) if len(legal) else int(Action.WAIT)


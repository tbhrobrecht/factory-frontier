"""Copy this file to start an assignment."""

from __future__ import annotations

import numpy as np

from factory_frontier.agents.base import Agent, Observation


class StudentAgent(Agent):
    def reset(self, observation: Observation) -> None:
        """Reset episode-specific memory here."""

    def act(self, observation: Observation) -> int:
        """Replace this legal random policy with your algorithm."""
        legal_actions = np.flatnonzero(observation["action_mask"])
        return int(np.random.choice(legal_actions)) if len(legal_actions) else 0


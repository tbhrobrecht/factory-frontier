"""Starter agent for students to extend throughout the course."""

from __future__ import annotations

import numpy as np

from factory_frontier.agents import Agent, Observation


class StudentAgent(Agent):
    """A legal random baseline; replace its policy with your own algorithm."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, observation: Observation) -> None:
        """Reset episode-specific memory here when your agent needs it."""

    def act(self, observation: Observation) -> int:
        """Select an action. The starter policy samples uniformly from legal actions."""

        legal_actions = np.flatnonzero(observation["action_mask"])
        return int(self.rng.choice(legal_actions)) if len(legal_actions) else 0

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Learn from one transition; intentionally empty in the starter agent."""

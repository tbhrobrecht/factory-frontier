"""Uniform legal-action baseline."""

from __future__ import annotations

import numpy as np

from factory_frontier.agents.base import Agent, Observation


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, observation: Observation) -> None:
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        return int(self.rng.choice(legal)) if len(legal) else 0


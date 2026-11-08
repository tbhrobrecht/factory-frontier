"""Minimal educational tabular Q-learning baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from factory_frontier.agents.base import Agent, Observation
from factory_frontier.core.actions import ACTION_COUNT


class TabularQAgent(Agent):
    """A deliberately small example keyed only by Factory position."""

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1, seed: int | None = None) -> None:
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def state_key(observation: Observation) -> tuple[int, int]:
        location = np.argwhere((observation["grid"] == 5))
        return tuple(int(value) for value in location[0]) if len(location) else (-1, -1)

    def act(self, observation: Observation) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        if not len(legal):
            return 0
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        values = self.q.setdefault(self.state_key(observation), np.zeros(ACTION_COUNT))
        masked = np.full(ACTION_COUNT, -np.inf)
        masked[legal] = values[legal]
        return int(np.argmax(masked))

    def update(self, observation: Observation, action: int, reward: float, next_observation: Observation,
               terminated: bool, truncated: bool = False) -> None:
        key = self.state_key(observation)
        next_key = self.state_key(next_observation)
        values = self.q.setdefault(key, np.zeros(ACTION_COUNT))
        next_values = self.q.setdefault(next_key, np.zeros(ACTION_COUNT))
        legal = np.flatnonzero(next_observation["action_mask"])
        bootstrap = float(np.max(next_values[legal])) if not terminated and len(legal) else 0.0
        target = reward + 0.99 * bootstrap
        values[action] += self.learning_rate * (target - values[action])

    def save(self, path: str | Path) -> None:
        payload = {f"{r},{c}": values.tolist() for (r, c), values in self.q.items()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularQAgent":
        agent = cls(epsilon=0.0)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent.q = {
            tuple(int(part) for part in key.split(",")): np.asarray(values, dtype=float)
            for key, values in payload.items()
        }
        return agent


"""Minimal educational tabular Q-learning baseline."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from factory_frontier.agents.base import Agent, Observation
from factory_frontier.core.actions import ACTION_COUNT


class TabularQAgent(Agent):
    """Small exact-observation baseline; students can design coarser state keys.

    Exact keys distinguish maps, fuel, resources and danger, but do not
    generalize between similar states or maps. Rich tasks need state design
    or function approximation for efficient learning.
    """

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.1, seed: int | None = None) -> None:
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q: dict[str, np.ndarray] = {}

    @staticmethod
    def state_key(observation: Observation) -> str:
        payload = {name: values.tolist() for name, values in observation.items()}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

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
        payload = {key: values.tolist() for key, values in self.q.items()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularQAgent":
        agent = cls(epsilon=0.0)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent.q = {
            key: np.asarray(values, dtype=float)
            for key, values in payload.items()
        }
        return agent


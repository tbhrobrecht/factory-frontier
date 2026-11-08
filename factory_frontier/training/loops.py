"""A minimal loop students can replace with their own algorithm."""

from __future__ import annotations

from typing import Protocol

from factory_frontier.agents.base import Observation
from factory_frontier.core.environment import FactoryFrontier


class LearningAgent(Protocol):
    def reset(self, observation: Observation) -> None: ...
    def act(self, observation: Observation) -> int: ...
    def update(self, observation: Observation, action: int, reward: float, next_observation: Observation,
               terminated: bool, truncated: bool = False) -> None: ...


def run_training_episode(env: FactoryFrontier, agent: LearningAgent, seed: int | None = None) -> float:
    observation, _ = env.reset(seed=seed)
    agent.reset(observation)
    total = 0.0
    done = False
    while not done:
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.update(observation, action, reward, next_observation, terminated, truncated)
        observation = next_observation
        total += reward
    return total


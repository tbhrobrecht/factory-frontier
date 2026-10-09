"""Reproducible single-agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factory_frontier.agents.base import Agent
from factory_frontier.core.environment import FactoryFrontier


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    returns: tuple[float, ...]
    lengths: tuple[int, ...]
    successes: int

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0


def evaluate_agent(agent: Agent, config: str = "intro", seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> EvaluationResult:
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    for seed in seeds:
        env = FactoryFrontier(config)
        observation, _ = env.reset(seed=seed)
        agent.reset(observation)
        total = 0.0
        done = False
        info = {}
        while not done:
            observation, reward, terminated, truncated, info = env.step(agent.act(observation))
            total += reward
            done = terminated or truncated
        returns.append(total)
        lengths.append(int(info["step"]))
        successes += int(info["winner"] == 0)
    return EvaluationResult(tuple(returns), tuple(lengths), successes)

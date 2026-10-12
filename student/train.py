"""Minimal training scaffold; add the course algorithm inside StudentAgent.update."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from factory_frontier import FactoryFrontier
from student.agent import StudentAgent


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    config: str
    episodes: int
    mean_return: float
    successes: int


def run_episode(env: FactoryFrontier, agent: StudentAgent, seed: int) -> tuple[float, bool]:
    observation, _ = env.reset(seed=seed)
    agent.reset(observation)
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(observation, action, reward, next_observation, done)
        observation = next_observation
        total_reward += reward
    return total_reward, info.get("winner") == 0


def train(config: str = "intro", episodes: int = 100, seed: int = 0) -> TrainingSummary:
    """Run training episodes. The starter update is a no-op until students implement it."""

    env = FactoryFrontier(config)
    agent = StudentAgent(seed=seed)
    returns: list[float] = []
    successes = 0
    for episode in range(episodes):
        episode_return, success = run_episode(env, agent, seed + episode)
        returns.append(episode_return)
        successes += int(success)
    return TrainingSummary(config, episodes, float(np.mean(returns)), successes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="intro")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = train(args.config, args.episodes, args.seed)
    output = Path(__file__).parent / "results" / "training_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(summary)
    print(f"Saved summary to {output}")
    print("The starter update() does not learn yet; implement it as the course progresses.")


if __name__ == "__main__":
    main()


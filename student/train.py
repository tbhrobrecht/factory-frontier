"""Run student training, evaluation and optional checkpoint saving."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from factory_frontier import FactoryFrontier
from factory_frontier.evaluation import evaluate_agent
from student.agent import StudentAgent


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    config: str
    episodes: int
    mean_return: float
    successes: int
    map_seed: int | None = None
    baseline_successes: int = 0
    greedy_successes: int = 0
    evaluation_episodes: int = 0
    baseline_mean_return: float = 0.0
    greedy_mean_return: float = 0.0
    first_100_mean_return: float = 0.0
    last_100_mean_return: float = 0.0
    checkpoint: str = ""


def run_episode(env: FactoryFrontier, agent: StudentAgent, seed: int) -> tuple[float, bool]:
    if env.config.opponent_enabled:
        raise ValueError("Use play_match(..., training_owners=...) for two-agent training")
    observation, _ = env.reset(seed=seed)
    agent.reset(observation)
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(observation, action, reward, next_observation, terminated, truncated)
        observation = next_observation
        total_reward += reward
    return total_reward, info.get("winner") == 0


def greedy_evaluation(agent, config, seeds, seed):
    """Evaluate without learning or consuming the training RNG sequence."""
    saved = {name: getattr(agent, name) for name in ("epsilon", "rng", "episode_return")
             if hasattr(agent, name)}
    try:
        if "epsilon" in saved:
            agent.epsilon = 0.0
        if "rng" in saved:
            agent.rng = np.random.default_rng(seed)
        return evaluate_agent(agent, config=config, seeds=seeds)
    finally:
        for name, value in saved.items():
            setattr(agent, name, value)


def train(
    config: str = "intro", episodes: int = 100, seed: int = 0, *,
    map_seed: int | None = None, vary_maps: bool = False,
    alpha: float | None = None, epsilon: float | None = None,
    eval_episodes: int = 20, checkpoint: str | Path | None = None,
) -> TrainingSummary:
    """Repeat one map by default for model_free_control; vary maps explicitly."""
    if episodes < 1 or eval_episodes < 1:
        raise ValueError("episodes and eval_episodes must be positive")
    if vary_maps and map_seed is not None:
        raise ValueError("Choose map_seed or vary_maps, not both")
    if config == "model_free_control" and map_seed is None and not vary_maps:
        map_seed = seed

    env = FactoryFrontier(config)
    agent = StudentAgent(seed=seed)
    # Students own algorithm parameters in agent.py. CLI overrides are optional.
    for name, value in (("alpha", alpha), ("epsilon", epsilon)):
        if value is not None:
            if not hasattr(agent, name):
                raise ValueError(f"This StudentAgent does not expose {name}")
            if not np.isfinite(value) or not (0 < value <= 1 if name == "alpha" else 0 <= value <= 1):
                raise ValueError(f"Invalid {name}")
            setattr(agent, name, value)
    if checkpoint is not None and not callable(getattr(agent, "save", None)):
        raise ValueError("--checkpoint requires StudentAgent.save(path)")
    evaluation_seeds = ((map_seed,) * eval_episodes if map_seed is not None else
                        tuple(range(seed + episodes, seed + episodes + eval_episodes)))
    baseline = greedy_evaluation(agent, config, evaluation_seeds, seed)
    print(f"Training maps: {'fixed seed ' + str(map_seed) if map_seed is not None else 'varying seeds'}")
    returns: list[float] = []
    successes = 0
    for episode in range(episodes):
        episode_return, success = run_episode(
            env, agent, map_seed if map_seed is not None else seed + episode
        )
        returns.append(episode_return)
        successes += int(success)
    finish = getattr(agent, "finish_training", None)
    if callable(finish):
        finish()
    greedy = greedy_evaluation(agent, config, evaluation_seeds, seed)
    env.close()
    saved_checkpoint = ""
    if callable(getattr(agent, "save", None)):
        checkpoint = Path(checkpoint) if checkpoint is not None else (
            Path(__file__).parent / "checkpoints" / f"{config}.checkpoint"
        )
        metadata = getattr(agent, "checkpoint_metadata", None)
        if isinstance(metadata, dict):
            metadata.update(config=config, map_seed=map_seed, agent_seed=seed)
        agent.save(checkpoint)
        saved_checkpoint = str(checkpoint)
    print(f"Evaluation (epsilon=0 when supported): baseline {baseline.successes}/{eval_episodes} wins, "
          f"mean return {baseline.mean_return:.3f}; trained {greedy.successes}/{eval_episodes} wins, "
          f"mean return {greedy.mean_return:.3f}")
    return TrainingSummary(
        config, episodes, float(np.mean(returns)), successes,
        map_seed=map_seed, baseline_successes=baseline.successes,
        greedy_successes=greedy.successes, evaluation_episodes=eval_episodes,
        baseline_mean_return=baseline.mean_return, greedy_mean_return=greedy.mean_return,
        first_100_mean_return=float(np.mean(returns[:100])),
        last_100_mean_return=float(np.mean(returns[-100:])), checkpoint=saved_checkpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="intro")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--map-seed", type=int, help="Repeat this environment seed every episode")
    maps.add_argument("--vary-maps", action="store_true", help="Use a new map seed every episode")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, help="Output path; requires StudentAgent.save(path)")
    args = parser.parse_args()

    summary = train(
        args.config, args.episodes, args.seed, map_seed=args.map_seed,
        vary_maps=args.vary_maps, alpha=args.alpha, epsilon=args.epsilon,
        eval_episodes=args.eval_episodes, checkpoint=args.checkpoint,
    )
    output = Path(__file__).parent / "results" / "training_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(summary)
    print(f"Saved summary to {output}")
    if summary.checkpoint:
        print(f"Saved checkpoint to {summary.checkpoint}")


if __name__ == "__main__":
    main()


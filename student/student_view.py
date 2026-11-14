"""Run the student agent visually against a curriculum environment."""

from __future__ import annotations

import argparse
import numpy as np

from factory_frontier import Action, FactoryFrontier
from student.agent import StudentAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="intro", help="Curriculum preset name")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible episode seed")
    parser.add_argument("--checkpoint", help="Load a trained agent; requires StudentAgent.load(path)")
    parser.add_argument(
        "--render-steps",
        type=int,
        default=12,
        help="Maximum number of turns to print; the episode still runs to completion",
    )
    args = parser.parse_args()

    env = FactoryFrontier(args.config, render_mode="ansi")
    if args.checkpoint:
        loader = getattr(StudentAgent, "load", None)
        if not callable(loader):
            parser.error("--checkpoint requires StudentAgent.load(path)")
        agent = loader(args.checkpoint)
        if hasattr(agent, "rng"):
            agent.rng = np.random.default_rng(args.seed)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0
    else:
        agent = StudentAgent(seed=args.seed)
    metadata = getattr(agent, "checkpoint_metadata", {})
    if args.checkpoint and metadata.get("config") not in (None, args.config):
        parser.error("Checkpoint configuration does not match --config")
    observation, info = env.reset(seed=args.seed)
    agent.reset(observation)

    print("Initial observation fields:")
    for name, value in observation.items():
        print(f"  {name}: shape={value.shape}")

    done = False
    total_reward = 0.0
    while not done:
        should_render = info["step"] < args.render_steps
        if should_render:
            print("\n" + env.render())
            legal = [action.name for action in Action if observation["action_mask"][action]]
            print("Legal actions:", legal)

        action = agent.act(observation)
        if should_render:
            print("Chosen action:", Action(action).name)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if should_render:
            print("Reward:", reward)
            print("Reward components:", info["reward_components"])

    print("\nFinal map:")
    print(env.render())
    print("Winner:", info["winner"])
    print("Episode length:", info["step"])
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()


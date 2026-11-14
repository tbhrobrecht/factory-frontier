"""Evaluate the student agent on fixed seeds without changing the engine."""

from __future__ import annotations

import argparse
import numpy as np

from factory_frontier.evaluation import evaluate_agent
from student.agent import StudentAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="intro")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--agent-seed", type=int, default=0)
    parser.add_argument("--checkpoint", help="Load a trained agent; requires StudentAgent.load(path)")
    args = parser.parse_args()

    if args.checkpoint:
        loader = getattr(StudentAgent, "load", None)
        if not callable(loader):
            parser.error("--checkpoint requires StudentAgent.load(path)")
        agent = loader(args.checkpoint)
        if hasattr(agent, "rng"):
            agent.rng = np.random.default_rng(args.agent_seed)
    else:
        agent = StudentAgent(seed=args.agent_seed)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    metadata = getattr(agent, "checkpoint_metadata", {})
    if args.checkpoint and metadata.get("config") not in (None, args.config):
        parser.error("Checkpoint configuration does not match --config")
    result = evaluate_agent(
        agent,
        config=args.config,
        seeds=tuple(args.seeds),
    )
    print("Returns:", result.returns)
    print("Episode lengths:", result.lengths)
    print("Successes:", result.successes, "/", len(result.returns))
    print("Mean return:", result.mean_return)


if __name__ == "__main__":
    main()


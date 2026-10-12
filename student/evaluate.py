"""Evaluate the student agent on fixed seeds without changing the engine."""

from __future__ import annotations

import argparse

from factory_frontier.evaluation import evaluate_agent
from student.agent import StudentAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="intro")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--agent-seed", type=int, default=0)
    args = parser.parse_args()

    result = evaluate_agent(
        StudentAgent(seed=args.agent_seed),
        config=args.config,
        seeds=tuple(args.seeds),
    )
    print("Returns:", result.returns)
    print("Episode lengths:", result.lengths)
    print("Successes:", result.successes, "/", len(result.returns))
    print("Mean return:", result.mean_return)


if __name__ == "__main__":
    main()


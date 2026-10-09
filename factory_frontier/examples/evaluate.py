"""Load and evaluate the tabular example."""

from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.evaluation import evaluate_agent


def main() -> None:
    result = evaluate_agent(TabularQAgent.load("tabular_agent.json"), "intro")
    print(f"mean return={result.mean_return:.3f}, successes={result.successes}")


if __name__ == "__main__":
    main()


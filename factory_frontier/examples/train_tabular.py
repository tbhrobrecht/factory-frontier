"""Train and save the intentionally basic tabular example."""

from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.core.environment import FactoryFrontier
from factory_frontier.training import run_training_episode


def main() -> None:
    env = FactoryFrontier("intro")
    agent = TabularQAgent(seed=7)
    for episode in range(500):
        run_training_episode(env, agent, seed=episode)
    agent.save("tabular_agent.json")


if __name__ == "__main__":
    main()


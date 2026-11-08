"""Run a tiny reproducible tournament."""

from factory_frontier.agents import RandomAgent, SurvivalAgent
from factory_frontier.competition import round_robin


def main() -> None:
    leaderboard, _ = round_robin(
        {"random": lambda: RandomAgent(seed=1), "survival": SurvivalAgent}, seeds=(10, 11)
    )
    for entry in leaderboard:
        print(entry.name, entry.points, entry.wins, entry.draws, entry.losses)


if __name__ == "__main__":
    main()

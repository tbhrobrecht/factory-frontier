"""Deterministic round-robin aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from factory_frontier.agents.base import Agent
from factory_frontier.competition.match import MatchResult, play_match


@dataclass(slots=True)
class LeaderboardEntry:
    name: str
    played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    resources: int = 0
    survival_steps: int = 0
    energy_spent: float = 0.0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws


def round_robin(
    agents: dict[str, Agent], seeds: tuple[int, ...] = (0, 1, 2, 3, 4), config: str = "competition"
) -> tuple[list[LeaderboardEntry], list[MatchResult]]:
    """Play both seat assignments for every pair and seed."""

    board = {name: LeaderboardEntry(name) for name in agents}
    matches: list[MatchResult] = []
    for first_name, second_name in combinations(sorted(agents), 2):
        for seed in seeds:
            for left_name, right_name in ((first_name, second_name), (second_name, first_name)):
                result = play_match(agents[left_name], agents[right_name], seed=seed, config=config)
                matches.append(result)
                left, right = board[left_name], board[right_name]
                left.played += 1
                right.played += 1
                left.resources += result.resources[0]
                right.resources += result.resources[1]
                left.energy_spent += result.energy_spent[0]
                right.energy_spent += result.energy_spent[1]
                left.survival_steps += result.steps
                right.survival_steps += result.steps
                if result.winner is None:
                    left.draws += 1
                    right.draws += 1
                elif result.winner == 0:
                    left.wins += 1
                    right.losses += 1
                else:
                    right.wins += 1
                    left.losses += 1
    ranking = sorted(board.values(), key=lambda item: (-item.points, -item.wins, -item.resources, item.name))
    return ranking, matches


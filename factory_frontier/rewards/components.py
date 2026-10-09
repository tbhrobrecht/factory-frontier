"""Reward functions kept separate from state-transition rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from factory_frontier.core.state import GameState


@dataclass(frozen=True, slots=True)
class RewardContext:
    before: GameState
    after: GameState
    owner: int
    invalid_action: bool = False


class RewardComponent(Protocol):
    name: str

    def __call__(self, context: RewardContext) -> float: ...


@dataclass(frozen=True, slots=True)
class TimeReward:
    name: str = "time"

    def __call__(self, context: RewardContext) -> float:
        return 1.0


@dataclass(frozen=True, slots=True)
class ResourceReward:
    name: str = "resource"

    def __call__(self, context: RewardContext) -> float:
        return float(context.after.gathered[context.owner] - context.before.gathered[context.owner])


@dataclass(frozen=True, slots=True)
class ProgressReward:
    name: str = "progress"

    def __call__(self, context: RewardContext) -> float:
        old = context.before.factory(context.owner)
        new = context.after.factory(context.owner)
        if old is None or new is None:
            return 0.0
        target = (0, context.after.width - 1) if context.owner == 0 else (context.after.height - 1, 0)
        old_distance = abs(old.position[0] - target[0]) + abs(old.position[1] - target[1])
        new_distance = abs(new.position[0] - target[0]) + abs(new.position[1] - target[1])
        return float(old_distance - new_distance)


@dataclass(frozen=True, slots=True)
class EnergyEfficiencyReward:
    name: str = "energy"

    def __call__(self, context: RewardContext) -> float:
        return -(context.after.energy_spent[context.owner] - context.before.energy_spent[context.owner])


@dataclass(frozen=True, slots=True)
class SurvivalReward:
    name: str = "survival"

    def __call__(self, context: RewardContext) -> float:
        return float(context.after.factory(context.owner) is not None)


@dataclass(frozen=True, slots=True)
class UnitLossReward:
    name: str = "unit_loss"

    def __call__(self, context: RewardContext) -> float:
        before = sum(e.alive and e.owner == context.owner for e in context.before.entities.values())
        after = sum(e.alive and e.owner == context.owner for e in context.after.entities.values())
        return -float(max(0, before - after))


@dataclass(frozen=True, slots=True)
class WinReward:
    name: str = "win"

    def __call__(self, context: RewardContext) -> float:
        return float(context.after.terminated and context.after.winner == context.owner)


@dataclass(frozen=True, slots=True)
class LossReward:
    name: str = "loss"

    def __call__(self, context: RewardContext) -> float:
        return float(
            context.after.terminated
            and context.after.winner is not None
            and context.after.winner != context.owner
        )


@dataclass(frozen=True, slots=True)
class OpponentDefeatReward:
    name: str = "opponent_defeat"

    def __call__(self, context: RewardContext) -> float:
        opponent = 1 if context.owner == 0 else 0
        return float(context.before.factory(opponent) is not None and context.after.factory(opponent) is None)


DEFAULT_COMPONENTS: tuple[RewardComponent, ...] = (
    TimeReward(), ResourceReward(), ProgressReward(), EnergyEfficiencyReward(), SurvivalReward(),
    UnitLossReward(), WinReward(), LossReward(), OpponentDefeatReward(),
)


class RewardSystem:
    """Weighted composition with a per-component diagnostic breakdown."""

    def __init__(self, weights: dict[str, float], components: tuple[RewardComponent, ...] = DEFAULT_COMPONENTS):
        self.weights = dict(weights)
        self.components = components

    def calculate(self, context: RewardContext) -> tuple[float, dict[str, float]]:
        breakdown = {
            component.name: component(context) * self.weights.get(component.name, 0.0)
            for component in self.components
        }
        return float(sum(breakdown.values())), breakdown

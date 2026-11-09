"""Declarative curriculum presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral, Real
from typing import Any

from factory_frontier.core.actions import Action, JUMP_ACTIONS, MOVEMENT_ACTIONS


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Complete engine configuration. Use :meth:`with_overrides` to customize."""

    name: str
    width: int = 5
    height: int = 5
    map_generator: str = "empty"
    enabled_actions: tuple[Action, ...] = (Action.WAIT, *MOVEMENT_ACTIONS)
    slip_probability: float = 0.0
    expose_model: bool = False
    resources_enabled: bool = False
    opponent_enabled: bool = False
    partial_observability: bool = False
    visibility_radius: int = 3
    danger_enabled: bool = False
    boundary_interval: int = 5
    initial_energy: float = 20.0
    movement_cost: float = 0.0
    jump_cost: float = 2.0
    mining_counts_as_resource: bool = True
    build_costs: dict[Action, float] = field(
        default_factory=lambda: {
            Action.BUILD_SCOUT: 4.0,
            Action.BUILD_WORKER: 5.0,
            Action.BUILD_MINER: 6.0,
            Action.BUILD_WALL: 2.0,
        }
    )
    build_cooldown: int = 2
    max_steps: int = 100
    resource_count: int = 0
    wall_density: float = 0.0
    max_grid_size: int = 16
    max_entities: int = 32
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"time": -0.01, "resource": 1.0, "win": 10.0, "loss": -10.0}
    )

    def __post_init__(self) -> None:
        for name in ("width", "height", "max_grid_size", "max_entities", "max_steps", "boundary_interval",
                     "visibility_radius", "build_cooldown", "resource_count"):
            value = getattr(self, name)
            minimum = 0 if name in ("visibility_radius", "build_cooldown", "resource_count") else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.width > self.max_grid_size or self.height > self.max_grid_size:
            raise ValueError("Map dimensions exceed stable observation padding")
        if self.opponent_enabled and self.max_entities < 2:
            raise ValueError("Opponent configurations need max_entities >= 2")
        if self.map_generator not in {"empty", "random", "control", "rooms", "symmetric"}:
            raise ValueError(f"Unknown map_generator {self.map_generator!r}")
        if self.map_generator == "rooms" and min(self.width, self.height) < 8:
            raise ValueError("The rooms generator requires width and height of at least 8")
        if self.map_generator == "symmetric" and (
            self.width < 8 or self.height < 4 or self.width % 2 or self.height % 2
            or (self.resources_enabled and self.resource_count % 2)
        ):
            raise ValueError("Symmetric maps need even dimensions (width >= 8, height >= 4) and an even resource count")
        for name in ("slip_probability", "wall_density", "initial_energy", "movement_cost", "jump_cost"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name in ("slip_probability", "wall_density") and value > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.enabled_actions or any(
            isinstance(action, bool) or not isinstance(action, Integral) or action not in tuple(Action)
            for action in self.enabled_actions
        ):
            raise ValueError("enabled_actions must contain canonical action indices")
        build_actions = {Action.BUILD_SCOUT, Action.BUILD_WORKER, Action.BUILD_MINER, Action.BUILD_WALL}
        if any(isinstance(action, bool) or not isinstance(action, Integral) for action in self.build_costs) or (
            set(self.build_costs) - build_actions
            or (set(self.enabled_actions) & build_actions) - set(self.build_costs)
        ):
            raise ValueError("build_costs must specify costs for enabled builds and only use build action keys")
        for costs, nonnegative in ((self.build_costs, True), (self.reward_weights, False)):
            if any(not isinstance(value, Real) or not isfinite(value) or (nonnegative and value < 0)
                   for value in costs.values()):
                raise ValueError("Build costs must be finite and nonnegative; reward weights must be finite")
        # Own mappings supplied through overrides as well as direct construction.
        object.__setattr__(self, "enabled_actions", tuple(Action(action) for action in self.enabled_actions))
        object.__setattr__(self, "build_costs", {Action(action): cost for action, cost in self.build_costs.items()})
        object.__setattr__(self, "reward_weights", dict(self.reward_weights))

    def with_overrides(self, **values: Any) -> "CurriculumConfig":
        unknown = set(values) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unknown config fields: {sorted(unknown)}")
        return replace(self, **values)


_MOVE = (Action.WAIT, *MOVEMENT_ACTIONS)

PRESETS: dict[str, CurriculumConfig] = {
    "intro": CurriculumConfig("intro", max_steps=40),
    "mdp": CurriculumConfig("mdp", slip_probability=0.2, expose_model=True, max_steps=50),
    "dynamic_programming": CurriculumConfig(
        "dynamic_programming", slip_probability=0.1, expose_model=True, max_steps=50
    ),
    "monte_carlo_td": CurriculumConfig(
        "monte_carlo_td", width=6, height=6, slip_probability=0.1, max_steps=75
    ),
    "model_free_control": CurriculumConfig(
        "model_free_control", width=8, height=8, map_generator="control", enabled_actions=(*_MOVE, Action.MINE),
        resources_enabled=True, danger_enabled=True, resource_count=5, wall_density=0.08,
        initial_energy=8.0, movement_cost=1.0, max_steps=40,
        reward_weights={"time": -0.1, "resource": 0.05, "win": 10.0, "loss": -10.0},
    ),
    "function_approximation": CurriculumConfig(
        "function_approximation", width=12, height=12, map_generator="control",
        enabled_actions=(*_MOVE, *JUMP_ACTIONS, Action.MINE),
        resources_enabled=True, danger_enabled=True, resource_count=14, wall_density=0.12,
        initial_energy=16.0, movement_cost=1.0, jump_cost=2.0, max_steps=60,
        reward_weights={"time": -0.1, "resource": 0.05, "win": 10.0, "loss": -10.0},
    ),
    "policy_methods": CurriculumConfig(
        "policy_methods", width=12, height=12, map_generator="control",
        enabled_actions=(*_MOVE, *JUMP_ACTIONS, Action.MINE),
        slip_probability=0.08, resources_enabled=True, danger_enabled=True, resource_count=14,
        wall_density=0.12, initial_energy=16.0, movement_cost=1.0, jump_cost=2.0,
        boundary_interval=6, max_steps=72,
        reward_weights={"time": -0.1, "resource": 0.05, "win": 10.0, "loss": -10.0},
    ),
    "ppo": CurriculumConfig(
        "ppo", width=14, height=14, map_generator="rooms",
        enabled_actions=(*_MOVE, *JUMP_ACTIONS, Action.MINE),
        slip_probability=0.05, resources_enabled=True, danger_enabled=True, partial_observability=True,
        resource_count=20, wall_density=0.14, initial_energy=20.0, movement_cost=1.0, jump_cost=2.0,
        boundary_interval=8, max_steps=112,
        reward_weights={"time": -0.1, "resource": 0.05, "win": 10.0, "loss": -10.0},
    ),
    "competition": CurriculumConfig(
        "competition", width=16, height=16, map_generator="symmetric",
        enabled_actions=(*_MOVE, *JUMP_ACTIONS, Action.MINE),
        slip_probability=0.05, resources_enabled=True, opponent_enabled=True, danger_enabled=True,
        partial_observability=True, visibility_radius=4, resource_count=24, wall_density=0.12,
        initial_energy=16.0, movement_cost=1.0, jump_cost=2.0,
        mining_counts_as_resource=False, boundary_interval=8, max_steps=60,
        reward_weights={"time": -0.1, "resource": 0.05, "win": 10.0, "loss": -10.0, "unit_loss": 10.0},
    ),
}


def get_config(config: str | CurriculumConfig = "intro", **overrides: Any) -> CurriculumConfig:
    """Resolve a preset name or copy a config with validated overrides."""

    if isinstance(config, str):
        try:
            resolved = PRESETS[config]
        except KeyError as error:
            raise ValueError(f"Unknown config {config!r}; choose from {sorted(PRESETS)}") from error
    elif isinstance(config, CurriculumConfig):
        resolved = config
    else:
        raise TypeError("config must be a preset name or CurriculumConfig")
    return resolved.with_overrides(**overrides)

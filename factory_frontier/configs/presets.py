"""Declarative curriculum presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

    def with_overrides(self, **values: Any) -> "CurriculumConfig":
        unknown = set(values) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unknown config fields: {sorted(unknown)}")
        return replace(self, **values)


_MOVE = (Action.WAIT, *MOVEMENT_ACTIONS)
_RICH = (*_MOVE, *tuple(Action)[5:])

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
        "ppo", width=14, height=14, map_generator="rooms", enabled_actions=_RICH,
        slip_probability=0.05, resources_enabled=True, danger_enabled=True, partial_observability=True,
        resource_count=20, wall_density=0.14, movement_cost=0.1, max_steps=350,
    ),
    "competition": CurriculumConfig(
        "competition", width=16, height=16, map_generator="symmetric", enabled_actions=_RICH,
        slip_probability=0.05, resources_enabled=True, opponent_enabled=True, danger_enabled=True,
        partial_observability=True, visibility_radius=4, resource_count=24, wall_density=0.12,
        movement_cost=0.1, max_steps=400,
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
    # Defensive copies keep global presets immutable even though mapping values are mutable.
    resolved = replace(
        resolved,
        build_costs=dict(resolved.build_costs),
        reward_weights=dict(resolved.reward_weights),
    )
    result = resolved.with_overrides(**overrides) if overrides else resolved
    if result.width > result.max_grid_size or result.height > result.max_grid_size:
        raise ValueError("Map dimensions exceed stable observation padding")
    return result

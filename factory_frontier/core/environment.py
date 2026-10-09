"""Gymnasium-compatible Factory Frontier environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factory_frontier.configs import CurriculumConfig, get_config
from factory_frontier.core.actions import ACTION_COUNT, MOVEMENT_ACTIONS, Action
from factory_frontier.core.entities import EntityType, make_entity
from factory_frontier.core.map_generation import generate_map
from factory_frontier.core.observations import encode_observation
from factory_frontier.core.rules import RuleEngine, RuleHook, movement_distribution
from factory_frontier.core.state import GameState
from factory_frontier.rewards import RewardContext, RewardSystem
from factory_frontier.rewards.components import RewardComponent


class FactoryFrontier(gym.Env[dict[str, np.ndarray], int]):
    """One engine configured for every curriculum stage.

    The normal Gymnasium ``step`` controls owner 0. Competition code calls
    :meth:`step_multi` to resolve both players simultaneously with the same rules.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: str | CurriculumConfig = "intro",
        render_mode: str | None = None,
        rules: Sequence[RuleHook] = (),
        reward_components: tuple[RewardComponent, ...] | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        self.config = get_config(config, **overrides)
        self.render_mode = render_mode
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode='ansi' is supported")
        self.action_space = spaces.Discrete(ACTION_COUNT)
        size = self.config.max_grid_size
        count = self.config.max_entities
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0, 12, (size, size), dtype=np.int16),
                "entities": spaces.Box(-1.0, 10.0, (count, 8), dtype=np.float32),
                "entity_mask": spaces.MultiBinary(count),
                "factory": spaces.Box(-1.0, 10.0, (8,), dtype=np.float32),
                "opponent": spaces.Box(-1.0, 10.0, (8,), dtype=np.float32),
                "danger": spaces.Box(-1.0, 10.0, (4,), dtype=np.float32),
                "visibility": spaces.MultiBinary((size, size)),
                "action_mask": spaces.MultiBinary(ACTION_COUNT),
                "map_shape": spaces.Box(0, size, (2,), dtype=np.int16),
            }
        )
        self.rule_engine = RuleEngine(self.config, rules)
        self.reward_system = (
            RewardSystem(self.config.reward_weights, reward_components)
            if reward_components is not None
            else RewardSystem(self.config.reward_weights)
        )
        self.state: GameState | None = None
        self._episode_done = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        walls, resources, nodes = generate_map(self.config, self.np_random)
        state = GameState(
            width=self.config.width,
            height=self.config.height,
            walls=walls,
            resources=resources,
            mining_nodes=nodes,
        )
        first = make_entity(
            state.next_entity_id, EntityType.FACTORY, 0, (state.height - 1, 0), self.config.initial_energy
        )
        state.entities[first.id] = first
        state.next_entity_id += 1
        if self.config.opponent_enabled:
            second = make_entity(
                state.next_entity_id, EntityType.FACTORY, 1, (0, state.width - 1), self.config.initial_energy
            )
            state.entities[second.id] = second
            state.next_entity_id += 1
        state.log("reset", seed=seed, config=self.config.name)
        self.state = state
        self._episode_done = False
        observation = encode_observation(state, self.config, 0)
        return observation, self._info(0, seed=seed)

    def step(self, action: int | Action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance owner 0; an enabled opponent waits outside competition matches."""

        results = self.step_multi({0: action, **({1: Action.WAIT} if self.config.opponent_enabled else {})})
        return results[0]

    def step_multi(
        self, actions: Mapping[int, int | Action]
    ) -> dict[int, tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]]:
        """Resolve multiple owners simultaneously and return per-owner Gym tuples."""

        state = self._require_state()
        if self._episode_done:
            raise RuntimeError("Episode is finished; call reset()")
        expected = {0, 1} if self.config.opponent_enabled else {0}
        if set(actions) != expected:
            raise ValueError(f"Expected actions for owners {sorted(expected)}, got {sorted(actions)}")
        before = state.copy()
        invalid = self.rule_engine.apply_actions(state, actions, self.np_random)
        truncated = state.step_count >= self.config.max_steps and not state.terminated
        if truncated and self.config.opponent_enabled:
            state.winner = self._score_winner(state)
        self._episode_done = state.terminated or truncated
        output = {}
        for owner in sorted(actions):
            reward, breakdown = self.reward_system.calculate(
                RewardContext(before, state, owner, invalid.get(owner, False))
            )
            output[owner] = (
                encode_observation(state, self.config, owner),
                reward,
                state.terminated,
                truncated,
                self._info(owner, reward_components=breakdown, invalid_action=invalid.get(owner, False)),
            )
        return output

    def _score_winner(self, state: GameState) -> int | None:
        scores = {}
        for owner in (0, 1):
            factory = state.factory(owner)
            scores[owner] = (
                int(factory is not None) * 1000
                + state.gathered[owner] * 10
                + (factory.energy if factory else 0.0)
            )
        return 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else None

    def _info(self, owner: int, **extra: Any) -> dict[str, Any]:
        state = self._require_state()
        factory = state.factory(owner)
        return {
            "config": self.config.name,
            "owner": owner,
            "step": state.step_count,
            "winner": state.winner,
            "resources_gathered": state.gathered[owner],
            "energy_spent": state.energy_spent[owner],
            "factory_alive": factory is not None,
            "events": [dict(event) for event in state.event_log],
            **extra,
        }

    def transition_model(
        self,
    ) -> dict[tuple[int, int], dict[int, list[tuple[float, tuple[int, int], float, bool]]]]:
        """Return P(s'|s,a), reward and terminal flags for finite model-based presets."""

        state = self._require_state()
        if not self.config.expose_model:
            raise RuntimeError(f"Transition model is hidden in the {self.config.name!r} curriculum stage")
        model: dict[tuple[int, int], dict[int, list[tuple[float, tuple[int, int], float, bool]]]] = {}
        actions = (Action.WAIT, *MOVEMENT_ACTIONS)
        for row in range(state.height):
            for column in range(state.width):
                position = (row, column)
                if position in state.walls:
                    continue
                model[position] = {}
                for action in actions:
                    outcomes: dict[tuple[tuple[int, int], float, bool], float] = {}
                    for realized, probability in movement_distribution(action, self.config.slip_probability).items():
                        trial = state.copy()
                        trial.step_count = 0
                        trial.terminated = False
                        trial.winner = None
                        factory = trial.factory(0)
                        assert factory is not None
                        factory.position = position
                        before = trial.copy()
                        deterministic = self.config.with_overrides(slip_probability=0.0)
                        RuleEngine(deterministic, self.rule_engine.extra_rules).apply_actions(
                            trial, {0: realized}, np.random.default_rng(0)
                        )
                        reward, _ = self.reward_system.calculate(RewardContext(before, trial, 0))
                        next_factory = trial.factory(0)
                        next_position = next_factory.position if next_factory else position
                        key = (next_position, reward, trial.terminated)
                        outcomes[key] = outcomes.get(key, 0.0) + probability
                    model[position][int(action)] = [
                        (probability, next_position, reward, terminal)
                        for (next_position, reward, terminal), probability in outcomes.items()
                    ]
        return model

    def render(self) -> str:
        from factory_frontier.rendering.ascii import render_ascii

        return render_ascii(self._require_state())

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the environment")
        return self.state

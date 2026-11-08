"""Agent-vs-agent matches using the shared engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from factory_frontier.agents.base import Agent
from factory_frontier.configs import CurriculumConfig
from factory_frontier.core.environment import FactoryFrontier


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _json_safe_event(event: dict[str, Any]) -> dict[str, object]:
    return {str(key): _json_safe(value) for key, value in event.items()}


@dataclass(frozen=True, slots=True)
class MatchResult:
    config: str
    seed: int
    winner: int | None
    steps: int
    resources: tuple[int, int]
    energy_spent: tuple[float, float]
    factory_progress: tuple[int, int]
    events: tuple[dict[str, object], ...]
    players: tuple[str, str]
    scores: tuple[tuple[int, int, float], tuple[int, int, float]]
    end_reason: str
    settings: dict[str, Any]

    def save_replay(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def play_match(
    first: Agent,
    second: Agent,
    seed: int = 0,
    config: str | CurriculumConfig = "competition",
    replay_path: str | Path | None = None,
    training_owners: tuple[int, ...] = (),
    player_names: tuple[str, str] | None = None,
) -> MatchResult:
    """Play simultaneous turns; optionally update selected learning agents.

    Evaluation is the default. Use separate agent instances for per-seat memory,
    even when they deliberately share model weights. Updates receive the chosen
    action and both ending flags, never the opponent's private observation.
    """
    agents = (first, second)
    if first is second:
        raise ValueError("Use separate agent instances for the two seats")
    if not set(training_owners) <= {0, 1}:
        raise ValueError("Training owners must be 0 and/or 1")
    updates = {}
    for owner in sorted(set(training_owners)):
        update = getattr(agents[owner], "update", None)
        if not callable(update):
            raise TypeError(f"Training agent {owner} must implement update()")
        updates[owner] = update
    env = FactoryFrontier(config)
    if not env.config.opponent_enabled:
        raise ValueError("Competition matches require opponent_enabled=True")
    first_observation, _ = env.reset(seed=seed)
    assert env.state is not None
    from factory_frontier.core.observations import encode_observation

    second_observation = encode_observation(env.state, env.config, 1)
    first.reset(first_observation)
    second.reset(second_observation)
    done = False
    while not done:
        observations = (first_observation, second_observation)
        actions = {owner: agent.act(observations[owner]) for owner, agent in enumerate(agents)}
        outputs = env.step_multi(actions)
        # Both actions are collected before either learner is updated.
        for owner, update in updates.items():
            next_observation, reward, terminated, truncated, _ = outputs[owner]
            update(observations[owner], actions[owner], reward, next_observation, terminated, truncated)
        first_observation, _, terminated, truncated, _ = outputs[0]
        second_observation = outputs[1][0]
        done = terminated or truncated
    state = env.state
    assert state is not None
    progress = []
    for owner in (0, 1):
        factory = state.factory(owner)
        if factory is None:
            progress.append(-1)
        elif owner == 0:
            progress.append(factory.position[1] + (state.height - 1 - factory.position[0]))
        else:
            progress.append((state.width - 1 - factory.position[1]) + factory.position[0])
    scores = env._competition_scores(state)
    result = MatchResult(
        config=env.config.name,
        seed=seed,
        winner=state.winner,
        steps=state.step_count,
        resources=(state.gathered[0], state.gathered[1]),
        energy_spent=(state.energy_spent[0], state.energy_spent[1]),
        factory_progress=(progress[0], progress[1]),
        events=tuple(_json_safe_event(event) for event in state.event_log),
        players=player_names if player_names is not None else (type(first).__name__, type(second).__name__),
        scores=(scores[0], scores[1]),
        end_reason="score" if all(state.factory(owner) is not None for owner in (0, 1)) else "survival",
        settings=asdict(env.config),
    )
    if replay_path is not None:
        result.save_replay(replay_path)
    return result

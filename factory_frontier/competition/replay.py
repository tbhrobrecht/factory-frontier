"""Load and render deterministic match replays from logged actions."""

from __future__ import annotations

import json
from pathlib import Path

from factory_frontier.competition.match import MatchResult
from factory_frontier.configs import CurriculumConfig
from factory_frontier.core.actions import Action
from factory_frontier.core.environment import FactoryFrontier


def load_replay(path: str | Path) -> MatchResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MatchResult(
        config=payload["config"],
        seed=int(payload["seed"]),
        winner=payload["winner"],
        steps=int(payload["steps"]),
        resources=tuple(payload["resources"]),
        energy_spent=tuple(payload["energy_spent"]),
        factory_progress=tuple(payload["factory_progress"]),
        events=tuple(payload["events"]),
        players=tuple(payload["players"]),
        scores=tuple(tuple(score) for score in payload["scores"]),
        end_reason=payload["end_reason"],
        settings=_decode_settings(payload["settings"]),
    )


def _decode_settings(settings: dict) -> dict:
    settings = dict(settings)
    settings["enabled_actions"] = tuple(Action(action) for action in settings["enabled_actions"])
    settings["build_costs"] = {Action(int(action)): cost for action, cost in settings["build_costs"].items()}
    return settings


def render_replay(replay: MatchResult | str | Path) -> tuple[str, ...]:
    """Re-run logged requested actions and return an ASCII frame for every state."""

    result = load_replay(replay) if isinstance(replay, (str, Path)) else replay
    env = FactoryFrontier(CurriculumConfig(**_decode_settings(result.settings)))
    env.reset(seed=result.seed)
    frames = [env.render()]
    actions_by_step: dict[int, dict[int, int]] = {}
    for event in result.events:
        if event.get("event") == "action":
            actions_by_step.setdefault(int(event["step"]), {})[int(event["owner"])] = int(event["requested"])
    for step in range(result.steps):
        actions = actions_by_step.get(step)
        if actions is None:
            raise ValueError(f"Replay has no actions for step {step}")
        env.step_multi(actions)
        frames.append(env.render())
    return tuple(frames)

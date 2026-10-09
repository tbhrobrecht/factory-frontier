"""Adapters that keep trained model APIs out of the engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from factory_frontier.agents.base import Agent, Observation


class ModelAdapter(Agent):
    """Wrap a callable or object with ``predict`` and apply the legal-action mask.

    The model may return one action, logits, Q-values, or ``(output, state)`` as
    used by several RL libraries. Full-sized outputs remain compatible with all
    curriculum presets because the canonical action count never changes.
    """

    def __init__(self, model: Any, preprocess: Callable[[Observation], Any] | None = None) -> None:
        self.model = model
        self.preprocess = preprocess or (lambda observation: observation)

    def act(self, observation: Observation) -> int:
        model_input = self.preprocess(observation)
        output = self.model.predict(model_input) if hasattr(self.model, "predict") else self.model(model_input)
        if isinstance(output, tuple):
            output = output[0]
        values = np.asarray(output)
        if values.size == 1:
            candidate = int(values.reshape(-1)[0])
            if 0 <= candidate < len(observation["action_mask"]) and observation["action_mask"][candidate]:
                return candidate
            return self._first_legal(observation)
        values = values.reshape(-1).astype(float)
        if len(values) != len(observation["action_mask"]):
            raise ValueError("Model output must be a scalar action or one value per canonical action")
        values[observation["action_mask"] == 0] = -np.inf
        return int(np.argmax(values)) if np.isfinite(values).any() else self._first_legal(observation)

    @staticmethod
    def _first_legal(observation: Observation) -> int:
        legal = np.flatnonzero(observation["action_mask"])
        return int(legal[0]) if len(legal) else 0


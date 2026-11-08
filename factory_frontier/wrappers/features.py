"""NumPy features and legal-action masking for student approximators."""

from __future__ import annotations

import numpy as np

from factory_frontier.agents.base import Observation


def encode_features(observation: Observation) -> np.ndarray:
    """Return a flat float32 vector (4165 values with default observation sizes).

    Layout: 13 categorical grid channels, visibility, mining nodes, masked entity
    features, entity mask, factory, opponent, danger, normalized map dimensions,
    and action mask. Grid channels are last before flattening. Unseen/padded
    cells have no active grid channel. Scalar/entity features retain the engine's
    existing scaling; no fitted statistics or episode memory are needed.
    """
    grid = observation["grid"]
    visible = observation["visibility"]
    channels = np.eye(13, dtype=np.float32)[grid] * visible[..., None]
    parts = (
        channels, visible, observation["mining_nodes"] * visible,
        observation["entities"] * observation["entity_mask"][:, None],
        observation["entity_mask"], observation["factory"], observation["opponent"],
        observation["danger"], observation["map_shape"] / np.asarray(grid.shape),
        observation["action_mask"],
    )
    return np.concatenate([part.reshape(-1) for part in parts]).astype(np.float32)


def mask_action_values(values: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Copy one action-value/logit vector, replacing illegal entries with -inf.

    Use before greedy selection or softmax, not on probabilities. A terminal
    all-zero mask returns all -inf; callers must skip selection/bootstrapping
    when there are no legal actions. Framework tensor training needs an
    equivalent differentiable mask in that framework.
    """
    values = np.asarray(values, dtype=np.float32)
    legal = np.asarray(action_mask, dtype=bool)
    if values.ndim != 1 or values.shape != legal.shape:
        raise ValueError("Expected one value per action and a matching one-dimensional mask")
    if not np.isfinite(values[legal]).all():
        raise ValueError("Legal action values must be finite")
    return np.where(legal, values, -np.inf)

"""The complete student-facing agent contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


Observation = dict[str, np.ndarray]


class Agent(ABC):
    """Anything that implements this interface can train, evaluate and compete."""

    def reset(self, observation: Observation) -> None:
        """Start an episode. Override when the agent has recurrent state."""

    @abstractmethod
    def act(self, observation: Observation) -> int:
        """Return one canonical action index."""


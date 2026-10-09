"""Public package interface for Factory Frontier."""

from factory_frontier.configs import CurriculumConfig, get_config
from factory_frontier.core.actions import Action
from factory_frontier.core.environment import FactoryFrontier

try:
    from gymnasium.envs.registration import register, registry

    if "FactoryFrontier-v0" not in registry:
        register(id="FactoryFrontier-v0", entry_point="factory_frontier.core.environment:FactoryFrontier")
except ImportError:  # pragma: no cover - Gymnasium is a declared dependency.
    pass

__all__ = ["Action", "CurriculumConfig", "FactoryFrontier", "get_config"]
__version__ = "0.1.0"

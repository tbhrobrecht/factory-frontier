from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from factory_frontier import Action, FactoryFrontier
from factory_frontier.configs import PRESETS
from factory_frontier.core.entities import EntityType
from factory_frontier.core.observations import OBSERVATION_FIELDS
from factory_frontier.core.rules import movement_distribution
from factory_frontier.rewards import RewardContext


def test_deterministic_seed() -> None:
    first, _ = FactoryFrontier("ppo").reset(seed=42)
    second, _ = FactoryFrontier("ppo").reset(seed=42)
    for field in OBSERVATION_FIELDS:
        np.testing.assert_array_equal(first[field], second[field])


def test_registered_gymnasium_entry_point() -> None:
    env = gym.make("FactoryFrontier-v0", config="intro")
    observation, _ = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    env.close()


def test_movement_and_wall_collision() -> None:
    env = FactoryFrontier("intro")
    env.reset(seed=0)
    assert env.state is not None
    factory = env.state.factory(0)
    assert factory is not None
    env.step(Action.NORTH)
    assert factory.position == (3, 0)
    env.state.walls.add((2, 0))
    observation, _, _, _, info = env.step(Action.NORTH)
    assert factory.position == (3, 0)
    assert info["invalid_action"] is True
    assert observation["action_mask"][Action.NORTH] == 0


def test_action_mask_has_canonical_shape_and_disables_advanced_actions() -> None:
    observation, _ = FactoryFrontier("intro").reset(seed=0)
    assert observation["action_mask"].shape == (len(Action),)
    assert observation["action_mask"][Action.BUILD_SCOUT] == 0
    assert observation["action_mask"][Action.NORTH] == 1


def test_entity_creation_cost_and_cooldown() -> None:
    env = FactoryFrontier("intro", enabled_actions=(Action.WAIT, Action.BUILD_SCOUT))
    observation, _ = env.reset(seed=0)
    initial_energy = float(observation["factory"][3])
    observation, _, _, _, _ = env.step(Action.BUILD_SCOUT)
    assert env.state is not None
    scouts = [entity for entity in env.state.entities.values() if entity.kind == EntityType.SCOUT]
    assert len(scouts) == 1
    assert observation["factory"][3] < initial_energy
    assert observation["action_mask"][Action.BUILD_SCOUT] == 0


def test_resource_reward_component() -> None:
    env = FactoryFrontier("intro", reward_weights={"resource": 2.0})
    env.reset(seed=0)
    assert env.state is not None
    env.state.resources[(3, 0)] = 1
    _, reward, _, _, info = env.step(Action.NORTH)
    assert reward == 2.0
    assert info["reward_components"]["resource"] == 2.0


def test_stochastic_transition_distribution_and_exposed_model() -> None:
    distribution = movement_distribution(Action.NORTH, 0.3)
    assert len(distribution) == 4
    assert sum(distribution.values()) == pytest.approx(1.0)
    env = FactoryFrontier("mdp")
    env.reset(seed=0)
    model = env.transition_model()
    assert (4, 0) in model
    outcomes = model[(4, 0)][int(Action.NORTH)]
    assert sum(item[0] for item in outcomes) == pytest.approx(1.0)
    hidden = FactoryFrontier("monte_carlo_td")
    hidden.reset(seed=0)
    with pytest.raises(RuntimeError):
        hidden.transition_model()


def test_goal_is_terminal() -> None:
    env = FactoryFrontier("intro")
    env.reset(seed=0)
    assert env.state is not None
    factory = env.state.factory(0)
    assert factory is not None
    factory.position = (0, env.state.width - 2)
    _, _, terminated, truncated, info = env.step(Action.EAST)
    assert terminated and not truncated
    assert info["winner"] == 0


@pytest.mark.parametrize("preset", tuple(PRESETS))
def test_observation_contract_is_stable_across_configs(preset: str) -> None:
    env = FactoryFrontier(preset)
    observation, _ = env.reset(seed=3)
    assert tuple(observation) == OBSERVATION_FIELDS
    assert env.observation_space.contains(observation)
    assert observation["grid"].shape == (16, 16)
    assert observation["entities"].shape == (32, 8)


def test_config_switching_changes_rules_not_interfaces() -> None:
    intro = FactoryFrontier("intro")
    advanced = FactoryFrontier("ppo")
    first, _ = intro.reset(seed=1)
    second, _ = advanced.reset(seed=1)
    assert first.keys() == second.keys()
    assert first["grid"].shape == second["grid"].shape
    assert first["action_mask"][Action.JUMP_EAST] == 0
    assert second["action_mask"][Action.JUMP_EAST] == 1


def test_custom_rule_and_reward_are_injectable() -> None:
    class BonusRule:
        def before_step(self, state, actions, config, rng) -> None:
            state.log("custom_before")

        def after_step(self, state, actions, config, rng) -> None:
            state.gathered[0] += 1

    class BonusReward:
        name = "bonus"

        def __call__(self, context: RewardContext) -> float:
            return float(context.after.gathered[0] - context.before.gathered[0])

    env = FactoryFrontier(
        "intro",
        rules=(BonusRule(),),
        reward_components=(BonusReward(),),
        reward_weights={"bonus": 3.0},
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(Action.WAIT)
    assert reward == 3.0
    assert any(event["event"] == "custom_before" for event in info["events"])


@pytest.mark.parametrize("overrides", [
    {"width": 0}, {"height": -1}, {"width": 2.5}, {"max_steps": 0},
    {"boundary_interval": 0}, {"visibility_radius": -1}, {"max_entities": 0},
    {"slip_probability": 1.5}, {"wall_density": float("nan")}, {"movement_cost": -1},
    {"map_generator": "typo"}, {"map_generator": "rooms"},
    {"map_generator": "symmetric", "width": 8, "height": 5},
    {"enabled_actions": (15,)}, {"enabled_actions": (Action.BUILD_SCOUT,), "build_costs": {}},
    {"reward_weights": {"win": float("inf")}},
])
def test_invalid_configs_fail_before_reset(overrides: dict) -> None:
    with pytest.raises(ValueError):
        FactoryFrontier("intro", **overrides)


def test_config_overrides_own_their_mappings() -> None:
    from factory_frontier.configs import get_config

    weights = {"win": 2.0}
    config = get_config("intro", reward_weights=weights)
    weights["win"] = 999.0
    copy = config.with_overrides()
    copy.reward_weights["win"] = 10.0
    assert config.reward_weights == {"win": 2.0}
    with pytest.raises(ValueError):
        config.with_overrides(max_steps=0)

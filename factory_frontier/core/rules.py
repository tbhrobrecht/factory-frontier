"""Rules and centralized state transitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from factory_frontier.configs.presets import CurriculumConfig
from factory_frontier.core.actions import ACTION_COUNT, DELTAS, JUMP_ACTIONS, MOVEMENT_ACTIONS, Action
from factory_frontier.core.entities import Entity, EntityType, make_entity
from factory_frontier.core.state import GameState


BUILD_TYPES = {
    Action.BUILD_SCOUT: EntityType.SCOUT,
    Action.BUILD_WORKER: EntityType.WORKER,
    Action.BUILD_MINER: EntityType.MINER,
}


class RuleHook(Protocol):
    """Optional mechanic hook for instructor extensions."""

    def before_step(
        self,
        state: GameState,
        actions: Mapping[int, int | Action],
        config: CurriculumConfig,
        rng: np.random.Generator,
    ) -> None: ...

    def after_step(
        self,
        state: GameState,
        actions: Mapping[int, int | Action],
        config: CurriculumConfig,
        rng: np.random.Generator,
    ) -> None: ...


def adjacent_cells(state: GameState, position: tuple[int, int], owner: int = 0) -> list[tuple[int, int]]:
    """Return in-bounds neighboring cells in a stable preference order."""

    row, column = position
    sign = -1 if owner == 1 else 1
    values = [(row, column + sign), (row + sign, column), (row, column - sign), (row - sign, column)]
    return [cell for cell in values if state.in_bounds(cell)]


def first_free_adjacent(state: GameState, position: tuple[int, int], owner: int = 0) -> tuple[int, int] | None:
    return next(
        (cell for cell in adjacent_cells(state, position, owner) if cell not in state.walls and state.entity_at(cell) is None),
        None,
    )


def action_mask(state: GameState, config: CurriculumConfig, owner: int = 0) -> np.ndarray:
    """Compute legal actions without changing canonical action positions."""

    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    factory = state.factory(owner)
    if factory is None or state.terminated:
        return mask
    for action in config.enabled_actions:
        mask[int(action)] = 1
    for action, delta in DELTAS.items():
        if not mask[int(action)]:
            continue
        destination = (factory.position[0] + delta[0], factory.position[1] + delta[1])
        cost = config.jump_cost if action in JUMP_ACTIONS else config.movement_cost
        if (
            not state.in_bounds(destination)
            or destination in state.walls
            or state.entity_at(destination) is not None
            or factory.energy < cost
        ):
            mask[int(action)] = 0
    free = first_free_adjacent(state, factory.position, owner)
    for action, cost in config.build_costs.items():
        if int(action) < ACTION_COUNT and mask[int(action)] and (
            factory.cooldown > 0 or factory.energy < cost or free is None
        ):
            mask[int(action)] = 0
    if mask[int(Action.REMOVE_WALL)] and not any(c in state.walls for c in adjacent_cells(state, factory.position)):
        mask[int(Action.REMOVE_WALL)] = 0
    if mask[int(Action.MINE)] and factory.position not in state.mining_nodes:
        mask[int(Action.MINE)] = 0
    return mask


def sample_movement(action: Action, probability: float, rng: np.random.Generator) -> Action:
    if action not in MOVEMENT_ACTIONS or probability <= 0 or rng.random() >= probability:
        return action
    alternatives = [candidate for candidate in MOVEMENT_ACTIONS if candidate != action]
    return alternatives[int(rng.integers(len(alternatives)))]


def movement_distribution(action: Action, probability: float) -> dict[Action, float]:
    if action not in MOVEMENT_ACTIONS or probability <= 0:
        return {action: 1.0}
    return {
        candidate: (1.0 - probability if candidate == action else probability / 3.0)
        for candidate in MOVEMENT_ACTIONS
    }


class RuleEngine:
    """Applies single or simultaneous owner actions to a ``GameState``."""

    def __init__(self, config: CurriculumConfig, extra_rules: Sequence[RuleHook] = ()):
        self.config = config
        self.extra_rules = tuple(extra_rules)

    def apply_actions(
        self, state: GameState, actions: Mapping[int, int | Action], rng: np.random.Generator
    ) -> dict[int, bool]:
        if state.terminated:
            raise RuntimeError("Cannot step a terminated episode; call reset()")
        actions = dict(sorted(actions.items()))
        for rule in self.extra_rules:
            rule.before_step(state, actions, self.config, rng)
        before_masks = {owner: action_mask(state, self.config, owner) for owner in actions}
        normalized: dict[int, Action] = {}
        invalid: dict[int, bool] = {}
        requested: dict[int, int] = {}
        for owner, raw_action in actions.items():
            try:
                action = Action(int(raw_action))
            except (ValueError, TypeError):
                action = Action.WAIT
                requested[owner] = -1
                invalid[owner] = True
            else:
                requested[owner] = int(action)
                invalid[owner] = not bool(before_masks[owner][int(action)])
            normalized[owner] = Action.WAIT if invalid[owner] else sample_movement(
                action, self.config.slip_probability, rng
            )
            state.log(
                "action",
                owner=owner,
                requested=requested[owner],
                resolved=int(normalized[owner]),
                invalid=invalid[owner],
            )

        # All spatial targets use the turn's starting state. Conflicting builds
        # and moves cancel without spending energy, irrespective of input order.
        targets: dict[int, tuple[int, int] | None] = {}
        for owner, action in normalized.items():
            factory = state.factory(owner)
            if factory is None:
                continue
            if action in BUILD_TYPES or action == Action.BUILD_WALL:
                targets[owner] = first_free_adjacent(state, factory.position, owner)
            elif action == Action.REMOVE_WALL:
                targets[owner] = next((cell for cell in adjacent_cells(state, factory.position, owner)
                                       if cell in state.walls), None)

        proposals: dict[int, tuple[int, int]] = {}
        for owner, action in normalized.items():
            factory = state.factory(owner)
            if factory is not None and action in DELTAS:
                delta = DELTAS[action]
                destination = (factory.position[0] + delta[0], factory.position[1] + delta[1])
                if state.in_bounds(destination) and destination not in state.walls:
                    if state.entity_at(destination) is None:
                        proposals[owner] = destination
        claims = Counter(proposals.values())
        claims.update(target for owner, target in targets.items() if target is not None
                      and (normalized[owner] in BUILD_TYPES or normalized[owner] == Action.BUILD_WALL))
        for owner, destination in proposals.items():
            factory = state.factory(owner)
            assert factory is not None
            # Occupied starting cells cannot be entered, even if vacated now.
            if claims[destination] == 1:
                action = normalized[owner]
                cost = self.config.jump_cost if action in JUMP_ACTIONS else self.config.movement_cost
                origin = factory.position
                factory.position = destination
                self._spend(state, factory, cost)
                state.log("move", owner=owner, entity=factory.id, origin=origin, destination=destination)

        for owner, action in normalized.items():
            if action not in DELTAS and action != Action.WAIT:
                target = targets.get(owner)
                if (action in BUILD_TYPES or action == Action.BUILD_WALL) and claims[target] > 1:
                    state.log("collision", owner=owner, position=target)
                    continue
                self._apply_non_movement(state, owner, action, target)

        self._collect_resources(state)
        self._advance_lifecycle(state)
        for rule in self.extra_rules:
            rule.after_step(state, actions, self.config, rng)
        state.step_count += 1
        self._advance_danger(state)
        self._check_terminal(state)
        return invalid

    def _spend(self, state: GameState, factory: Entity, amount: float) -> None:
        factory.energy -= amount
        state.energy_spent[factory.owner] += amount

    def _apply_non_movement(
        self, state: GameState, owner: int, action: Action, target: tuple[int, int] | None,
    ) -> None:
        factory = state.factory(owner)
        if factory is None:
            return
        if action in BUILD_TYPES:
            position = target
            if position is None:
                return
            entity = make_entity(state.next_entity_id, BUILD_TYPES[action], owner, position)
            state.entities[entity.id] = entity
            state.next_entity_id += 1
            self._spend(state, factory, self.config.build_costs[action])
            factory.cooldown = self.config.build_cooldown + 1
            state.log("build", owner=owner, entity=entity.id, kind=entity.kind.name, position=position)
        elif action == Action.BUILD_WALL:
            position = target
            if position is not None:
                state.walls.add(position)
                self._spend(state, factory, self.config.build_costs[action])
                factory.cooldown = self.config.build_cooldown + 1
                state.log("build_wall", owner=owner, position=position)
        elif action == Action.REMOVE_WALL:
            position = target
            if position is not None:
                state.walls.discard(position)
                state.log("remove_wall", owner=owner, position=position)
        elif action == Action.MINE and factory.position in state.mining_nodes:
            state.gathered[owner] += int(self.config.mining_counts_as_resource)
            factory.energy += 1.0
            state.log("mine", owner=owner, position=factory.position)

    def _collect_resources(self, state: GameState) -> None:
        for entity in state.entities.values():
            if entity.alive and entity.position in state.resources:
                amount = state.resources.pop(entity.position)
                state.gathered[entity.owner] += amount
                if entity.kind == EntityType.FACTORY:
                    entity.energy += amount
                state.log("resource", owner=entity.owner, entity=entity.id, amount=amount)

    def _advance_lifecycle(self, state: GameState) -> None:
        for entity in state.entities.values():
            entity.cooldown = max(0, entity.cooldown - 1)

    def _advance_danger(self, state: GameState) -> None:
        if not self.config.danger_enabled:
            return
        if state.step_count % self.config.boundary_interval == 0:
            state.danger_column += 1
        for entity in state.entities.values():
            if not entity.alive:
                continue
            column = entity.position[1]
            endangered = column <= state.danger_column
            if self.config.opponent_enabled:
                endangered = column <= state.danger_column or column >= state.width - 1 - state.danger_column
            if endangered:
                entity.alive = False
                state.log("danger_loss", owner=entity.owner, entity=entity.id)

    def _check_terminal(self, state: GameState) -> None:
        first = state.factory(0)
        second = state.factory(1)
        if self.config.opponent_enabled:
            if first is None or second is None:
                state.terminated = True
                state.winner = 1 if first is None and second is not None else 0 if second is None and first is not None else None
        elif first is None:
            state.terminated = True
            state.winner = 1
        elif first.position == (0, state.width - 1):
            state.terminated = True
            state.winner = 0

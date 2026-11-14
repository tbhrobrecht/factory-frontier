"""Shared scripts must work with the original minimal student interface."""

import sys

import numpy as np
import pytest

import student.evaluate as evaluation
import student.student_view as viewing
import student.train as training


class MinimalAgent:
    """No Q-table, epsilon, alpha, RNG, reward counter or checkpoint methods."""

    instances = []

    def __init__(self, seed=None):
        self.updates = 0
        self.instances.append(self)

    def reset(self, observation):
        pass

    def act(self, observation):
        return int(np.flatnonzero(observation["action_mask"])[0])

    def update(self, observation, action, reward, next_observation, terminated, truncated):
        self.updates += 1


@pytest.mark.parametrize("config", [
    "intro", "mdp", "dynamic_programming", "monte_carlo_td", "model_free_control",
    "function_approximation", "policy_methods", "ppo",
])
def test_shared_scripts_accept_original_agent_interface(monkeypatch, config, capsys):
    MinimalAgent.instances.clear()
    monkeypatch.setattr(training, "StudentAgent", MinimalAgent)
    summary = training.train(config, episodes=2, eval_episodes=2)
    assert summary.episodes == 2
    assert summary.checkpoint == ""
    assert MinimalAgent.instances[-1].updates > 0

    monkeypatch.setattr(evaluation, "StudentAgent", MinimalAgent)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", config, "--seeds", "0"])
    evaluation.main()
    assert MinimalAgent.instances[-1].updates == 0

    monkeypatch.setattr(viewing, "StudentAgent", MinimalAgent)
    monkeypatch.setattr(sys, "argv", ["student_view", "--config", config, "--render-steps", "0"])
    viewing.main()
    assert MinimalAgent.instances[-1].updates == 0
    assert "Final map:" in capsys.readouterr().out


def test_explicit_checkpoint_request_explains_missing_optional_interface(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "StudentAgent", MinimalAgent)
    with pytest.raises(ValueError, match="StudentAgent.save"):
        training.train(episodes=1, checkpoint=tmp_path / "model")
    for module in (evaluation, viewing):
        monkeypatch.setattr(module, "StudentAgent", MinimalAgent)
        monkeypatch.setattr(sys, "argv", ["runner", "--checkpoint", str(tmp_path / "model")])
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2

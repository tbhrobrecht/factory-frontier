"""Regressions for sampled training and TD bootstrap boundaries."""

import numpy as np
import pytest

from factory_frontier import Action, FactoryFrontier
from factory_frontier.agents.tabular_q_agent import TabularQAgent
from factory_frontier.training import run_training_episode
from student.train import run_episode


@pytest.mark.parametrize("runner", [run_training_episode, run_episode])
@pytest.mark.parametrize("ending", ["goal", "limit"])
def test_training_delivers_final_transition_with_separate_flags(runner, ending):
    env = FactoryFrontier("monte_carlo_td", slip_probability=0.0,
                          max_steps=75 if ending == "goal" else 1)

    class Recorder:
        def reset(self, observation):
            self.flags = []

        def act(self, observation):
            row, _ = np.argwhere(observation["grid"] == 5)[0]
            return Action.WAIT if ending == "limit" else Action.NORTH if row else Action.EAST

        def update(self, observation, action, reward, next_observation, terminated, truncated):
            self.flags.append((terminated, truncated))

    agent = Recorder()
    runner(env, agent, seed=0)
    assert agent.flags[:-1] == [(False, False)] * (len(agent.flags) - 1)
    assert agent.flags[-1] == ((True, False) if ending == "goal" else (False, True))
    assert len(agent.flags) == (10 if ending == "goal" else 1)
    with pytest.raises(RuntimeError, match="hidden"):
        env.transition_model()


@pytest.mark.parametrize("terminated,truncated", [(False, False), (False, True), (True, False)])
def test_td_bootstraps_legal_values_except_on_termination(terminated, truncated):
    env = FactoryFrontier("monte_carlo_td")
    observation, _ = env.reset(seed=0)
    next_observation, reward, _, _, _ = env.step(Action.NORTH)
    agent = TabularQAgent(learning_rate=1.0)
    values = np.full(env.action_space.n, 999.0)
    values[next_observation["action_mask"] != 0] = 2.0
    agent.q[agent.state_key(next_observation)] = values
    agent.update(observation, Action.NORTH, reward, next_observation, terminated, truncated)
    expected = reward if terminated else reward + 0.99 * 2.0
    assert agent.q[agent.state_key(observation)][Action.NORTH] == pytest.approx(expected)

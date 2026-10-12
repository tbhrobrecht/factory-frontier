"""Starter checks for the student submission contract."""

from factory_frontier import FactoryFrontier
from factory_frontier.agents import Agent
from student.agent import StudentAgent


def test_student_agent_returns_a_legal_action() -> None:
    env = FactoryFrontier("intro")
    observation, _ = env.reset(seed=42)
    agent = StudentAgent(seed=42)
    agent.reset(observation)
    action = agent.act(observation)
    assert env.action_space.contains(action)
    assert observation["action_mask"][action] == 1


def test_student_agent_uses_public_interface() -> None:
    assert isinstance(StudentAgent(), Agent)


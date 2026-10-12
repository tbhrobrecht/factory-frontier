from factory_frontier import Action, FactoryFrontier
# from factory_frontier.examples.student_agent import StudentAgent
from factory_frontier.agents import RuleBasedAgent as StudentAgent

env = FactoryFrontier("intro", render_mode="ansi", slip_probability=0.0)
agent = StudentAgent()

observation, info = env.reset(seed=42)
agent.reset(observation)

print("Initial observation fields:")
for name, value in observation.items():
    print(f"  {name}: shape={value.shape}")

done = False

while not done:
    print("\n" + env.render())

    legal_actions = [
        action.name
        for action in Action
        if observation["action_mask"][action]
    ]
    print("Legal actions:", legal_actions)

    action = agent.act(observation)
    print("Chosen action:", Action(action).name)

    observation, reward, terminated, truncated, info = env.step(action)

    print("Reward:", reward)
    print("Reward components:", info["reward_components"])

    done = terminated or truncated

print("\nFinal map:")
print(env.render())
print("Winner:", info["winner"])
print("Episode length:", info["step"])
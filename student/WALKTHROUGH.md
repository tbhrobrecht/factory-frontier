# Factory Frontier student walkthrough

This walkthrough takes you from a fresh checkout to your first working agent.
Complete the sections in order. You only need basic Python knowledge; no
reinforcement-learning implementation is required until the final exercises.

All commands assume that your terminal is open in the repository root—the
directory containing `pyproject.toml`, `factory_frontier/`, and `student/`.

## 1. Set up the project

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Your prompt should begin with `(.venv)`. Confirm that the starter project is
healthy:

```powershell
python -m pytest
```

You should see all tests pass. If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You must reactivate `.venv` whenever you open a new terminal.

## 2. Run the starter agent

```powershell
python -m student.student_view
```

The program prints a grid such as:

```text
. . . . .
. . . . .
. . . . .
. . . . .
F . . . .
```

`F` is your Factory. In the `intro` environment it starts in the lower-left
corner, and its goal is the upper-right corner. The starter agent chooses a
random legal action, so it may fail to reach the goal before the 40-step limit.

The program also prints:

- legal actions for the current state;
- the action selected by the agent;
- the resulting reward;
- the reward-component breakdown;
- the final winner and episode length.

### Practice

Run the same episode again:

```powershell
python -m student.student_view --seed 42
```

Because the environment and starter agent use the same seed, the result should
be reproducible. Then change the seed:

```powershell
python -m student.student_view --seed 7
```

Observe what changes.

## 3. Understand the agent contract

Open `student/agent.py`. Every submitted agent must implement these methods:

```python
def reset(self, observation):
    ...

def act(self, observation):
    return action
```

`reset()` is called at the beginning of an episode. `act()` receives the current
observation and returns one canonical action index.

The starter uses:

```python
legal_actions = np.flatnonzero(observation["action_mask"])
return int(self.rng.choice(legal_actions))
```

The action mask contains one value for every action:

- `1` means currently legal;
- `0` means disabled or illegal in the current state.

An agent should never assume that an enabled action is legal everywhere. A
movement can become illegal at a map boundary, beside a wall, or when another
entity occupies its destination.

## 4. Write your first purposeful policy

Replace the body of `StudentAgent.act()` in `student/agent.py` with:

```python
from factory_frontier import Action

# Keep this import at the top of the file.

def act(self, observation: Observation) -> int:
    mask = observation["action_mask"]

    if mask[Action.NORTH]:
        return int(Action.NORTH)

    if mask[Action.EAST]:
        return int(Action.EAST)

    return int(Action.WAIT)
```

Move `from factory_frontier import Action` to the import section at the top of
the file. Do not place it inside the class.

Run the visualizer again:

```powershell
python -m student.student_view --seed 42 --render-steps 12
```

The Factory should move north to the top row and then east to the goal. Check:

- Did the episode terminate before 40 steps?
- Is `Winner` equal to `0`?
- Is the final total reward positive?

### Practice

Change the preference order to east first and then north. Does it still reach
the goal? How many steps does it take?

## 5. Inspect an observation

Temporarily add these lines to `student/student_view.py` immediately after
`env.reset(...)`:

```python
height, width = observation["map_shape"]

print("Active grid:")
print(observation["grid"][:height, :width])

print("Factory features:")
print(observation["factory"])

print("Action mask:")
for action in Action:
    print(action.name, bool(observation["action_mask"][action]))
```

Run the visualizer and answer:

1. Why are `SOUTH` and `WEST` initially illegal?
2. Which grid value represents the Factory?
3. Why is the complete grid 16×16 while `map_shape` is 5×5?
4. Which observation field should an agent check before returning an action?

Remove the temporary prints when you are finished.

## 6. Evaluate over multiple episodes

A single rendered episode is useful for debugging, but it is not a reliable
evaluation. Run:

```powershell
python -m student.evaluate
```

The evaluation reports:

- return for every seed;
- episode length for every seed;
- number of successful episodes;
- mean return.

Choose explicit seeds with:

```powershell
python -m student.evaluate --config intro --seeds 0 1 2 3 4 5 6 7 8 9
```

### Practice

Compare the random starter policy with your north/east policy. Record:

| Policy | Successes | Mean return | Typical length |
|---|---:|---:|---:|
| Random |  |  |  |
| North/east |  |  |  |

Why is a fixed set of evaluation seeds important when comparing two agents?

## 7. Experience stochastic movement

The `intro` environment is deterministic. The `mdp` environment has a 20%
movement-slip probability: a requested direction sometimes resolves as one of
the other three directions.

Run:

```powershell
python -m student.student_view --config mdp --seed 42 --render-steps 20
```

The visualizer prints the requested action. Compare it with the Factory's actual
movement on the next grid.

### Practice

Run several seeds:

```powershell
python -m student.student_view --config mdp --seed 1 --render-steps 20
python -m student.student_view --config mdp --seed 2 --render-steps 20
python -m student.student_view --config mdp --seed 3 --render-steps 20
```

Answer:

1. Does the same action always produce the same next state?
2. Can a hand-written deterministic policy still succeed?
3. Why might a safe policy differ from the shortest deterministic path?

## 8. Inspect the MDP transition model

Create a temporary file named `student/inspect_mdp.py`:

```python
from factory_frontier import Action, FactoryFrontier


env = FactoryFrontier("mdp")
env.reset(seed=42)
model = env.transition_model()

state = (4, 0)
for probability, next_state, reward, terminal in model[state][int(Action.NORTH)]:
    print(
        f"probability={probability:.3f}, "
        f"next_state={next_state}, reward={reward:.3f}, terminal={terminal}"
    )
```

Run it:

```powershell
python -m student.inspect_mdp
```

The probabilities for an action should sum to 1. Model-free curriculum stages
intentionally do not expose this model.

## 9. Use the training scaffold

Run:

```powershell
python -m student.train --config intro --episodes 100 --seed 0
```

The command saves a summary to:

```text
student/results/training_summary.json
```

Initially, `StudentAgent.update()` is empty. Therefore, the command collects
experience but does not improve the policy. This is intentional: later
assignments will implement Monte Carlo, TD, SARSA, Q-learning, or another update
inside this method or in additional modules.

The transition passed to `update()` is:

```text
(observation, action, reward, next_observation, done)
```

### Practice

Add a temporary counter to `StudentAgent`:

```python
def __init__(self, seed: int | None = None) -> None:
    self.rng = np.random.default_rng(seed)
    self.transitions_seen = 0

def update(self, observation, action, reward, next_observation, done) -> None:
    self.transitions_seen += 1
```

This is not learning yet, but it verifies where experience reaches the agent.

## 10. Run your tests

Run all official and student-facing checks:

```powershell
python -m pytest
```

Run only your workspace tests:

```powershell
python -m pytest student/tests
```

Add new tests under `student/tests/` as your algorithms become more complex.
Never change an engine test merely to make a failing implementation pass.

## 11. Know your workspace boundary

You may create or edit files under:

```text
student/
```

Typical additions include:

```text
student/model.py
student/features.py
student/algorithms/q_learning.py
student/checkpoints/q_table.json
student/tests/test_q_learning.py
```

You may read the official `factory_frontier/` package, but do not modify it.
During grading, your `student/` directory is placed beside a clean official
engine, so coursework must not depend on engine changes.

## Completion checklist

Before moving to the first RL assignment, make sure you can:

- [ ] activate the virtual environment;
- [ ] run `python -m student.student_view`;
- [ ] explain `reset()`, `act()`, and `step()`;
- [ ] select only actions allowed by `action_mask`;
- [ ] write a simple policy that reaches the introductory goal;
- [ ] evaluate an agent over several fixed seeds;
- [ ] explain the difference between deterministic and stochastic movement;
- [ ] run the complete test suite;
- [ ] keep all coursework inside `student/`.

Once these steps are comfortable, you are ready to implement the algorithm
assigned for the next curriculum stage.


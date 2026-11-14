# Getting started

Follow these steps from a complete repository checkout or source archive.
You need Python 3.10+ and basic Python knowledge. Keep your terminal in the
repository root (the directory containing `pyproject.toml`).

## 1. Install and check

In PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

All tests should pass. If `.venv` already exists, reuse it by activating it.
Reactivate it whenever you open a new terminal. If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux, use `python3 -m venv .venv`, then `source .venv/bin/activate`;
run the same install and test commands afterward.

## 2. Run the random starter

```powershell
python -m student.student_view --config intro --seed 42 --render-steps 40
```

`F` starts at the bottom-left of a 5x5 map. Its goal is the top-right cell,
which has no special symbol. The program prints legal actions, selected actions,
rewards and the final result. The unchanged starter with seed 42 finishes at
40 steps with `Winner: None` and approximately `-0.4` reward. Random failure is normal.

Run again with the same seed to reproduce the episode, then try `--seed 7`.

## 3. Make your first successful policy

Open `student/agent.py`. Add this import at the top of the file:

```python
from factory_frontier import Action
```

Replace the existing `act` method inside `StudentAgent` with the following,
keeping its indentation inside the class:

```python
    def act(self, observation: Observation) -> int:
        mask = observation["action_mask"]
        if mask[Action.NORTH]:
            return int(Action.NORTH)
        if mask[Action.EAST]:
            return int(Action.EAST)
        return int(Action.WAIT)
```

Run the same command again. Expect **8 steps**, **`Winner: 0`**, and about
**9.92 total reward**. Your factory moves north, then east. Try reversing those
preferences and compare the path.

`reset(observation)` runs at the start of an episode; `act(observation)` chooses
one integer action. A mask entry of 1 means legal, 0 means unavailable.

## 4. Inspect and evaluate

Temporarily add these lines inside `StudentAgent.reset()` in `student/agent.py`,
indented like the method's existing body:

```python
        height, width = observation["map_shape"]
        print("Full grid shape:", observation["grid"].shape)
        print("Active map shape:", observation["map_shape"])
        print("Active grid:\n", observation["grid"][:height, :width])
        print("Factory:", observation["factory"])
        print("Legal-action mask:", observation["action_mask"])
```

Run the visualizer again. These print the agent's input once at episode start;
they do not change its behavior. The full 16x16 grid is padded so all stages
share a shape; `intro` uses only 5x5. South and west initially leave the map.
Use the [student reference](README.md#agent-interface-and-observations) to decode
the fields, then remove these prints.

Evaluate your policy on fixed seeds:

```powershell
python -m student.evaluate --config intro --seeds 0 1 2 3 4
python -m pytest student/tests
```

The north/east policy should succeed on all five intro episodes. Record return,
successes and episode length when comparing policies.

## 5. Try stochastic movement and training

```powershell
python -m student.student_view --config mdp --seed 42 --render-steps 50
python -m student.train --config intro --episodes 100 --seed 0
```

In `mdp`, ordinary moves have 20% slip: the requested action may produce a
different direction. Compare the printed action with the next map.

Training calls `StudentAgent.update()` after each transition. It is initially
empty, so running training does not improve the policy yet. The script saves
metrics to `student/results/training_summary.json`; implement learning and
optional `save(path)` / `load(path)` methods in your agent. The shared scripts
handle checkpoint paths when those methods exist. Without them, no checkpoint
is saved and later runs create a fresh agent.

## 6. Continue with the course

Use the [curriculum and algorithm reference](README.md#curriculum-and-algorithms)
for the next assigned stage, through PPO. Implement the algorithm in `agent.py`
and add helper modules or your own tests under `student/` as needed. The shared
scripts and official engine do not need algorithm-specific changes. Planning
uses a known model; MC/TD and later methods learn through `update()`.

For fuel tasks, first use `--map-seed 0` to repeat a map. Add checkpoint support
before trying to evaluate a trained policy in a separate process; see the
[training reference](README.md#train-evaluate-and-watch).

Before moving on, check that you can run the tests, reach the intro goal, read
an action mask, evaluate fixed seeds and explain how slip changes movement.

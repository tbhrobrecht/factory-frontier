# Student workspace

This directory is the only place where coursework should be written. You may
read the rest of the repository to understand Factory Frontier, but do not
modify the official `factory_frontier/` engine. During grading, only this
`student/` directory is copied into a clean official environment.

New students should begin with the hands-on
[`WALKTHROUGH.md`](WALKTHROUGH.md) and complete its checklist before starting
the first reinforcement-learning assignment.

For map symbols, entities, danger, actions, and the factory's objectives, see
the [game guide and legend](GAME_README.md).

## Setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Start here

```powershell
python -m student.student_view
python -m student.evaluate
python -m pytest student/tests
```

The starter `StudentAgent` chooses a random legal action. Begin by editing
`student/agent.py`. Its public contract is always:

```python
agent.reset(observation)
action = agent.act(observation)
```

Use `observation["action_mask"]` so the agent selects legal actions. The three
most important introductory fields are `grid`, `map_shape`, and `action_mask`.

## Files you own

- `agent.py`: policy and learning logic.
- `student_view.py`: visual debugging and exploration.
- `train.py`: training loop scaffold.
- `evaluate.py`: repeatable evaluation on explicit seeds.
- `tests/`: your own tests.
- `checkpoints/`: saved tables or model weights. Small checkpoints may be
  committed; common large neural-weight formats are ignored and should use the
  course artifact submission channel.
- `results/`: metrics, logs, and plots; generated files are ignored.

You may add modules such as `model.py`, `features.py`, or an `algorithms/`
package as assignments become more advanced.

## Training scaffold

```powershell
python -m student.train --config intro --episodes 100
```

The starter `update()` is deliberately empty, so this initially measures a
random baseline rather than learning. Implement the assigned algorithm inside
`StudentAgent` or additional modules, then save checkpoints under
`student/checkpoints/`.

Both training loops call
`update(observation, action, reward, next_observation, terminated, truncated)`.
Use `terminated or truncated`
to finish an episode buffer. For TD learning, suppress the next-state value
only on `terminated`; a time limit still supplies a next observation to bootstrap
from. For Monte Carlo learning, accumulate transitions until either ending flag
and compute returns backward before the next `reset()`. A truncated trajectory
is a partial episode: collect complete episodes for pure episodic MC, or explicitly
choose a tail-value estimate or a finite-horizon objective for truncated data.

## Course progression

For `function_approximation`, start with
`from factory_frontier.wrappers import encode_features, mask_action_values`.
`encode_features(observation)` returns a 4165-element float32 vector with
categorical grid channels and the numerical state features. Use it for a
linear value function or as input to your own neural network; alternatives
such as local features or spatial encoders are encouraged. The root README
documents the feature layout and a linear-model example.

Apply `mask_action_values(values, observation["action_mask"])` before choosing
an action, and mask next-action values in TD targets too. For epsilon-greedy
exploration, sample from the nonzero action-mask entries. A terminal or empty
next-action mask has zero bootstrap value. Keep tensor-based masking inside
your chosen framework when computing gradients.

For `policy_methods`, use `masked_policy_probabilities(logits, action_mask)` from
the same module, then sample an action with your agent's seeded NumPy generator.
The helper applies softmax only over legal actions; it accepts an optional
positive temperature. Keep the selected action and original mask in your
trajectory. Environmental slip changes the movement outcome, not the action
whose log-probability belongs in the policy loss. `ModelAdapter` greedily selects
vector outputs, so sample explicitly in `StudentAgent.act()` for stochastic
policies. Implement differentiable log-probabilities and entropy in your chosen
framework; the NumPy helper is for action sampling and experimentation. See the
root README for a runnable example and episode/actor-critic guidance.

The same student interface works with every curriculum preset:

```powershell
python -m student.student_view --config mdp
python -m student.evaluate --config monte_carlo_td
```

Do not change canonical action indices or engine observation definitions.

# Student reference

Start with the [walkthrough](WALKTHROUGH.md) for installation and your first agent.
This file is the reference for game rules, observations and learning algorithms.
All commands run from the repository root. The course ends at `ppo`; competition
is outside this release's coursework.

## Your workspace

Implement algorithms in `student/agent.py`; add helper modules and tests under
`student/` as needed. You may read the engine but must not modify it. The shared
training, evaluation and visualization scripts work through the same agent interface.

| File or directory | Purpose |
|---|---|
| `agent.py` | `StudentAgent`: policy, episode memory and learning |
| `student_view.py` | Print an episode for debugging |
| `train.py`, `evaluate.py` | Collect training experience; evaluate on fixed seeds |
| `tests/` | Your tests |
| `checkpoints/` | Saved tables, models and feature scaling |
| `results/` | Generated metrics and plots (ignored by Git) |

Add modules as needed. Submit an importable `StudentAgent` and its required
checkpoints inside `student/`. Grading uses a clean engine. Common neural-weight
formats are ignored by Git; submit those through the course artifact channel.

## Game rules

Your agent controls a moving factory and chooses one action per turn.
Coordinates are `(row, column)`, starting at the top-left `(0, 0)`.

- **Goal:** start at the bottom-left and reach `(0, width - 1)` to win.
  Danger destroying your factory ends the episode in defeat. The time limit
  ends an unfinished episode through `truncated`.
- **Fuel:** a factory collecting a resource gains 1 energy and 1 resource point.
  `MINE` on a renewable node adds 1 energy and 1 resource point. Zero energy
  prevents unaffordable actions; it does not destroy the factory.
- **Movement:** walls and occupied cells block landing. Jumps travel two cells
  and can cross walls or units. Ordinary moves may slip into another direction;
  blocked realized moves stay in place. Jumps, mining and waiting do not slip.
- **Danger:** starts outside the map and advances by one column at its interval.
  It moves from left to right. Covered columns destroy entities every turn,
  including on entry. Danger is
  resolved before goal detection. Walls offer no protection.
- **Visibility:** PPO shows cells within Manhattan distance 3; the other course
  presets are fully observed. Walls do not block visibility. The ASCII display
  shows the full map for debugging; policies must use their observations.

### Map and action legend

| Printed symbol | Meaning |
|---|---|
| `F` | Your factory |
| `S`, `W`, `M` | Scout, worker, miner (custom configurations only) |
| `#` | Wall |
| `*` | Collectible resource |
| `n` | Renewable mining node |
| `.` | Empty ground |

Entities hide terrain underneath them; resources hide mining-node symbols.
The goal and danger have no printed symbol. Read the goal coordinates and
`danger` features. Destroyed entities disappear.

Always select an action whose `action_mask` entry is `1`. An illegal action
becomes `WAIT` and still consumes a turn. Action indices stay fixed across presets.

| Index | Action | Effect |
|---|---|---|
| 0 | `WAIT` | Stay; time and cooldowns advance; no energy regeneration |
| 1-4 | `NORTH`, `SOUTH`, `EAST`, `WEST` | Move one cell |
| 5-8 | `JUMP_NORTH`, `JUMP_SOUTH`, `JUMP_EAST`, `JUMP_WEST` | Jump two cells |
| 9-11 | `BUILD_SCOUT`, `BUILD_WORKER`, `BUILD_MINER` | Build for 4, 5, 6 energy respectively |
| 12 | `BUILD_WALL` | Build for 2 energy |
| 13 | `REMOVE_WALL` | Remove an adjacent wall for free |
| 14 | `MINE` | Refuel on a mining node |

Building/removal are disabled in all course presets. In custom configurations,
builds pick the first free neighbor: east, south, west, north. Wall removal uses
the same order among walls. Default build cooldown requires two intervening turns. Built units are
stationary: they collect resources on their cell but do not move, extend
visibility or automatically mine. There is no combat.

### Rewards

| Presets | Per turn | Per resource point | Win | Loss |
|---|---:|---:|---:|---:|
| `intro` through `monte_carlo_td` | -0.01 | +1 | +10 | -10 |
| `model_free_control` through `ppo` | -0.1 | +0.05 | +10 | -10 |

Mining has net reward -0.05 in the fuel tasks: it replenishes energy but still
costs a turn. Rewards are undiscounted; apply your chosen discount in the algorithm.

## Agent interface and observations

Keep the `StudentAgent(seed=None)` constructor and the existing methods:
`reset(observation)` prepares episode memory, `act(observation)` returns an
integer action, and `update(...)` learns from experience. Preserve learned
parameters across resets. Training calls:

```python
update(observation, action, reward, next_observation, terminated, truncated)
```

The environment uses Gymnasium's signatures:

```python
observation, info = env.reset(seed=42)
next_observation, reward, terminated, truncated, info = env.step(action)
```

Stop on `terminated or truncated`, then reset before stepping again.

| Observation key | Default shape | Contents |
|---|---|---|
| `grid` | `(16, 16)` | 0 empty/unseen/padding; 1 wall; 2 resource; 3 mining node; 5-8 friendly factory/scout/worker/miner; 9-12 opponents |
| `visibility` | `(16, 16)` | 1 for observed cells; use with `grid` |
| `map_shape` | `(2,)` | Active height, width inside padding |
| `mining_nodes` | `(16, 16)` | Visible nodes, including beneath entities/resources |
| `action_mask` | `(15,)` | 1 for legal actions |
| `factory`, `opponent` | `(8,)` each | Factory features; `opponent` is reserved and zero in course presets |
| `entities`, `entity_mask` | `(32, 8)`, `(32,)` | Visible entity features and valid rows |
| `danger` | `(4,)` | Enabled, boundary/width, distance to your side's boundary/width, interval/100 |

Factory features are `[alive, row/(height-1), column/(width-1), energy/100,
health/3, cooldown/10, gathered/100, step/max_steps]`; position denominators
are at least 1. Entity rows are `[kind/4, friendly, normalized row, normalized
column, energy/100, health/3, cooldown/10, 1]`.
Coordinates and action directions are absolute.
Use observations as policy inputs; `env.state` and full event logs are debug tools.

## Curriculum and algorithms

These are environment stages, not ready-made algorithm implementations.
Implement the algorithm assigned in your course inside `student/`.

| Preset | Map | Slip | Step limit | Main exercise |
|---|---|---:|---:|---|
| `intro` | 5x5 empty | 0% | 40 | Legal actions and a hand-written goal policy |
| `mdp` | 5x5 empty | 20% | 50 | States, rewards and transition probabilities |
| `dynamic_programming` | 5x5 empty | 10% | 50 | Policy evaluation, policy iteration, value iteration |
| `monte_carlo_td` | 6x6 empty | 10% | 75 | Monte Carlo returns and TD prediction |
| `model_free_control` | 8x8 random walls | 0% | 40 | SARSA/Q-learning, exploration and state design |
| `function_approximation` | 12x12 random walls | 0% | 60 | Linear or neural value functions |
| `policy_methods` | 12x12 random walls | 8% | 72 | REINFORCE, baselines, actor-critic |
| `ppo` | 14x14 rooms, local view | 5% | 112 | PPO rollouts, advantages and clipped updates |

Fuel-task defaults:

| Preset | Initial energy | Collectibles | Danger advances every | Early mining stop |
|---|---:|---:|---:|---|
| `model_free_control` | 8 | 5 | 5 turns | `(7, 3)` |
| `function_approximation` | 16 | 14 | 5 turns | `(11, 3)` |
| `policy_methods` | 16 | 14 | 6 turns | `(11, 3)` |
| `ppo` | 20 | 20 | 8 turns | `(13, 3)` |

Ordinary movement costs 1 in fuel tasks and 0 in earlier stages. Jumps cost 2
and are enabled from `function_approximation` onward; mining starts at
`model_free_control`. Single-player fuel maps protect a route via the mining
stop; refuel before danger arrives. Stochastic slips can still cause failure.
At least one mining action is needed in default `model_free_control`.

### MDP and dynamic programming

After `reset()`, call `env.transition_model()` in `mdp` or
`dynamic_programming`. It maps `(row, column)` and action to outcomes
`(probability, next_state, reward, terminal)`. Later stages hide the model.
Invalid actions are waits without slip. Legal moves keep their direction with
probability `1 - slip`; each other direction gets `slip / 3`.

Choosing `--config mdp` selects the environment, not a planning algorithm.
Your agent must obtain the model from a reset environment with matching settings
and implement planning itself. The shared scripts pass observations to the agent;
they do not automatically supply a model or change the agent's algorithm.

The planning problem is an **infinite-horizon discounted MDP**, with
`gamma = env.config.gamma` (default `0.99`, configurable via
`FactoryFrontier("mdp", gamma=0.95)`, valid range `0 <= gamma < 1`). Maximize
the expected sum of `gamma**t * reward_t`. Apply discounting in your planner;
the engine and model return undiscounted rewards.

Use `probability * (reward + gamma * V[next_state])` in Bellman updates,
with zero continuation on terminal outcomes. `WAIT` is a deterministic
self-transition. Blocked slip outcomes stay in place and their probabilities
are added. A normal step pays `-0.01`; first entering the goal pays `+9.99`.
The goal is terminal and absorbing in the model: every action returns
`(1.0, goal, 0.0, True)`, so its value is zero. The simulator requires a reset
after an episode ends and cannot pay the goal reward again.

The **50-step simulation cutoff** reports `truncated=True`, not MDP termination,
and is excluded from the model. Do not treat it as a terminal state in discounted
planning or add remaining steps to the planning state for this exercise.
Position is sufficient for the underlying MDP: transition/reward distributions
do not depend on episode history. The observation also carries a simulation
step counter, which does not change these dynamics. Model queries do not advance
the episode or RNG. Custom position models require movement-only rules and
stationary position/action rewards. Implement Value Iteration and Policy
Iteration yourself; the repository supplies the model, not their solutions.

### Monte Carlo, TD and control

The starter `update()` is empty. For MC, buffer a trajectory and compute returns
backward. Truncation gives a partial episode: choose a tail-value estimate or
finite-horizon objective explicitly. For TD, bootstrap unless `terminated` is
true. Finish episode buffers on either ending flag.

For SARSA/Q-learning, explore only legal actions and mask next-action values.
An empty next-action mask has zero continuation. The reference `TabularQAgent`
uses exact observation keys; design abstractions and compare fixed training
seeds before expecting generalization to unseen maps.

### Function approximation and policy methods

`factory_frontier.wrappers` provides:

- `encode_features(observation)`: independent float32 vector of length 4165,
  including one-hot grid categories, visibility, mining nodes, masked entities,
  entity masks, factory/opponent/danger, scaled map shape and action mask.
- `mask_action_values(values, action_mask)`: illegal Q-values/logits become `-inf`.
- `masked_policy_probabilities(logits, action_mask)`: stable masked softmax;
  sample with your agent's RNG. It rejects empty masks.

Train linear/neural models or choose your own features. NumPy helpers do not
carry gradients: implement analytic gradients for NumPy models, or differentiable
masking, log-probabilities and entropy inside your tensor framework. Ignore zero-probability actions in entropy.
Store the selected action and original mask, even if the environment slips.
`ModelAdapter` accepts scalar actions or full action vectors and selects greedily;
it does not sample stochastic policies or manage recurrent memory.

### PPO

Store observations, masks, selected actions, old log-probabilities, value
predictions, rewards, both ending flags and actual next observations. Optimize
the clipped policy objective, value loss and entropy in your framework. Reuse
stored masks and keep old log-probabilities fixed during an update.

Bootstrap next-state values unless terminated. Stop advantage recursion at
either ending flag. At rollout cutoffs, bootstrap without resetting a live
environment; at truncation, use the final observation before resetting. Reset
recurrent episode memory at episode boundaries and keep training sequences
ordered. Feed-forward policies are valid; memory is an optional experiment.
Buffering agents may implement `finish_training()` to optimize their final partial
rollout. The trainer calls it before final evaluation and checkpoint saving.

## Train, evaluate and watch

| Script | What it does |
|---|---|
| `student.train` | Creates an agent, evaluates a baseline, trains with `update()`, evaluates again and writes a summary |
| `student.evaluate` | Evaluates an agent without calling `update()` |
| `student.student_view` | Displays one episode without calling `update()` |

Start by learning one fixed map; replace the preset as you progress:

```powershell
python -m student.train --config model_free_control --map-seed 0 --episodes 3000 --seed 0
```

`--map-seed` repeats the environment seed. Without it, `model_free_control`
repeats `--seed` by default; other presets vary seeds. Use `--vary-maps` to
explicitly train across maps. Optional `--alpha` / `--epsilon` overrides require
attributes of those names on your agent; otherwise set parameters in `agent.py`.
The starter's empty `update()` collects experience without learning.

Training writes `student/results/training_summary.json`. If your agent implements
`save(path)`, it also saves `student/checkpoints/<preset>.checkpoint`; override the
path with `--checkpoint`. Implement `load(path)` as a class method returning a
ready-to-use agent. Save the encoder/scaling with its parameters. These methods
are optional; the starter has neither.

After implementing save/load and training, evaluate or watch the checkpoint:

```powershell
python -m student.evaluate --config model_free_control --seeds 0 --checkpoint student/checkpoints/model_free_control.checkpoint
python -m student.student_view --config model_free_control --seed 0 --checkpoint student/checkpoints/model_free_control.checkpoint --render-steps 40
```

Without `--checkpoint`, both commands create a fresh agent; they do not load
previous training automatically. Use checkpoints from the same algorithm and
feature encoder. Evaluation disables epsilon exploration when supported;
softmax policies still sample from their distributions.

Compare returns, success rate and episode length. Reported returns are
**undiscounted**. Same-map wins demonstrate learning on that map, not
generalization: test separate seeds such as `100 101 102` and several agent seeds
with `--agent-seed`. Value prediction alone does not improve a policy; control
algorithms must also use learned values or policy parameters to choose actions.

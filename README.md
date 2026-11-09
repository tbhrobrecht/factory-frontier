# Factory Frontier

Factory Frontier is a grid-based, Gymnasium-compatible reinforcement-learning teaching environment. A single rules engine supports a semester-long progression from a deterministic introductory task to reproducible two-agent tournaments. The canonical action indices and observation keys/shapes do not change between stages, so student code and checkpoints can move forward with the curriculum.

## Install and try it

Students should download or clone the complete repository, or unpack its source
distribution, and run the commands below from that directory. These include the
editable `student/` workspace and course guides. The wheel installs the engine
and reference examples; use the complete source download for coursework.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Student workspace

All coursework belongs in the top-level `student/` directory. Students may
read the complete environment source, but should not modify the official
`factory_frontier/` package. During grading, only `student/` should be copied
into a clean official environment.

Start with:

```powershell
python -m student.student_view
python -m student.evaluate
python -m pytest student/tests
```

Begin with [`student/WALKTHROUGH.md`](student/WALKTHROUGH.md), then use
[`student/README.md`](student/README.md) as the workspace reference.

```python
from factory_frontier import FactoryFrontier
from factory_frontier.agents import RandomAgent

env = FactoryFrontier(config="model_free_control")
agent = RandomAgent(seed=7)
observation, info = env.reset(seed=42)
agent.reset(observation)

done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(agent.act(observation))
    done = terminated or truncated

print(env.render())
```

The standard Gymnasium return signatures are used: `reset()` returns `(observation, info)` and `step()` returns `(observation, reward, terminated, truncated, info)`.
After importing `factory_frontier`, registry-based construction is also available as `gymnasium.make("FactoryFrontier-v0", config="intro")`.

## Stable student interface

`Action` is one append-only `IntEnum`: `WAIT`, four moves, four jumps, three unit builds, wall build/removal, and `MINE`. Every environment exposes `Discrete(15)`; unavailable or currently illegal actions are zeroed in `observation["action_mask"]`.

Every preset returns the same structured, padded observation:

- `grid` — `(16, 16)` spatial codes
- `entities` and `entity_mask` — `(32, 8)` entity features and valid rows
- `factory`, `opponent`, and `danger` — fixed feature vectors
- `visibility` — distinguishes unseen cells from empty cells
- `action_mask` — one value per canonical action
- `map_shape` — active height and width inside the padded grid
- `mining_nodes` — `(16, 16)` visible mining locations, including beneath resources or entities

Advanced mechanics are represented by zeros when absent. New actions must be appended, never inserted or reordered; new observation data should fill reserved/default-compatible fields or be added without changing existing fields.

Implementing an agent only requires:

```python
from factory_frontier.agents import Agent

class MyAgent(Agent):
    def reset(self, observation):
        pass

    def act(self, observation):
        legal = observation["action_mask"].nonzero()[0]
        return int(legal[0])
```

The active starter files are in `student/`. Additional reference examples remain
under `factory_frontier/examples/`. `ModelAdapter` wraps callable models or
objects with `predict`, accepts scalar actions or full canonical logits/Q-values,
and masks illegal outputs. `TabularQAgent.save()` / `.load()` demonstrates a
transparent checkpoint format without supplying an advanced assignment solution.

## Curriculum

All presets use `FactoryFrontier`; only configuration changes.

| Preset | Main additions |
|---|---|
| `intro` | deterministic 5×5 movement and goal |
| `mdp` | stochastic motion and exposed transition model |
| `dynamic_programming` | finite model for policy/value iteration |
| `monte_carlo_td` | sampled trajectories; model hidden |
| `model_free_control` | random map, energy, resources, danger |
| `function_approximation` | larger solvable layouts, fuel management, and jumps |
| `policy_methods` | stochastic navigation, fuel management, and masked policies |
| `ppo` | connected room layouts, local observations, and masked policy experiments |
| `competition` | mirrored survival/resource contests with simultaneous turns and terminal scoring |

Model-based stages expose `env.transition_model()`, a mapping from state and action to `(probability, next_state, reward, terminal)` outcomes. Later stages intentionally reject access to it.

For `mdp` and `dynamic_programming`, states are `(row, column)` positions.
Call `reset()` before requesting the model. Every canonical action index is
included: unavailable actions and moves into walls or outside the map behave
as `WAIT`, without slipping. Legal movement keeps its requested direction with
probability `1 - slip_probability`, or takes each other direction with
probability `slip_probability / 3`; a blocked realized direction stays in place.
Use `action_mask` when selecting legal actions in the running environment.

The top-right goal is an absorbing model state with zero subsequent reward.
Entering it earns the configured win reward once, plus the usual step reward.
The model describes stationary dynamics; its terminal flag does not include
the external time limit (`50` steps for these presets). The live environment
reports that limit through `truncated`, separately from `terminated`. For a
finite-horizon planning exercise, track remaining steps in the planner. After
either episode-ending flag, call `reset()` before stepping again.

Model queries leave episode state and random draws unchanged and work after an
episode ends. Position-only models reject overrides adding resources, danger,
opponents, energy costs, advanced actions, or rule hooks. Custom reward
components must also be stationary functions of position/action rather than
episode history for this state representation to be valid.

### Model-free control exercises

`model_free_control` is an 8×8 fully observable navigation-and-refuelling task.
Each seeded map has random walls and five resources, plus a protected route to
the goal. A renewable mining node at `(7, 3)` is reachable in three east moves;
a randomized wall-free monotone route leads onward from that stop to `(0, 7)`.
The default task is solvable with mining before danger arrives. Other routes
and resource stops remain available for students to explore.

The factory starts with 8 energy and spends 1 per successful move. A collectible
or a mining action supplies 1 energy. Each step costs `-0.1` reward, each gathered
resource earns `+0.05`, a win earns `+10`, and a loss costs `-10`. Mining therefore
has net reward `-0.05`: it is useful for fuel, not for farming return. Danger
advances every five steps and destroys entities in covered columns every turn,
including on entry. The 40-step cap matches the time when danger covers the map;
destruction takes precedence over truncation or reaching the goal that turn.

The observation exposes energy, time, danger, remaining resources, mining nodes,
and legal actions. The reference Q-agent keys the full observation to avoid
aliasing different maps or resource states. It is an exact-table baseline;
students should investigate state abstractions and fixed training seeds before
expecting efficient tabular learning or generalization to unseen layouts.
Builds and jumps are disabled, and `transition_model()` remains unavailable.

```powershell
python -m student.student_view --config model_free_control
python -m student.train --config model_free_control --episodes 100
```

### Function approximation exercises

`function_approximation` extends navigation and refuelling to fully observable
12×12 maps with 14 collectibles and random walls. The factory starts with 16
energy; moves cost 1 and two-cell jumps cost 2. Jumping saves a turn and can cross
a wall if the landing cell is free. A mining stop at `(11, 3)` and a randomized
wall-free route onward guarantee a solution for the default configuration, even
using ordinary moves. Resources, mining and route choice provide fuel options.
Danger advances every five turns, covering the entire map at the 60-step cap.
Rewards match `model_free_control`, including the net `-0.05` cost of mining.
Building and wall removal are disabled in this stage; their canonical action
indices are retained. Model access is hidden.

Use the NumPy helpers in `factory_frontier.wrappers` as a starting point:

```python
import numpy as np
from factory_frontier import FactoryFrontier
from factory_frontier.wrappers import encode_features, mask_action_values

env = FactoryFrontier("function_approximation")
observation, _ = env.reset(seed=42)
features = encode_features(observation)  # float32, shape (4165,)
weights = np.zeros((env.action_space.n, features.size), dtype=np.float32)
q_values = weights @ features
action = int(np.argmax(mask_action_values(q_values, observation["action_mask"])))
```

The encoder one-hot encodes the 13 grid categories and zeros unseen/padded cells.
It includes visibility, mining nodes, masked entity features, entity masks,
factory/opponent/danger features, map dimensions scaled by padded grid size, and
the action mask, in that order. Existing scalar feature scaling is preserved.
The encoder uses no learned statistics and returns an independent array. It
works with `ModelAdapter(model, preprocess=encode_features)` for inference.
Linear models can start with the example above; neural models can use the same
vector or a student-designed spatial encoder. Training remains student work.

Mask action values during greedy/exploratory action selection and next-state
maximization. Use zero continuation on `terminated` or an empty next-action mask;
otherwise bootstrap from legal next actions, including at external truncation.
The NumPy mask helper applies to Q-values or logits, before softmax if needed;
tensor-based training should apply the equivalent mask inside its framework.
Use fixed training seeds for debugging and separate held-out seeds for evaluating
generalization. Students can vary features, architectures, exploration and
discounts without changing the engine. Overrides to map size, fuel, rewards or
danger require checking feasibility again; the default guarantee does not cover
arbitrary combinations.

```powershell
python -m student.student_view --config function_approximation
python -m student.train --config function_approximation --episodes 100
python -m student.evaluate --config function_approximation --seeds 100 101 102
```

### Policy methods exercises

`policy_methods` uses the same 12×12 protected-route layouts, 16 starting energy,
move/jump costs, 14 collectibles, and rewards as `function_approximation`. Ordinary
moves slip with probability `0.08`, split equally among the other three cardinal
directions; a blocked realized direction stays in place. Jumps, mining and waiting
do not slip. Danger advances every six steps, covering the map at the 72-step
cap, allowing more recovery time for stochastic movement. Building and wall
removal remain disabled. The world is fully observed but its transition model
is hidden. Protected routes ensure feasible layouts, not guaranteed success
under every possible sequence of slips.

Use a stochastic policy to experiment with REINFORCE, baselines or actor-critic
methods. For example, this samples from a uniform legal policy:

```python
import numpy as np
from factory_frontier import FactoryFrontier
from factory_frontier.wrappers import masked_policy_probabilities

env = FactoryFrontier("policy_methods")
observation, _ = env.reset(seed=42)
rng = np.random.default_rng(7)  # Agent randomness is independent of environment randomness.
logits = np.zeros(env.action_space.n)  # Replace with your model's output.
probabilities = masked_policy_probabilities(logits, observation["action_mask"])
action = int(rng.choice(env.action_space.n, p=probabilities))
log_probability = float(np.log(probabilities[action]))
next_observation, reward, terminated, truncated, info = env.step(action)
```

The helper masks illegal logits before a numerically stable softmax and accepts
an optional positive `temperature` (default `1.0`). Higher temperatures flatten
the distribution. It rejects an empty action mask; stop or reset instead of
sampling after termination. `encode_features` supplies the same 4165-value
input as in the preceding stage. `ModelAdapter` takes argmax for vector outputs;
raw logits passed there do not produce stochastic sampling. Sample explicitly
inside your agent's `act()` method for policy experiments.

Save the selected action, its original observation/action mask, reward and ending
flags in your trajectory. Learn the probability of the action you selected,
not the direction realized after environmental slip. REINFORCE can accumulate
complete episodes and calculate returns at termination; actor-critic methods
can update each step and bootstrap state values at truncation, but not true
termination. For tensor-based training, compute log-probabilities and entropy
from the same masked distribution inside your framework so gradients are
preserved; this NumPy helper does not build a gradient graph. Entropy calculations
must ignore zero-probability entries. Implement the algorithm and checkpointing
in `student/`, and compare multiple environment and agent seeds.

```powershell
python -m student.student_view --config policy_methods
python -m student.train --config policy_methods --episodes 100
python -m student.evaluate --config policy_methods --seeds 100 101 102 --agent-seed 7
```

### PPO exercises

`ppo` is a 14×14 partially observable navigation/refuelling task with four rooms.
Two partition walls have randomized doorways; additional interior obstacles
fill the requested wall-density target when the partitions do not already exceed
it. Door approaches, an early mining stop at `(13, 3)`, and a monotone route to
`(0, 13)` are protected. The room generator requires dimensions of at least 8×8.
Default layouts have a feasible ordinary-movement route; slip and poor decisions
can still cause failure. Jumps can cross a partition if their landing cell is free.

The factory starts with 20 energy; moves cost 1, jumps cost 2. There are 20
collectibles, 5% slip on ordinary moves, and the same balanced rewards as
`policy_methods`. Danger advances every eight turns and covers the map at the
112-step cap. Building and wall removal are disabled. The goal and loss rules
are unchanged, and the transition model is hidden.

Only cells within Manhattan distance 3 of the factory are visible. Walls do not
block this radius-based visibility. The grid, mining-node layer and entity
features hide off-screen contents; `visibility` distinguishes unseen/padded
cells from observed empty ground. Action masks describe current legal choices;
two-cell jump destinations are inside the visible radius. The full-map ASCII
renderer and `env.state` are debugging tools, not agent inputs.

`encode_features` and `masked_policy_probabilities` work with this preset without
changing dimensions. Begin with a feed-forward policy; compare it with a history
of observations/actions or a recurrent policy to investigate memory. PPO does
not require a recurrent architecture. Manage recurrent hidden state in your
agent; the generic `ModelAdapter` does not manage it. For a controlled comparison, use
`FactoryFrontier("ppo", partial_observability=False)` with the same map seeds.

Implement PPO inside the student workspace. When collecting rollouts, keep each
observation, its original action mask, selected action, old log-probability,
value prediction, reward, both ending flags, and the actual next observation.
Recompute new log-probabilities under the same stored mask during optimization;
do not replace old log-probabilities after an update, and do not use the direction
realized after slip as the selected action. Tensor computations for the clipped
objective, value loss and entropy must stay inside the learning framework.

For advantage estimates, bootstrap the next-state value unless `terminated` is
true. Stop advantage recursion across either termination or truncation so it
never crosses into another episode. At a rollout-length cutoff, bootstrap the
last next observation without resetting a still-active environment. At external
time-limit truncation, use the final next observation before resetting. Reset
recurrent episode memory on either episode-ending flag; training rollout buffers
may span episodes but must retain those boundaries. Keep recurrent sequences
ordered if using memory. Save any feature scaling and model state in checkpoints.
The generic training loop supplies individual transitions; students implement
rollout batching, advantage calculation and PPO optimization themselves.

```powershell
python -m student.student_view --config ppo
python -m student.train --config ppo --episodes 100
python -m student.evaluate --config ppo --seeds 100 101 102 --agent-seed 7
```

### Dynamic programming exercises

Use `FactoryFrontier("dynamic_programming")` for a fully observable 5×5
position-state MDP with 10% movement slip and an exposed transition model.
The preset supports iterative policy evaluation, policy iteration, and value
iteration using the same model. For each outcome, the Bellman contribution is
`probability * (reward + discount * V[next_state])`, with the continuation term
set to zero when `terminal` is true. Set the goal value to zero; its reward is
earned on entry. The discount and convergence tolerance belong to the planner,
not the environment configuration.

The model includes invalid actions as waits. When executing a policy, choose
an action allowed by the current observation's `action_mask`. Read the factory
position from the cell coded `5` in `grid`. Reset between episodes, and replan
if you change the map or configuration. A stationary policy plans beyond the
external 50-step limit; to optimize specifically for that deadline, use backward
induction with remaining steps as an additional planning state and zero value
at zero remaining steps. Planning algorithms remain student exercises.

Preview the default environment with:

```powershell
python -m student.student_view --config dynamic_programming
```

## Instructor extension guide

Create a variation without changing the engine:

```python
from factory_frontier.configs import get_config
from factory_frontier import Action, FactoryFrontier

config = get_config("ppo").with_overrides(
    name="lecture_10",
    width=10,
    height=10,
    enabled_actions=(Action.WAIT, Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.MINE),
    reward_weights={"time": -0.02, "resource": 2.0, "progress": 0.1, "win": 10.0},
)
env = FactoryFrontier(config)
```

Configurations validate numeric ranges, action indices, build costs and generator
dimensions when constructed or overridden. Generator names are `empty`, `random`,
`control`, `rooms` and `symmetric`. Valid custom settings still require checking
task feasibility; validation does not guarantee that an arbitrary task is solvable.

Extension points are intentionally small:

- Add entity data/types in `core/entities`; behavior remains in rules.
- Add map generators through `core/map_generation`, add their accepted names to configuration validation, and select them in a config.
- Add a reward callable implementing `RewardComponent` and pass it through `reward_components=`. Treat its before/after game states as read-only; the frozen context does not freeze those states. Implement state changes through `RuleHook`.
- Add mechanics with `RuleHook.before_step` / `after_step` and pass them through `rules=`; state mutation remains in rules rather than agents.
- Extend encoding in `core/observations.py`, retaining existing field meanings and shapes.
- Append canonical actions in `core/actions.py`, then update masks/rules while preserving old indices.

`GameState` is the sole episode state container. Maps, rewards, observation encoding, rendering, and agents do not own hidden game state. A seed passed to `reset` drives map generation and transition stochasticity.

## Training, evaluation, and competition

`training.run_training_episode` is a minimal algorithm-neutral learning loop. `evaluation.evaluate_agent` reports returns, episode lengths, and successes over explicit seeds.

The `monte_carlo_td` preset is a fully observable 6×6 movement task with 10%
slip, no exposed transition model, and a 75-step external time limit. Training
updates receive `(observation, action, reward, next_observation, terminated,
truncated)` so students can implement episode-based MC or step-based TD without
losing the reason an episode stopped. Both ending flags stop the rollout;
only true termination forces a zero bootstrap value for the underlying task.
See `student/README.md` for buffer handling. Learning algorithms remain student
exercises.

```python
from factory_frontier.agents import RandomAgent, SurvivalAgent
from factory_frontier.competition import round_robin

leaderboard, matches = round_robin(
    {"random": lambda: RandomAgent(seed=1), "survival": SurvivalAgent},
    seeds=(10, 11, 12),
)
```

Tournament pairs play both seat assignments on every seed. Factories create
fresh agents for each match; load trained checkpoints in those factories when
evaluating learned policies. Agent instances are also accepted, but their
`reset()` must restore episode/RNG state for repeatable evaluation. The supplied
seeded `RandomAgent` does this. Both seats require separate instances, including
for policies that deliberately share model weights.

### Competition rules and training

Competition is a survival/resource contest on a 16×16 map, with no combat or
unit building. Walls, 24 collectibles and mining nodes are mirrored by 180-degree
rotation. Protected escape lanes and clear central columns connect both spawn
corners; every collectible is placed in their shared reachable component.
Mining stops at `(15, 3)` and `(0, 12)` are paired. The symmetric generator
requires even dimensions (width at least 8, height at least 4) and an even resource
count; wall counts round down to whole mirrored pairs.

Factories start with 16 energy. Moves cost 1, jumps cost 2, and ordinary moves
have 5% slip. Mining adds 1 energy but **no resource points**. Collecting a finite
resource adds 1 energy and 1 resource point. Danger advances from both sides
every eight turns. At the 60-turn deadline, central columns 7 and 8 remain safe.
One surviving factory wins immediately; simultaneous factory loss is a draw.
If both survive to the deadline, compare `(alive, collected_resources, energy)`
lexicographically: a resource point beats any energy difference, and equal
scores draw. The deadline is true termination, so final win/loss rewards are
delivered and learners do not bootstrap beyond the round.

Rewards are `-0.1` per turn, `+0.05` per collectible, `+10` for a win and `-10`
for a loss. Losing your factory also costs `-10`, including on a double-death
draw, so dying together early is not rewarded over surviving. Mining has net
reward `-0.1`. Terminal scores decide tournament results independently of reward.

Actions use the same pre-turn state: occupied starting cells cannot be entered,
and same-destination movement/build conflicts cancel without energy costs.
Removing the same wall twice removes only that wall. Building/removal actions
are disabled in the default contest, but custom overrides use the same conflict
rules with mirrored adjacency preferences. Observations have visibility radius
4 and relative friendly/enemy codes; coordinates and action names are absolute.
Each player receives only its own events in partial-observation `info`. The
full event log is reserved for match results and debugging after play.

Use the two-agent loop for learning, rather than the single-player commands:

```python
from student.agent import StudentAgent
from factory_frontier.agents import SurvivalAgent
from factory_frontier.competition import play_match

learner = StudentAgent(seed=0)
for seed in range(100):
    result = play_match(learner, SurvivalAgent(), seed=seed, training_owners=(0,))
```

Use `training_owners=(0, 1)` to update both learners. Each update receives its
own pre-turn observation, selected action, reward, next observation, termination
and truncation flags. Both actions are selected before updates; frozen opponents
receive no updates. Keep learned parameters between matches and reset only
episode memory. Evaluation defaults to no learning. The low-level Gym `step()`
is a debug convenience against a waiting opponent; real contests use
`step_multi()` or `play_match()`. Single-player training/evaluation helpers reject
competition configs to avoid silently using a waiting opponent.

Results include player names/seats, outcome, final score tuples, end reason,
progress, resources, energy spent, full events and a configuration snapshot.
Call `result.save_replay(path)` or pass `replay_path` to `play_match` for JSON;
`render_replay` reconstructs frames using the saved settings, including custom
configurations. The leaderboard awards 3 points for a win and 1 for a draw,
then orders ties by wins, gathered resources and name. Algorithms, checkpoint
loading and student strategy remain in `student/`.

```powershell
python -m factory_frontier.examples.tournament
```

For submission, provide the `student/` directory, including an importable
`StudentAgent` implementation and any required checkpoint files. Grading should
place it beside a clean official engine, so modifications outside `student/`
cannot affect results.

## Layout

```text
factory_frontier/
  core/          actions, state, entities, rules, maps, observations, environment
  configs/       curriculum presets
  rewards/       composable reward components
  agents/        interface and basic references
  training/      minimal learning loop
  evaluation/    seeded evaluation
  competition/   matches, replay data, round-robin leaderboard
  rendering/     ASCII renderer
  examples/      student template and runnable examples
tests/            deterministic unit and integration tests
student/          writable coursework, training, evaluation, and student tests
```

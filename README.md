# Factory Frontier

Factory Frontier is a grid-based, Gymnasium-compatible reinforcement-learning teaching environment. A single rules engine supports a semester-long progression from a deterministic introductory task to reproducible two-agent tournaments. The canonical action indices and observation keys/shapes do not change between stages, so student code and checkpoints can move forward with the curriculum.

## Install and try it

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
| `function_approximation` | larger layouts and full action vocabulary |
| `policy_methods` | meaningful stochastic transitions |
| `ppo` | rooms, cooldowns, entities, partial observability |
| `competition` | symmetric maps and simultaneous two-agent play |

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

Extension points are intentionally small:

- Add entity data/types in `core/entities`; behavior remains in rules.
- Add map generators through `core/map_generation` and select them in a config.
- Add a reward callable implementing `RewardComponent` and pass it through `reward_components=`; components receive immutable before/after context.
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
from factory_frontier.agents import RandomAgent, RuleBasedAgent
from factory_frontier.competition import round_robin

leaderboard, matches = round_robin(
    {"student_a": RandomAgent(seed=1), "student_b": RuleBasedAgent()},
    seeds=(10, 11, 12),
)
```

Tournament pairs play both seat assignments on every seed. Results include win/loss/draw, survival steps, progress, gathered resources, energy spent, and the engine event log. Call `MatchResult.save_replay(path)` or pass `replay_path` to `play_match` for JSON replay data; `render_replay` deterministically reconstructs ASCII frames. Matches call the same engine's simultaneous `step_multi`; there is no separate competition game.

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

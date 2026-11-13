# Factory Frontier

A grid-based reinforcement-learning teaching environment, from a first movement
policy to two-agent tournaments. All nine curriculum presets use the same
Gymnasium API, action indices and default observation shapes.

## Start here

Download or clone the **complete repository** and use Python 3.10 or newer.
The source archive also includes the student workspace; the wheel installs
only the engine and reference examples.

- [Student walkthrough](student/WALKTHROUGH.md): setup, first run and first successful policy.
- [Student reference](student/README.md): game rules, agent interface and algorithm progression.

Students write their code in `student/`. The starter runs immediately with a
random legal policy; students implement the assigned learning algorithms.

## Repository layout

| Path | Contents |
|---|---|
| `student/` | Editable agent, training/evaluation scripts, tests and checkpoints |
| `factory_frontier/core/` | State, actions, rules, maps and observations |
| `factory_frontier/configs/` | Curriculum presets and configuration validation |
| `factory_frontier/agents/` | Agent interface and reference baselines |
| `factory_frontier/wrappers/` | NumPy feature encoding and action masking |
| `factory_frontier/training/`, `evaluation/` | Single-player learning and evaluation loops |
| `factory_frontier/competition/` | Matches, tournaments and replays |
| `factory_frontier/rewards/`, `rendering/`, `examples/` | Reward components, ASCII display and examples |
| `tests/` | Official environment tests |

## Development and distribution

After following the setup walkthrough, run `python -m pytest` and
`python -m pip check`. [CI](.github/workflows/checks.yml) checks Windows/Linux,
multiple Python versions, and wheel/source distributions.

Instructors can customize `FactoryFrontier("ppo", ...)` with configuration
overrides, reward components and rule hooks. See
[presets](factory_frontier/configs/presets.py),
[environment](factory_frontier/core/environment.py) and
[rules](factory_frontier/core/rules.py). Check feasibility after changing settings.
Preserve existing action indices and observation meanings when extending the engine.

For grading, copy each submission's `student/` directory beside a clean official
engine. This project is distributed under the [Apache 2.0 license](LICENSE).

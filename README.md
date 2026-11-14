# Factory Frontier

A grid-based reinforcement-learning teaching environment, from a first movement
policy to PPO. The eight student curriculum presets share the Gymnasium API,
action indices and default observation shapes. Competition is deferred; its
engine code is retained outside the current course scope.

## Start here

Use Python 3.10+ and download or clone the **complete student repository**.
The source archive includes the student workspace; the wheel installs only the
engine and reference examples.

- [Walkthrough](student/WALKTHROUGH.md): setup, first run and first successful policy.
- [Student reference](student/README.md): rules, agent interface, algorithms and training commands.

Students implement their algorithms in `student/agent.py` and may add helper
modules and tests under `student/`. The shared scripts run training, evaluation
and visualization. The starter selects random legal actions and does not learn.

## Repository layout

| Path | Contents |
|---|---|
| `student/` | Editable agent, shared scripts, tests, checkpoints and results |
| `factory_frontier/core/` | State, actions, rules, maps and observations |
| `factory_frontier/configs/` | Curriculum presets and validation |
| `factory_frontier/agents/` | Agent interface and reference baselines |
| `factory_frontier/wrappers/` | NumPy feature encoding and action masking |
| `factory_frontier/training/`, `evaluation/` | Shared single-player loops |
| `factory_frontier/rewards/`, `rendering/`, `examples/` | Rewards, ASCII display and examples |
| `factory_frontier/competition/` | Deferred tournament extension |
| `tests/` | Official engine tests |

## Instructor and release notes

`student/sample_code/` holds local instructor solution guidelines. It is excluded
from normal Git staging and source distributions. Keep it out of manually
assembled ZIP files and student handouts as well; students receive the starter
agent, not completed solutions. Preserve a private copy of the instructor files.

Before distribution, run `python -m pytest` and `python -m pip check`.
[CI](.github/workflows/checks.yml) checks Windows/Linux, multiple Python versions,
and wheel/source distributions. Keep tests independent of a particular completed
student algorithm so the unchanged starter passes the checks.

Instructors can customize presets through `FactoryFrontier("ppo", ...)`, reward
components and rule hooks. See [presets](factory_frontier/configs/presets.py) and
[rules](factory_frontier/core/rules.py); check feasibility after changing settings.
Preserve action indices and existing observation meanings when extending the engine.

For grading, place each submission's `student/` beside a clean official engine.
The project uses the [Apache 2.0 license](LICENSE).

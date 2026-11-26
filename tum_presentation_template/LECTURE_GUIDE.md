# RL project case study: teaching guide

The deck uses the supplied TUM Beamer class, fonts, logo and 16:9 layout. The cover follows the supplied course example, with the School of Management, Chair of Business Analytics and Heilbronn date of 25.11.2026. The original `main.tex` and template files remain unchanged.

Content slides show `Introduction to Reinforcement Learning | Project Case Study` in the footer, with the page number at the right. Code uses the template's monospaced font, with syntax colors for Python keywords, shell commands, options, strings and comments.

## Suggested class flow

Allow about 90 minutes. Adapt these times to the class; they are a proposed session plan, not course requirements.

| Slides | Activity | Suggested time |
|---|---|---:|
| 1–3 | Introduce the case study and session workflow | 5 min |
| 4–6 | Download from public GitHub, install and test; follow the relevant OS slide | 15 min |
| 7–8 | Tour the workspace and trace the agent loop | 10 min |
| 9–15 | Guided walkthrough with pauses at checkpoints | 35 min |
| 16–17 | Independent exploration and experiment notes | 20 min |
| 18 | Check readiness and address blockers | 5 min |

The two final reference slides remain in the source but are hidden from the PDF. Change `\showreferencesfalse` to `\showreferencestrue` to include them. Students can clone the public repository or download its ZIP directly from GitHub.

The source also contains per-slide Beamer notes, including discussion prompts and repository references. They are hidden in the student-facing PDF. To produce notes, add `\setbeameroption{show only notes}` before `\begin{document}` in a copy of the source.

## Before class

- Use the public GitHub repository: <https://github.com/tbhrobrecht/factory-frontier>. Students can clone it or use Code / Download ZIP without a GitHub account or invitation. A wheel alone omits the student workspace.
- For a manually assembled archive, exclude `student/sample_code/` and preserve the instructor copy separately. It contains local instructor solutions.
- Use a clean starter for the first demo. Run `python -m pytest` and `python -m pip check` before distribution, as the repository README requests.
- Have the walkthrough and `student/agent.py` open. Keep a saved copy of the starter so the random run is reproducible after demonstrating edits.
- No deadlines, team sizes, grading weights or course submission URLs were supplied. Announce those separately when decided. The slide activities are learning checkpoints, not invented assessment requirements.

## Guided walkthrough prompts

1. **Random starter:** run the slide 9 command without edits. Seed 42 gives 40 steps, no winner and approximately −0.4 return. Ask why legal actions can still produce failure.
2. **First policy:** on slide 10, put the `Action` import at file level. Indent the replacement `act` method inside `StudentAgent`. Retain the existing `Observation` import. Expect 8 steps, winner 0 and approximately 9.92 return.
3. **Observation:** indent slide 11's diagnostic statements inside `reset()`. Explain active versus padded shape, coordinates and the legal-action mask. Remove the prints afterward.
4. **Evaluation:** use the same fixed seeds for both policy variants. North/east and east/north both take a shortest intro path, so path changes need not change return.
5. **Slip:** compare requested actions with realized positions. The `mdp` preset changes movement dynamics but does not automatically install a planning algorithm.
6. **Training:** show the call to `update()` in `student/train.py`. The starter update is empty. Neither the random policy nor the hand-written first policy learns by running the trainer.

## Independent work and discussion

Give students about 20 minutes for slide 16. Ask them to predict the path before running, make one policy change, retain a baseline result and explain an observed stochastic transition. Early finishers can write a legal-action test under `student/tests/`.

Ask each student to leave with the experiment note described on slide 17. This is a suggested record format. For the final checkpoint, ask students to point to the learning hook and explain the difference between running training and implementing learning.

## Rebuilding

From `tum_presentation_template/`, with a complete LaTeX installation:

```text
latexmk -pdf rl_project_case_study.tex
```

The supplied `latexmkrc` adds `tum/` to the TeX search path. Compilation may require a second pass for PDF outlines and frame counts. The presentation PDF for this task is also available in `../output/pdf/rl_project_case_study.pdf`.

The editable source is LaTeX, matching the supplied template. The PowerPoint export in `../output/pptx/rl_project_case_study.pptx` preserves each slide as a high-resolution image.

## Content references

- `../README.md`: repository layout, distribution rules and student workspace boundaries.
- `../student/WALKTHROUGH.md`: installation, starter, first successful policy and readiness checks.
- `../student/README.md`: rules, observations, curriculum, evaluation and checkpoints.
- `../student/agent.py`, `train.py`, `evaluate.py`: actual interface and command behavior.
- Repository URL and public status supplied by the instructor in this conversation.

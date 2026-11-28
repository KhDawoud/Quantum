# AI_USE_B: Model B assistance log

## 2026-11-26 — implementation against the team repository

- User requested that the earlier standalone RBF implementation be redone on
  the cloned repository, matching the other branches' conventions and outputs.
- Codex inspected main, the VQC and quantum-kernel branches, and the alternative
  evaluator branch. It copied the user-provided Desktop checkout into this
  chat's outputs folder and created local branch `feat/rbf-svm` from main.
- Codex replaced the Model B placeholder with scikit-learn SVC, preserved the
  requested eight-setting grid, added synthetic/math/integration tests, and
  added a runner using Person A's actual evaluator and result formatter.
- Codex wrote design notes and README instructions. Existing shared evaluator,
  preprocessing, team requirements, and other models were not modified.
- No branch was pushed and no commit was created. No human code edits or
  independent human verification are recorded for this delivery yet.

Validation results and environment limits are recorded in `DESIGN_B.md`.
Codex ran the complete main-branch test suite plus the new Model B tests
(31 passed, treating FutureWarning as an error), the standalone synthetic test,
and the full-grid and quick Pima runners. The generated CSVs are included.

## 2026-11-26 — team decision on balanced class weighting

- Read PROJECT_CONTEXT.md first and worked on the existing `feat/rbf-svm` branch.
- Previously Model B omitted `class_weight`, using SVC's unweighted default.
  Set `class_weight="balanced"` in `make_model()`; C, gamma, and all other
  model settings are unchanged. The grid still has eight combinations and
  does not tune class weighting.
- Added checks for the balanced setting and for weights computed from fit-time
  labels (7/3 gives 0.714286/1.666667; refitting with 5/5 gives 1/1).
- Updated design notes and marked old unweighted benchmark results as historical.
  Shared evaluation, data preprocessing, other models, and result CSVs were untouched.
- Only quick Model B tests were requested; no benchmark or long training run,
  commit, push, or Git author-setting change was made.
- Validation: six quick tests passed in 3.32 seconds, with FutureWarning treated
  as an error. The synthetic grid-search and shared-evaluator integration tests
  were deselected (two tests). `git diff --check` passed; `src/evaluate.py` and
  `src/data.py` have no changes.


## 2026-11-26 — deliverables aligned with Model C

- User requested the GitHub deliverable pattern of completed Model C applied
  to Model B. Codex fetched origin and inspected C `cb90428` and main `5bd1d52`.
  C's committed scope is model + tests + encoding experiment code; no C result
  files or Member C notes were present in that fetched branch.
- Codex made the RBF runner explicitly use 25 main splits / 5 quick splits,
  reran both with balanced weights, and replaced obsolete unweighted CSVs.
- Added the RBF-only training-size experiment using shared tuning/splitting/
  evaluation functions, 40 per-draw results, a figure, and a protocol test.
  Added an import in src/experiments.py; other experiment functions are unchanged.
- Updated README, design notes, and the installed-package snapshot. Installed
  the pinned Matplotlib version into the local environment to render the chart.
- Validation: all 33 local tests passed; benchmark and experiment completed;
  chart visually inspected. In an isolated export of main `5bd1d52`, all 38
  data/evaluator/logistic/RBF/experiment tests passed; VQC tests were not run.
- No human edits or independent human verification are recorded for this step.
  No commit, push, or Git author change was made. Publishing awaits the user's
  requested review/confirmation. Full all-model Experiment 2 is not claimed done.


## 2026-11-26 — publication authorized

- User explicitly approved adding the Model B package on top of main while
  preserving teammates' work and avoiding duplicates.
- Refreshed origin/main (`5bd1d52`) and checked for overlapping contributions.
  The RBF implementation replaces only its placeholder; B-specific results,
  figure, runner, tests, and notes are new paths. README gets a B section and
  src/experiments.py gets only the B experiment import.
- Integrate this contribution on the fetched main history and use a normal
  fast-forward push. Do not force-push or merge unrelated model branches.


## 2026-11-28 — Experiment 1 completion and publication

- User requested running RBF Experiment 1 and pushing it in the locations used
  by teammates. Codex worked from main `e11fd45` in an isolated worktree,
  preserving unrelated local runner changes and rerun results.
- Matched A/C/D's five-split, full-grid protocol and the notebook's expected
  exp1_rbf_svm_q4/q6 filenames. Reused published 8-feature baseline rows 0..4.
- Extended src/experiment_rbf.py with an explicit exp1 CLI; no-argument behavior
  still runs Experiment 2. Model, preprocessing, and shared evaluator unchanged.
- Generated the two CSVs, an RBF figure, experiment notes and paired tests.
  Verified selected columns and split sizes agree with A, C and D.
- Refreshed only the combined notebook's Experiment 1 output/coverage text
  and combined feature-count figure; teammate results are preserved.
- No independent human pre-run prediction or human verification is claimed.
  The model grid was not changed in response to observed scores.
- Validation: all 77 tests passed; the notebook Experiment 1 cell executed
  successfully and verified identical selected columns across all four models.
  Both changed charts were visually inspected. Existing result CSVs, other
  notebook sections, and the kernel-spread figure are unchanged.

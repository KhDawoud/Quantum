# AI use log: Member D (Variational Quantum Classifier)

Required by PROJECT_CONTEXT.md ("Keep track of AI tool usage for the final report").
Newest entries at the bottom. Update after every working session.

## Tool
Claude Code (Anthropic, model "Sonnet 5.5") inside VS Code, working in this repository.

## Session 1: Day 1 (2026-11-25): building `src/model_vqc.py`

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Docs check | Read the installed PennyLane 0.45.1 docstrings (`StronglyEntanglingLayers`, `qnode`, `AdamOptimizer`) instead of relying on memory. | Confirmed `weights` shape `(L, M, 3)`, `diff_method='backprop'`, broadcasting, `qml.AdamOptimizer` exist. Found the 0.45 docs write the alias `qp` where older material writes `qml`; both import fine, no warnings. |
| Circuit | Wrote the encodings (angle, ZZ-style), the trainable layers and the `<Z_0>` readout. | Unit test: batched run equals patient-by-patient run. Unit test: `IsingZZ` equals the literal CNOT-RZ-CNOT sandwich from PROJECT_CONTEXT.md. |
| Estimator | Wrote `VQCClassifier` (fit / predict / predict_proba / `loss_history_`), weighted BCE, Adam, mini-batches, seed 42. | Unit tests: shapes, probabilities in [0,1], same seed gives identical weights, rejects inputs outside [0, pi], works with `clone` and `GridSearchCV`. |
| Gradients | Added a finite-difference check of the autograd gradient. | Passes (atol 1e-6) for both encodings. |
| Sanity checks | Wrote `src/vqc_sanity.py` (overfit / easy columns / loss / timing). | Run, results printed and recorded in `notes/DESIGN_D.md`. |

### Adapted sources (also cited in code comments)
- PennyLane demo "Variational classifier" (https://pennylane.ai/qml/demos/tutorial_variational_classifier): pattern of weights array + batch cost function + `optimizer.step`.
- PennyLane 0.45.1 documentation for `qml.StronglyEntanglingLayers`, `qml.qnode` (parameter broadcasting, `diff_method='backprop'`), `qml.AdamOptimizer`.
- ZZ-style feature map as specified in PROJECT_CONTEXT.md (after Havlicek et al., Nature 567, 209 (2019)).

### Corrections / things the AI got wrong first (for honesty in the report)
- A test I wrote used unshuffled folds on sorted labels, so `GridSearchCV` saw one-class folds; fixed with stratified shuffled folds.
- First idea was to store the compiled circuit inside the fitted model; changed to rebuild it in `predict_proba` so fitted models stay picklable.
- The "overfit 20 patients" check did NOT pass at first (75-80%). Investigated instead of loosening it; see DESIGN_D.md.

### Human review still needed
- Member D should re-run `python -m src.vqc_sanity <section>` and `pytest -v` and be able to explain each choice in DESIGN_D.md on Nov 30.

### Day-1 timing result (same session)
Measured with `python -m src.vqc_sanity timing`: 25 splits would take about 5-7 h (> 3 h), so the 5-split fallback was recommended. Recorded in DESIGN_D.md.

## Session 4: 2026-11-26: read-only review of teammate C's quantum kernel (`origin/feat/qkernel`)

Findings in `notes/REVIEW_C.md`. Nothing merged, committed or pushed; no source or test file edited.

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Fetch and compare | `git fetch --all`; read C's files with `git show` and a temporary worktree (removed). | Diff vs initial commit, grep of grids and encodings. |
| Tests | Ran `tests/test_qkernel.py` once in the temporary worktree with the shared `.venv`. | 18 passed in 5.5 s. Installed `matplotlib==3.11.2` into `.venv` (it was missing). |
| Cross-check | Tested C's runner against `main`'s `results/logreg.csv`. | Found the training-time column is dropped (`train_time_sec` not recognised). |

### Limits (for honesty in the report)
- C committed no results files, so the paired comparison with logreg and VQC could not be done.
- No full `evaluate()` run of C's model was performed; the run-time projection is C's own benchmark.

## Session 5: 2026-11-26: check that the VQC matches the team's "balanced class weighting" decision

Result: **no change to `src/model_vqc.py` was needed.** The code already matches balanced weighting exactly.

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Code check | Read `fit()`, `predict_proba()`, `predict()` and `PARAM_GRID` in `src/model_vqc.py`. | Weights computed in `fit()` (line ~167) as `n / (2 x count)` from the `y` it receives; per-patient weights sliced with the same mini-batch indices; loss is `mean(weight x cross-entropy)`; `predict` does not use them; no weighting option in `PARAM_GRID`. |
| Test | Added `test_balanced_class_weights_computed_from_the_labels_given_to_fit` to `tests/test_vqc.py` (7/3 labels give 0.714 and 1.667; 5/5 gives 1/1; 9/1 gives 0.556 and 5.0; ratio equals the count ratio). | Ran it with 4 other quick tests (class-weight test, grid size, IsingZZ equivalence, input range): 5 passed in 4 s. The slower training tests were not run. |
| Notes | Added the team sentence to `notes/DESIGN_D.md`. | n/a |

### Honest limits
- The loss divides by the batch size, not by the sum of the weights, so the average weight of a mini-batch varies slightly with its class mix (same convention as scikit-learn's weighted losses). Not a mismatch, but worth knowing.
- A first attempt to run the whole `tests/test_vqc.py` exceeded 2 minutes (the training tests are not quick, or the CPU was shared) and was stopped.

## Session 6: 2026-11-26: switch the main table from 5 to 25 splits

Team decision: main table = 25 splits for all models; experiments = first 5 splits, passed explicitly; quick mode = 5. `src/evaluate.py` belongs to teammate A, who agreed to the change.

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Search | Searched the repo for `DEFAULT_EVAL_SPLITS`, `n_splits_to_run`, "5 splits", "first 5", "fallback" and hard-coded split counts. | Listed every hit before editing. |
| `src/evaluate.py` | `DEFAULT_EVAL_SPLITS` 5 -> 25; new `QUICK_EVAL_SPLITS = 5` (used by quick mode) and `EXPERIMENT_EVAL_SPLITS = 5`; replaced the fallback comment. Split generation, training-size mode and `compare_models()` unchanged. | |
| Tests | Updated the default-run test to 25; the qubit and moons tests now pass `n_splits_to_run=EXPERIMENT_EVAL_SPLITS`; added tests for the constants, 25 by default, quick = 5 (even when asked for 25), experiment = 5, and first 5 of 25 identical to the 5-split run (same rows in the CSV). | `tests/test_evaluate.py` + `tests/test_data.py`: 23 passed in 25 s. No VQC training. |
| Notes | Updated PROJECT_CONTEXT.md, notes/DESIGN_A.md, notes/DESIGN_D.md, notes/REVIEW_C.md. | n/a |

### Honest limits
- `results/logreg.csv` and `results/vqc_zz.csv` are 5-split results and must be rerun for the main table; not rerun or edited here.
- C's experiment runner on `feat/qkernel` still needs `n_splits_to_run=EXPERIMENT_EVAL_SPLITS` (not on main, not changed).
- A note in DESIGN_D.md says "25 split with 8 combinations"; the team decision and PROJECT_CONTEXT say at most 4 combinations for the VQC. Left as written, flagged to the user.

## Session 7: 2026-11-26: diagnostic, is the VQC underfitting?

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Diagnostic | Ran a scratch script (deleted afterwards) on exam 0: preprocessing fitted on the study patients only, then ZZ VQC, angle VQC and balanced logistic regression (C = 0.3 and 1.0) trained on them. Reported loss per round, study-patient accuracy and balanced accuracy, and the share predicted diabetic. | ZZ exam balanced accuracy (0.584) matched exam 0 in `results/vqc_zz_25.csv`. Total run time 37 s. |
| Notes | Added the "Diagnostic: is the VQC underfitting?" section to the end of `notes/DESIGN_D.md` (text supplied by the user, from the AI's report). | n/a |

### Honest limits
- One exam only; the causes listed are hypotheses, not proven.
- The diagnostic printed exam-0 test scores; no design choice was changed because of them (disclosed in DESIGN_D.md).
- The pasted section refers to a "dead-dials note" that does not exist in DESIGN_D.md, and quotes main-run figures (63.1% balanced accuracy, 6 min per exam) that I did not check against `results/vqc_zz_25.csv`.

## Session 8: 2026-11-26: experiment 3 (angle vs ZZ encoding)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Step 0 | Checked git (clean, up to date), the names in `src/evaluate.py` and `src/model_vqc.py`, that `results/vqc_zz_25.csv` has 25 rows, matplotlib installed, and that `src/model_vqc.py` was last changed (14:47) before the main run's results (22:02). | All checked by commands. |
| Prediction | Wrote the user's own prediction into `notes/EXPERIMENTS_D.md` at 22:42, before any run. | n/a |
| Script | Wrote `src/run_exp_d.py`: runs through the shared `evaluate()`, refuses to overwrite, writes the log and the manifest row, and produces the comparison table and figure. | Quick check (1 exam, 2 settings) in `scratch_quick/`, then deleted. |
| Run | Angle encoding, 5 exams, 4 combinations, one run, nothing else running (26 min). | `results/experiments/exp3_vqc_angle_5ex.csv`; compare_models vs the first 5 rows of the ZZ main run: +8.5 points balanced accuracy, p = 0.001. |
| Notes | Results, per-exam winners, figure, and explanation (marked hypothesis where not measured). | n/a |

### Honest limits
- The manifest's git commit (`cab351d`) is the commit checked out during the run; `src/run_exp_d.py` itself was not yet committed at that time.
- The manifest's command is written as `python -m src.run_exp_d exp3`; it was run with `.venv/Scripts/python`.
- The comparison with logistic regression is for reference only and was not saved as a file.

## Session 9: 2026-11-26: experiment 1 (4 and 6 columns, ZZ)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Prediction | Wrote the user's prediction into `notes/EXPERIMENTS_D.md` at 23:23, before any run, plus a note that it was also written after experiment 3's result. | n/a |
| Script | Added `exp1`, `exp1-dials`, `exp1-report` to `src/run_exp_d.py`. | Quick checks for 4 and 6 columns in `scratch_quick/`, then deleted. |
| Live dials | Gradient check on `build_qnode` with random dials: 12/24, 20/36, 28/48 at 4/6/8 qubits, same for ZZ and angle. | Run took 10 s. |
| Runs | 4 columns, then 6 columns, one after the other (8 and 12 min). | compare_models vs the first 5 rows of the ZZ main run: 4 vs 8 +3.1 points (p = 0.47), 6 vs 8 -2.3 points (p = 0.56); no significant difference found. |
| Code reading | Found that `PimaPreprocessor` outputs the kept columns in rank order (`src/data.py` lines 207, 235), so ZZ neighbour pairs change with the column count. | Confirmed from `selected_features` in the result files. |

### Honest limits
- The manifest rows record commit `cab351d`; `src/run_exp_d.py` was not yet committed at that time.

## Session 10: 2026-11-27: experiment 2 (training patients, ZZ)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Prediction | Wrote the user's prediction (verbatim) into `notes/EXPERIMENTS_D.md` at 00:24, before any run, plus a note that it was written after experiments 3 and 1. | n/a |
| Script | Added `exp2` and `exp2-report` to `src/run_exp_d.py`: tune once on the 614-patient pool (`tune_hyperparameters`), save the winner in `results/experiments/logs/exp2_vqc_tuning.txt`, then 4 sizes x 10 draws with `fixed_params`. | Quick check (3 draws of 30) in `scratch_quick/`, then deleted. |
| Run | Tuning winner epochs 40, lr 0.05 (297 s); 40 trainings; total 9 min, one step at a time. | Balanced accuracy 54.5% (30), 54.6% (60), 57.5% (120), 60.5% (200). |
| Notes | Results, spread, figure, explanation (hypothesis marked). Pointed out that the significance part of the prediction cannot be tested with `compare_models`. | n/a |

### Honest limits
- No significance test between sizes was run.
- Manifest rows record commit `cab351d`; `src/run_exp_d.py` was not yet committed then.

## Session 11: 2026-11-28: read-only check of experiment 5 (moons)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Search | `git fetch --all --prune`, then `git ls-tree`, `git grep -i` and `git log` on main and every remote branch for moons / exp5 files, text, results, figures and notes. | Read-only git commands only. No model or experiment was run; `compare_models` was not run. |
| Results file | Read `results/exp5_qkernel_moons.csv` from main (C's ZZ kernel) and computed its rows, split ids, n_train / n_test, mean and spread. | 5 rows, 320 / 80, balanced accuracy 93.25% +/- 3.38. |
| Comparison | Compared the data and `evaluate()` calls of C and A's deleted run with the agreed setup. | Everyone used 400 points, not 768; no shared `src/moons_data.py`. |
| Report | Wrote `notes/MOONS_CHECK.md`. | n/a |

### Honest limits
- A's deleted run is known only from A's own AI log, not from a file (unverified).
- The per-file commit for C's CSV and figure is unverified; where "768 patients" was agreed is unverified.
- Nothing else was edited, created, deleted or committed.

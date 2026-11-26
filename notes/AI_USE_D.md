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

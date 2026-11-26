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

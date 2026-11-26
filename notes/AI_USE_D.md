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

## Session 2: 2026-11-26: implementing `src/evaluate.py` (branch `evaluate`)

Note: this file was copied from branch `vqc` onto branch `evaluate` so the log stays in one place; when merging, keep both sessions.

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Reading | Read PROJECT_CONTEXT.md, `src/data.py`, `tests/test_data.py` and the skeleton; reused `load_pima` and `PimaPreprocessor` unchanged. | n/a |
| `get_splits` | `RepeatedStratifiedKFold`, seed 42. | Tests: same splits every call; each patient tested 5x in 25 splits and 1x with `n_repeats=1`; first 5 of 25 equal the 1-repeat splits. |
| `evaluate` | Per split: fit preprocessor on training rows, inner 3-fold `GridSearchCV`, final refit (timed), score test rows, write CSV. Added `n_repeats` and `n_draws` only. Quick mode and training-size mode. | Tests: CSV columns and row counts, quick mode, `fixed_params`, training-size mode (one fixed 154-patient test set). |
| Leakage tests | Spy subclass records exactly which rows the preprocessor is fitted on; second test flips test labels and checks the tuned settings do not change. | Pass. |
| `compare_models` | Nadeau-Bengio corrected paired t-test. | Tests: identical results give mean_diff 0 and p 1.0; made-up pair matches a hand calculation; zero-variance and mismatched-split cases. |
| Real run | Ran a temporary logistic regression in quick mode and 5-split mode. | Quick: 0.1 s. 5-split: 1.3 s. Temporary CSVs deleted. |
| Environment | Found `.venv` without pandas / scikit-learn; installed the pinned versions from `requirements.txt`. | `pytest`: 28 passed (18 new + 10 existing data cases; first logged as "20 + 8", corrected in Session 3). |

### Things to know / honest limits
- Tuning runs after the preprocessor is fitted on the whole outer training set (spec order). No test rows involved; see `notes/EVALUATE_NOTES.md`.
- `evaluate()` cannot take make_moons data yet (signature frozen); flagged in `notes/EVALUATE_NOTES.md`.

### Human review still needed
- Re-run `.venv\Scripts\python -m pytest -v` and be able to explain the Nadeau-Bengio correction and the no-leakage order of steps on Nov 30.

## Session 3: 2026-11-26: read-only review of the evaluate work

Full findings: `notes/EVALUATE_CHECKS.md`. No file in `src/` or `tests/` was edited; no commit, no branch switch.

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Fit count | Re-counted trainings per split. | Counter wrapped around `fit()`: 19 per split for logreg. |
| Leakage tests | Planted two deliberate leaks in a scratch copy of the code. | Test 1 caught "fit on all rows", test 2 did not; neither caught a peek inside tuning. |
| Branch comparison | Compared tests, data and `evaluate.py` across branches with read-only git commands. | No tests lost; no other `evaluate.py`. |

### Corrections to my own earlier statements (for honesty in the report)
- I said tuning costs "about 7 fits per split": wrong, it is 19 (6 settings x 3 inner folds + 1 final fit). The code was right, my explanation was not.
- I reported "20 new + 8 existing data tests": it is 18 new + 10 existing cases (8 functions, one runs 3 times).
- My leakage tests are weaker than I implied (see EVALUATE_CHECKS.md, concern 5b).

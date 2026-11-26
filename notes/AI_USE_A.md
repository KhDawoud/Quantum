# AI use log: Member A (Shared Setup, Logistic Regression, Exp 1 & 5)

Required by `PROJECT_CONTEXT.md` ("Keep track of AI tool usage for the final report").
Newest entries at the bottom. Update after every working session.

## Tool
Antigravity (Google Deepmind coding assistant) working inside this repository.

## Session 1: 2026-11-26 — Environment setup & `src/model_logreg.py`

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Environment setup | Created `.venv` with Python 3.13 and installed pinned dependencies from `requirements.txt`. | Verified all 10 existing tests in `tests/test_data.py` pass in `.venv`. |
| TDD tests (`tests/test_logreg.py`) | Wrote 6 tests covering `PARAM_GRID` budget (`<= 8`), model defaults/overrides, linearly separable synthetic fit/proba, seed-42 determinism, `Pipeline` + `GridSearchCV` warning-free convergence on Pima, and reduced-feature / `make_moons` compatibility. | Ran `.venv\Scripts\pytest.exe tests/test_logreg.py -v` before implementation and confirmed all 6 tests failed as expected (`NotImplementedError` / placeholder grid). |
| Logistic Regression (`src/model_logreg.py`) | Implemented `make_model(**kwargs)` (`solver='lbfgs'`, `max_iter=1000`, `random_state=42`, default L2 without deprecated `penalty`) and `PARAM_GRID` (`C` in `[0.01, 0.1, 1.0, 10.0]`, `class_weight` in `[None, 'balanced']`). | Ran `.venv\Scripts\pytest.exe -v` (all 16 tests passed in 3.96s with zero warnings) and `.venv\Scripts\python.exe -m src.data`. |
| Notes | Drafted `notes/DESIGN_A.md` with plain-language model explanation and design rationale table. | Cross-checked every parameter name, test count, and rule against `PROJECT_CONTEXT.md` and `src/model_logreg.py`. |

## Session 2: 2026-11-26 — Shared Evaluation Harness (`src/evaluate.py`)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| TDD tests (`tests/test_evaluate.py`) | Wrote 8 tests covering `get_splits()` determinism/stratification/coverage, outer-fold zero-leakage with extreme test perturbation, CSV schema + quick mode, oversized grid rejection, `n_features=4` & `make_moons`, `train_size` + `fixed_params` + `tune_hyperparameters()`, and exact Nadeau–Bengio t-test math + edge cases. | Ran `.venv\Scripts\pytest.exe tests/test_evaluate.py -v` before implementation and confirmed all 8 tests failed against the stub. |
| Evaluation harness (`src/evaluate.py`) | Implemented `get_splits()`, `get_fixed_test_split()`, `get_trainsize_splits()`, `tune_hyperparameters()`, `evaluate()`, `compare_models()`, and `format_summary()`, applying the 5-split fallback (`DEFAULT_EVAL_SPLITS = 5`) per `PROJECT_CONTEXT.md` and `origin/vqc:notes/DESIGN_D.md`. | Ran `.venv\Scripts\pytest.exe -v`: all 24 tests across `test_data.py`, `test_evaluate.py`, and `test_logreg.py` passed in 7.15s with zero warnings. |
| Official baseline run | Executed `.venv\Scripts\python.exe -m src.evaluate` to run Logistic Regression through the official pipeline and save `results/logreg.csv`. | Verified 5-split output (`74.47% +/- 4.58%` accuracy, `73.90% +/- 3.63%` balanced accuracy, `72.88% +/- 4.31%` macro F1, `0.2388s` mean fit time) and inspected `results/logreg.csv`. |
| Notes | Updated `notes/DESIGN_A.md` with `src/data.py` & `src/evaluate.py` design rationale, 5-split disclosure, Nadeau–Bengio explanation, and official `logreg` metrics. | Cross-checked reported numbers directly against `results/logreg.csv` and command output. |

## Session 3: 2026-11-26 — Fixed Balanced Weighting & Official 25-Split Baseline

| Step | What the AI did | What was checked by running code |
|---|---|---|
| TDD tests (`tests/test_logreg.py`, `tests/test_evaluate.py`) | Updated grid/default assertions to expect fixed `class_weight='balanced'` and 8 `C` values; added `test_balanced_class_weighting_adapts_to_fit_labels` checking imbalanced vs. 50/50 fit labels. | Ran `.venv\Scripts\pytest.exe tests/test_logreg.py -v` before code changes and verified the 3 expected failures. |
| Model update (`src/model_logreg.py`) | Fixed `class_weight='balanced'` inside `make_model()` (matching B, C, and D) and expanded `PARAM_GRID` to `C` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]`. | Ran `.venv\Scripts\pytest.exe -v`: all 54 tests across all 6 test modules passed in 25.05s with zero warnings. |
| Official 25-split run & paired test | Regenerated `results/logreg.csv` on 25 splits via `.venv\Scripts\python.exe -m src.evaluate` and ran `compare_models()` against `results/rbf_svm.csv`. | Verified `75.05% +/- 3.74%` accuracy, `74.21% +/- 3.77%` balanced accuracy, `73.32% +/- 3.86%` macro F1, and `"no significant difference found"` (`p = 0.6846` on balanced accuracy). |
| Notes | Updated `notes/DESIGN_A.md` with the fixed balanced-weighting rationale, official 25-split numbers, and LogReg vs. RBF SVM paired table. | Verified every reported metric and p-value directly against `results/logreg.csv` and tool output. |

### Adapted sources (also cited in code comments)
- `scikit-learn` `LogisticRegression` API documentation (`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`) for solver, default L2 behavior in scikit-learn 1.9.1, and `class_weight='balanced'`.
- `scikit-learn` cross-validation / `Pipeline` / `GridSearchCV` documentation (`https://scikit-learn.org/stable/modules/cross_validation.html`).
- Nadeau, C., & Bengio, Y. (2003). *Inference for the Generalization Error*, Machine Learning, 52(3), 239–281 (corrected resampled paired t-test formula).

### Human review still needed
- Member A should review the updated `src/model_logreg.py` and `notes/DESIGN_A.md` (including why `class_weight="balanced"` is fixed across all 4 models and what the LogReg vs. RBF SVM tie means).
- Next steps for Person A: commit and push these updates on `main`, then build Experiments 1 & 5 in `src/experiments.py` and `notebooks/main.ipynb`.

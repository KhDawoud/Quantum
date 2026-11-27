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

## Session 4: 2026-11-26 — Experiment 1 (Feature Count 4, 6, 8 — LogReg Only)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| TDD tests (`tests/test_experiment_logreg.py`) | Wrote 2 tests checking `evaluate()` is called for `n_features in [4, 6, 8]` with `n_splits_to_run=EXPERIMENT_EVAL_SPLITS=5`, CSV/PNG outputs, valid `selected_features` counts, and `src.experiments` re-export. | Ran `.venv\Scripts\pytest.exe tests/test_experiment_logreg.py -v` before implementation and verified both failed with `ModuleNotFoundError`. |
| Exp 1 LogReg runner (`src/experiment_logreg.py`, `src/experiments.py`) | Implemented `run_exp1_qubits_logreg()`, `plot_qubits_logreg()`, and `summarize_selected_features()`, plus 1-line export in `src/experiments.py` matching B's pattern. | Ran `.venv\Scripts\pytest.exe -v`: all 56 tests passed in 30.22s with zero warnings. |
| Official Exp 1 LogReg run | Ran `.venv\Scripts\python.exe -m src.experiment_logreg` producing `results/exp1_qubits_logreg.csv` (15 rows) and `figures/exp1_qubits_logreg.png`. | Verified `k=8` rows match the first 5 splits of `results/logreg.csv` identically via `pd.testing.assert_frame_equal`, inspected `figures/exp1_qubits_logreg.png` visually, and recorded paired `compare_models()` p-values. |
| Notes | Added Section 5 to `notes/DESIGN_A.md` with `k=4,6,8` results, `f_classif` column frequencies, paired p-values, and takeaway marked with hypothesis. | Cross-checked all numbers and feature counts against `results/exp1_qubits_logreg.csv`. |

## Session 5: 2026-11-26 — Experiment 5 (`make_moons` Artificial Positive Control — LogReg Only)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| TDD tests (`tests/test_experiment_logreg.py`) | Added 2 tests verifying `evaluate()` is called with `n_features=2`, `zero_as_missing_cols=()`, `n_splits_to_run=5`, `400` points, explicit `"artificial"`/`"make_moons"` dataset labelling, CSV/PNG output, and `src.experiments` export. | Ran `.venv\Scripts\pytest.exe tests/test_experiment_logreg.py -v` before implementation and confirmed the 2 new tests failed (`AttributeError` / `ImportError`). |
| Exp 5 LogReg runner (`src/experiment_logreg.py`, `src/experiments.py`) | Implemented `load_moons()` (`n=400`, `noise=0.20`, `seed=42`), `run_exp5_moons_logreg()`, `plot_moons_logreg()` (Split-0 linear boundary + 5-split bars), and CLI `exp1`/`exp5`; re-exported `run_exp5_moons_logreg` in `src/experiments.py`. | Ran `.venv\Scripts\pytest.exe -v`: all 58 tests passed in 30.34s with zero warnings. |
| Official Exp 5 LogReg run | Ran `.venv\Scripts\python.exe -m src.experiment_logreg exp5` saving `results/exp5_moons_logreg.csv` and `figures/exp5_moons_logreg.png`. | Verified `83.50% +/- 2.05%` accuracy (`50.00%` floor) in `results/exp5_moons_logreg.csv` and visually inspected both panels of `figures/exp5_moons_logreg.png`. |
| Notes | Added Section 6 to `notes/DESIGN_A.md` documenting artificial-control settings, results, and the Build Guide caveat. | Cross-checked metrics and chosen `C` values against `results/exp5_moons_logreg.csv`. |

## Session 6: 2026-11-26 — Aligning Experiment 1 with Team Protocol (`notes/EXPERIMENTS_A.md`)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Shared-file cleanup | Reverted the temporary import in `src/experiments.py` and deleted superseded `results/exp1_qubits_logreg.csv` & `figures/exp1_qubits_logreg.png`. | Verified `git diff -- src/experiments.py src/model_logreg.py src/evaluate.py src/data.py results/logreg.csv` is empty. |
| Step 0 & TDD refactor (`tests/test_experiment_logreg.py`, `src/experiment_logreg.py`) | Refactored `run_exp1_qubits_logreg()` to save `results/exp1_logreg_q4.csv` and `results/exp1_logreg_q6.csv`, use the first 5 rows of `results/logreg.csv` as the `q8` baseline, and plot `figures/exp1_logreg.png`. | Verified TDD RED failure before edit, then ran `.venv\Scripts\pytest.exe -v` (all 56 tests passed in 30.49s with zero warnings). |
| Quick check, estimate & real run | Ran `quick=True` on `n_features=4` and deleted its temp CSV; verified `~2.27s` estimate from `results/logreg.csv`; ran `python -m src.experiment_logreg exp1`. | Inspected `results/exp1_logreg_q4.csv`, `results/exp1_logreg_q6.csv`, and `figures/exp1_logreg.png` visually; verified `compare_models()` p-values (`0.6735`, `0.3301`). |
| Notes | Created `notes/EXPERIMENTS_A.md` (recording honestly that Exp 1 ran before the prediction-first prompt) and updated `notes/DESIGN_A.md` Section 5. | Cross-checked every Step 0 symbol, CSV row, feature list, and p-value against tool output. |
| Experiment 5 removal | Deleted `results/exp5_moons_logreg.csv`, `figures/exp5_moons_logreg.png`, and removed Experiment 5 code/test/section at Member A's request so only Experiment 1 (and upcoming Experiment 2) remain. | Verified all 55 tests pass in `.venv\Scripts\pytest.exe -v` and no `exp5` files remain in `git status`. |

## Session 7: 2026-11-27 — Experiment 2 (Training Size 30, 60, 120, 200 — Logistic Regression)

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Repo sync | Fetched `origin/main` (`056b00e..c8eea62`), checked `git diff --name-status` for zero conflicts with local/frozen files, and ran `git pull --ff-only`. | Verified `src/model_logreg.py`, `src/evaluate.py`, `src/data.py`, `src/experiments.py`, and `results/logreg.csv` had zero local diffs. |
| Step A & B | Recorded Member A's verbatim prediction (`2026-11-27 07:14 +02:00`) and setup in `notes/EXPERIMENTS_A.md` before running any Exp 2 code or tests. | Verified prediction text and setup disclosure written first. |
| TDD (`tests/test_experiment_logreg.py`, `src/experiment_logreg.py`) | Added `test_exp2_trainsize_logreg_tunes_once_and_saves_per_size_csvs`, confirmed TDD RED, then implemented `run_exp2_trainsize_logreg()`, `plot_exp2_logreg()`, and shared `_plot_metric_curve()`. | Verified TDD RED failure (`AttributeError`), then TDD GREEN (`2 passed in 15.22s`). |
| Quick check, estimate & real run | Ran `quick=True` on `train_size=30` (deleted `exp2_logreg_quick_check.csv`), computed `< 2s` time estimate, and ran `python -m src.experiment_logreg exp2`. | Verified `tuned_params={'C': 0.1}`, inspected `exp2_logreg_n30/n60/n120/n200.csv` (10 rows each) and `figures/exp2_logreg.png` visually, and ran full `.venv\Scripts\pytest.exe -v` (all **75/75** tests passed in `36.18s`). |
| Notes | Completed Experiment 2 Steps C–H in `notes/EXPERIMENTS_A.md` honestly comparing results against Member A's pre-run prediction. | Cross-checked all means, SDs, min–max ranges, and floor (`64.94%`) directly against CSVs and tool output. |

### Adapted sources (also cited in code comments)
- `scikit-learn` `LogisticRegression` API documentation (`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`) for solver, default L2 behavior in scikit-learn 1.9.1, and `class_weight='balanced'`.
- `scikit-learn` cross-validation / `Pipeline` / `GridSearchCV` documentation (`https://scikit-learn.org/stable/modules/cross_validation.html`).
- Nadeau, C., & Bengio, Y. (2003). *Inference for the Generalization Error*, Machine Learning, 52(3), 239–281 (corrected resampled paired t-test formula).

### Human review still needed
- Member A should review `notes/EXPERIMENTS_A.md`, `figures/exp2_logreg.png`, and commit Experiment 1 & Experiment 2 files on `main`. Next team steps: compile the combined 4-model tables/charts, `notebooks/main.ipynb`, report, and slides.

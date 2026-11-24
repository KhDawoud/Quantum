# PROJECT_CONTEXT.md

Read this file before planning, editing code, or running experiments. It is the single source of truth for both team members and AI coding tools.

Competition: Q//Horizon 2026 (The American University in Cairo). Team of 4 students.
Deadline: Nov 28, 23:59 Cairo time (aim for 18:00). Presentation: Nov 30.
Goal: Build quantum machine learning classifiers, compare them FAIRLY with classical models on one dataset, and explain honestly where quantum helps, where it struggles, and why. Beating classical models is NOT required.
Judges reward: reproducible working code, explained design choices, a fair comparison, depth of insight, and clear code and report.

## HARD RULES
- **Qubits & Tools**: At most `10` qubits. Simulators only (`default.qubit`). PennyLane for quantum, scikit-learn for classical.
- **Determinism**: Random seed `42` everywhere (`RANDOM_SEED = 42`).
- **Zero Test-Data Leakage**: NEVER use test data for any decision (imputing, clipping, scaling, feature ranking, or tuning). Fit everything on training rows only, inside each outer split.
- **Angle Bounds**: Keep all encoded inputs in `[0, pi]` and quantum kernel `gamma <= 1.0` so single-qubit rotations never wrap past `pi`.
- **Single Harness**: Never write custom train/test split or scoring code inside model files. Every classifier plugs into `src/evaluate.py`.
- **Attribution**: Cite the source of any adapted tutorial or documentation snippet in a code comment and in the report.
- **Honesty**: Never hide or soften weak results. Report non-significant comparisons as `"no significant difference found"`, never `"equal"`. Keep track of AI tool usage for the final report.
- **Explanations & Git**: Explain choices in plain language so every member can answer judges' questions on Nov 30. Do not change git author settings (`-c user.name` or `-c user.email`).
- **Verification**: Run `.venv/bin/pytest -v` after code changes.

## DATASET
- **Pima Indians Diabetes**: OpenML dataset `'diabetes'` (ID `37`), cached locally at `data/raw/pima_diabetes.csv`.
- **Size & Labels**: `768` patients, `8` numeric columns (`preg`, `plas`, `pres`, `skin`, `insu`, `mass`, `pedi`, `age`). Target `tested_positive = 1` (`268`), `tested_negative = 0` (`500`).
- **Baseline Floor**: Always-guess floor = `65.1%`. Earlier dataset-screening scores (`77.1%` / `76.5%`) used different preprocessing: NEVER report them as final results.

## PREPROCESSING (training rows only, inside each split)
1. In `plas`, `pres`, `skin`, `insu`, `mass`: `0` means not measured -> convert to `NaN`. Do NOT touch `preg`.
2. Fill missing values with the training column median.
3. Clip each column to its training 1st–99th percentile.
4. Min-max rescale to `[0, pi]` using training values (clip out-of-range test values into `[0, pi]`).
Every model, quantum and classical, receives these exact same numbers.

## EVALUATION & FAIR COMPARISON
- **Splits**: 5-fold stratified cross-validation × 5 repeats = `25` splits (`seed = 42`), identical for every model.
- **Tuning**: At most `8` setting combinations per model (`<= 4` for VQC), 3-fold inner `GridSearchCV` on training rows only, `scoring='balanced_accuracy'`.
- **Metrics**: Accuracy, balanced accuracy, macro F1, training time (`mean ± std` across splits). Save CSVs in `results/` and charts in `figures/`.
- **Quick Mode**: First 5 splits and smaller grids for development.
- **Timing Decision (end of Day 1)**: Measure 1 VQC fit at 8 qubits. Full tuning needs ~325 fits (`25 splits × (4 combinations × 3 inner folds + 1 refit)`). If `325 × fit_time > 3 hours`, ALL models use the same first 5 splits (one 5-fold run) for the main table and experiments, and the report discloses this. Never mix split counts across models.
- **Significance Test**: `compare_models(results_a, results_b)` in `src/evaluate.py` applies the Nadeau–Bengio corrected resampled paired t-test (variance multiplied by `1/J + n_test/n_train`, where `J` is the number of splits).

## MODELS (Main comparison uses all 8 columns = 8 qubits)
- **Logistic Regression (`src/model_logreg.py`)**: Linear classical baseline.
- **RBF SVM (`src/model_rbf_svm.py`)**: Nonlinear classical baseline.
- **Quantum Kernel SVM (`src/model_qkernel.py`)**: ZZ-style feature map, 2 repetitions (`H` on every qubit; `RZ(x_i)` on each qubit `i`; linear neighbour pairs `(i, i+1)` with no wrap-around: `CNOT(i, i+1)`, `RZ(x_i * x_(i+1))` on qubit `i+1`, `CNOT(i, i+1)`). Fidelity kernel (`|⟨ψ(x_i)|ψ(x_j)⟩|^2`), `SVC(kernel='precomputed')`, `gamma` in `{0.5, 1.0}`. Also supports `'angle'` encoding and `kernel_stats(K)`.
- **VQC (`src/model_vqc.py`)**: Selectable `'zz'` or `'angle'` feature map, `2` trainable `StronglyEntanglingLayers` (configurable), `expval(PauliZ(0))` mapped to probability, Adam, binary cross-entropy with class weights, `diff_method='backprop'` with mini-batch vectorization, `loss_history_` recorded per epoch.

## CODE CONTRACT
- `src/data.py`: `load_pima() -> (X, y)`; `PimaPreprocessor` (scikit-learn transformer outputting `[0, pi]`, with `n_features` option to keep top-`k` columns ranked by `f_classif` on training rows).
- `src/model_*.py`: `make_model()` returning a scikit-learn style estimator with `fit(X, y)` and `predict(X)`, plus `PARAM_GRID` (`<= 8` combinations; `<= 4` for VQC).
- `src/evaluate.py`: `get_splits()`, `evaluate(name, make_model, param_grid, n_features=8, train_size=None, quick=False, fixed_params=None)`, and `compare_models(results_a, results_b)`.

## EXPERIMENTS (Nov 26 Freeze — one change at a time, CSV + PNG + 2–3 sentences on why)
1. **Number of Qubits**: `4`, `6`, `8` top columns (`results/exp1_qubits.csv`, `figures/exp1_qubits.png`) + `kernel_stats()` spread figure.
2. **Training Size**: `30`, `60`, `120`, `200` training patients on a fixed `20%` (`154`-patient) test set, `10` random draws each (`results/exp2_trainsize.csv`, `figures/exp2_trainsize.png`). Tune once on the `614`-patient pool and reuse via `fixed_params`.
3. **Encoding**: Simple angle vs. ZZ-style for both Quantum Kernel and VQC (`results/exp3_encoding_kernel.csv`, `results/exp3_encoding_vqc.csv`, `figures/`).
4. **Circuit Depth (spare, if time allows)**: VQC with `1`, `2`, `3` layers (`results/exp4_depth.csv`).
5. **Curved-Data Control**: `sklearn.datasets.make_moons` (`~400` points, moderate noise, `2` qubits) through `evaluate()` (`results/exp5_moons.csv`, `figures/exp5_moons.png`), time-boxed to 2 hours.

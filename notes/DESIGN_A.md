# DESIGN_A: design choices for Person A

Scope: `src/data.py` (preprocessing & feature ranking), `src/model_logreg.py` (classical linear baseline), and `src/evaluate.py` (shared CV harness & Nadeau–Bengio significance test).
Experiments 1 & 5 and `notebooks/main.ipynb` notes will be added next as they are built.

---

## 1. Shared Data & Preprocessing (`src/data.py`)

Why the preprocessing steps happen in this exact order, learned strictly inside `fit()` on training rows only:

| Step | Decision | Why, in plain language |
|---|---|---|
| 1. Hidden zeros -> `NaN` | Convert `0` to missing in `plas`, `pres`, `skin`, `insu`, `mass`; keep `preg` unchanged | A glucose, blood pressure, skin thickness, insulin, or BMI of `0` is biologically impossible and means "not measured" (`374` zeros in `insu`, `227` in `skin`). `0` pregnancies (`111` rows) is a real measurement, so leaving `preg` untouched keeps real information. |
| 2. Median imputation | Fill `NaN` using training column medians | Computing medians before clipping avoids letting extreme outliers skew the fill value, and medians resist skewed distributions like insulin better than means. |
| 3. Percentile clipping | Clip each column to its training 1st–99th percentile | A few extreme clinical measurements would otherwise stretch the min-max scale and squash normal patients into a tiny angle band. |
| 4. `[0, pi]` min-max scaling | Rescale clipped values into `[0, pi]` (clipping out-of-range test values to `[0, pi]`) | Single-qubit rotations wrap around past `2*pi` (or duplicate cosine projections past `pi`). Keeping inputs in `[0, pi]` prevents wrap-around ambiguity for quantum circuits while giving classical models the exact same numbers. |
| 5. Optional top-`k` ranking | Rank features by `f_classif` ANOVA F-score inside training rows (`n_features`) | Exp 1 tests 4, 6, and 8 qubits. Computing `f_classif` inside each training fold guarantees zero test-data leakage into column selection. |

---

## 2. Logistic Regression Baseline (`src/model_logreg.py`)

1. **Weighted sum**: multiplies each patient's 8 scaled numbers (`[0, pi]`) by a learned weight and adds an intercept offset.
2. **Sigmoid**: maps that score to a probability in `(0, 1)` for `tested_positive = 1`.
3. **Straight-line boundary**: predicting class 1 when `P >= 0.5` draws a flat (linear) decision boundary, giving us the linear baseline against RBF SVM, Quantum Kernel, and VQC.

| Choice | Why |
|---|---|
| `scikit-learn` `LogisticRegression` | Hard rule (`scikit-learn` for classical baselines); fast, convex, matches `fit`/`predict`/`predict_proba`. |
| `solver="lbfgs"`, default L2, `max_iter=1000` | Deterministic quasi-Newton solver for small tabular data; omitting deprecated `penalty` in `scikit-learn==1.9.1` keeps default L2 regularization with zero warnings, and `1000` iterations prevents non-convergence across all folds/sizes. |
| `random_state=42` | Reproducibility hard rule (`RANDOM_SEED = 42`). |
| `C` in `[0.01, 0.1, 1.0, 10.0]`, `class_weight` in `[None, "balanced"]` | `4 x 2 = 8` combinations (uses the full `<= 8` budget). Spans 4 orders of magnitude of regularization and lets inner balanced-accuracy tuning check whether balancing the 65/35 classes helps. |

---

## 3. Shared Fair-Comparison Harness (`src/evaluate.py`)

| Choice | Why |
|---|---|
| `get_splits()` (`RepeatedStratifiedKFold(5, 5, seed=42)`) | Generates all 25 deterministic stratified splits; every patient appears in exactly 5 test sets overall (and exactly 1 test set in the first 5 splits). |
| **5-split fallback (`DEFAULT_EVAL_SPLITS = 5`)** | Per `PROJECT_CONTEXT.md` Timing Decision and Member D's Day-1 measurement (`origin/vqc:notes/DESIGN_D.md`), 25 splits of 8-qubit VQC-ZZ would take ~6.0–7.8 h (`> 3 h`). Using the first 5 splits (one complete stratified 5-fold repeat, ~1.2–1.35 h for VQC) keeps all 4 models on identical splits and is explicitly disclosed here and in the report. |
| `Pipeline([("prep", PimaPreprocessor), ("clf", model)])` | Putting preprocessing inside the scikit-learn `Pipeline` guarantees both outer test rows AND inner 3-fold validation rows are transformed using only their respective training rows—zero data leakage during `GridSearchCV`. |
| Inner tuning (`3`-fold `GridSearchCV`, `balanced_accuracy`, `<= 8` combos) | Gives every model the same tuning budget and selects hyperparameters using balanced accuracy because Pima is 65.1% negative / 34.9% positive. |
| `train_size` + `fixed_params` + `tune_hyperparameters()` | Supports Experiment 2: draws 10 stratified subsets (`30`, `60`, `120`, `200`) from the `614`-patient pool evaluated on a fixed `20%` (`154`-patient) test set, skipping noisy inner CV on tiny sets by reusing settings tuned once on the `614`-patient pool. |
| `compare_models()` (Nadeau–Bengio corrected t-test) | Standard paired t-test assumes test folds are independent, which overstates confidence in k-fold CV because training sets overlap (`~75%` overlap in 5-fold CV). Nadeau & Bengio (2003) scale the fold-difference variance by `1/J + n_test/n_train` instead of `1/J`. When `p >= 0.05`, we report `"no significant difference found"` (never `"equal"`). |

---

## 4. Official Logistic Regression Results (`results/logreg.csv`, 5 splits, 8 features)

| Metric | Value (`mean ± std`) | Notes |
|---|---|---|
| Always-guess floor | `65.10%` | Guessing `tested_negative` for every patient (`500 / 768`) |
| Accuracy | `74.47% ± 4.58%` | Beats the majority floor by +9.37 percentage points |
| Balanced accuracy | `73.90% ± 3.63%` | Mean of sensitivity and specificity across both classes |
| Macro F1 | `72.88% ± 4.31%` | Unweighted mean of class-0 and class-1 F1 scores |
| Train time / split | `0.2388s ± 0.0192s` | Includes 3-fold `GridSearchCV` + refit on `[614, 8]` |

Inner 3-fold CV picked `class_weight="balanced"` on all 5 splits (`C=0.1` on 3 splits, `C=1.0` on 2 splits).

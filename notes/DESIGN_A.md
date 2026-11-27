# DESIGN_A: design choices for Person A

Scope: `src/data.py` (preprocessing & feature ranking), `src/model_logreg.py` (classical linear baseline), `src/evaluate.py` (shared CV harness & Nadeau–Bengio significance test), and `src/experiment_logreg.py` (Person A's Experiment 1 runner).
Detailed Experiment 1 & Experiment 2 protocol logs live in `notes/EXPERIMENTS_A.md`; `notebooks/main.ipynb` notes will be added when built.

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
| `class_weight="balanced"` (fixed) | Matches the Nov 26 team decision across all four models (B, C, and D also fix balanced weighting). Upweights minority positives dynamically inside each fit (`n / (2 * class_count)`), roughly `0.77` for negative and `1.43` for positive on full Pima, instead of spending tuning budget on class weights. |
| `C` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]` | `8` combinations (uses the full `<= 8` budget) across log-spaced inverse regularization strengths suited for `[0, pi]`-scaled inputs. |

---

## 3. Shared Fair-Comparison Harness (`src/evaluate.py`)

| Choice | Why |
|---|---|
| `get_splits()` (`RepeatedStratifiedKFold(5, 5, seed=42)`) | Generates all 25 deterministic stratified splits; every patient appears in exactly 5 test sets overall (and exactly 1 test set in the first 5 splits). |
| **Split counts (`DEFAULT_EVAL_SPLITS = 25`, `QUICK_EVAL_SPLITS = 5`, `EXPERIMENT_EVAL_SPLITS = 5`)** | Team decision (Nov 26): the main table uses all 25 splits for all four models (the `evaluate()` default); experiments use the first 5 splits for all models, passed explicitly with `n_splits_to_run=EXPERIMENT_EVAL_SPLITS`; quick mode stays at 5. The measured 8-qubit VQC time is about 7 minutes per split (about 3 h for 25), so the earlier 5-split fallback (based on an old 6–7.8 h estimate) no longer applies. |
| `Pipeline([("prep", PimaPreprocessor), ("clf", model)])` | Putting preprocessing inside the scikit-learn `Pipeline` guarantees both outer test rows AND inner 3-fold validation rows are transformed using only their respective training rows—zero data leakage during `GridSearchCV`. |
| Inner tuning (`3`-fold `GridSearchCV`, `balanced_accuracy`, `<= 8` combos) | Gives every model the same tuning budget and selects hyperparameters using balanced accuracy because Pima is 65.1% negative / 34.9% positive. |
| `train_size` + `fixed_params` + `tune_hyperparameters()` | Supports Experiment 2: draws 10 stratified subsets (`30`, `60`, `120`, `200`) from the `614`-patient pool evaluated on a fixed `20%` (`154`-patient) test set, skipping noisy inner CV on tiny sets by reusing settings tuned once on the `614`-patient pool. |
| `compare_models()` (Nadeau–Bengio corrected t-test) | Standard paired t-test assumes test folds are independent, which overstates confidence in k-fold CV because training sets overlap (`~75%` overlap in 5-fold CV). Nadeau & Bengio (2003) scale the fold-difference variance by `1/J + n_test/n_train` instead of `1/J`. When `p >= 0.05`, we report `"no significant difference found"` (never `"equal"`). |

---

## 4. Official Logistic Regression Results (`results/logreg.csv`, 25 splits, 8 features)

| Metric | Value (`mean ± std`) | Notes |
|---|---|---|
| Always-guess floor | `65.10%` | Guessing `tested_negative` for every patient (`500 / 768`) |
| Accuracy | `75.05% ± 3.74%` | Beats the majority floor by +9.95 percentage points |
| Balanced accuracy | `74.21% ± 3.77%` | Mean of sensitivity and specificity across both classes |
| Macro F1 | `73.32% ± 3.86%` | Unweighted mean of class-0 and class-1 F1 scores |
| Train time / split | `0.2266s ± 0.0053s` | Includes 3-fold `GridSearchCV` + refit on ~614 training patients |

Inner 3-fold CV selected `C=0.3` (7 splits), `C=0.1` (7 splits), `C=1.0` (6 splits), `C=0.03` (2 splits), `C=30.0` (2 splits), and `C=3.0` (1 split), all with fixed `class_weight="balanced"`.

### Paired comparison vs. RBF SVM (`results/rbf_svm.csv`, 25 splits)

Using `compare_models()` (Nadeau–Bengio corrected resampled paired t-test, `df = 24`):

| Metric | LogReg | RBF SVM | Mean diff | `t` | `p` | Result |
|---|---:|---:|---:|---:|---:|---|
| Accuracy | `75.05%` | `74.76%` | `+0.29%` | `+0.1515` | `0.8809` | no significant difference found |
| Balanced accuracy | `74.21%` | `74.99%` | `-0.79%` | `-0.4112` | `0.6846` | no significant difference found |
| Macro F1 | `73.32%` | `73.47%` | `-0.15%` | `-0.0775` | `0.9389` | no significant difference found |

---

## 5. Experiment 1: Number of Features / Qubits (`src/experiment_logreg.py`, LogReg Contribution Only)

> **Scope note:** Covers Person A's Logistic Regression runs (`results/exp1_logreg_q4.csv`, `results/exp1_logreg_q6.csv`, `figures/exp1_logreg.png`, and `notes/EXPERIMENTS_A.md`) across the first 5 stratified splits (`EXPERIMENT_EVAL_SPLITS = 5`), using the first 5 rows of `results/logreg.csv` as the `8`-column baseline.

| Variant | Accuracy (`mean ± std`) | Balanced Acc (`mean ± std`) | Macro F1 (`mean ± std`) | Time / split |
|---|---:|---:|---:|---:|
| `q4` (`4` cols) | `73.42% ± 5.90%` | `72.83% ± 5.21%` | `71.79% ± 5.73%` | `0.2428s ± 0.0216s` |
| `q6` (`6` cols) | `73.30% ± 4.37%` | `72.48% ± 3.23%` | `71.58% ± 3.99%` | `0.2473s ± 0.0170s` |
| `q8` (`results/logreg.csv[:5]`) | `74.60% ± 4.82%` | `74.00% ± 3.83%` | `73.00% ± 4.55%` | `0.2287s ± 0.0047s` |

See `notes/EXPERIMENTS_A.md` for the exact per-split `selected_features` table, `compare_models()` p-values on `balanced_accuracy` (`q4 vs q8`: `p = 0.6735`, `q6 vs q8`: `p = 0.3301`, `"no significant difference found"`), and plain-language notes.

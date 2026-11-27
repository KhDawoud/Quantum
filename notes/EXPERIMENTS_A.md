# EXPERIMENTS_A: Person A (Logistic Regression)

Role: **A** (Logistic Regression baseline)
Frozen model file: `src/model_logreg.py` (fixed `class_weight="balanced"`, `C` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]`, `solver="lbfgs"`, `max_iter=1000`, `random_state=42`)
Main results file (25 splits): `results/logreg.csv`

---

## Step 0: Tool & Frozen-Model Check (2026-11-26)

- **`src/evaluate.py` names verified (read-only):** `evaluate()`, `EXPERIMENT_EVAL_SPLITS` (`5`), `n_splits_to_run`, `n_features`, `train_size`, `fixed_params`, `n_draws`, `get_fixed_test_split()`, `tune_hyperparameters()`, `compare_models()`, `format_summary()`. All 11 match the prompt verbatim.
- **`src/model_logreg.py` frozen check:** `git diff 056b00e -- src/model_logreg.py src/evaluate.py src/data.py results/logreg.csv` is empty. Uses fixed `class_weight="balanced"` and tunes `C` over 8 values.
- **Main baseline (`results/logreg.csv`):** Verified `25` rows; first `5` rows (`split 0..4`, 8 columns) give `74.00% ± 3.83%` balanced accuracy, `74.60% ± 4.82%` accuracy, `73.00% ± 4.55%` macro F1, `0.2287s ± 0.0047s` training time.
- **Plotting:** `matplotlib==3.11.2` installed and verified in `.venv`.

---

## Experiment 1: Number of Measurements (Columns = 4, 6, 8)

### A. Prediction
> **Honest note:** Experiment 1 was initially implemented and run before receiving the prediction-first team prompt, so **no pre-run prediction was recorded before the first run** (`unverified` as a pre-run prediction). Experiment 2 will record Person A's timestamped prediction in this file before running any code.

### B. Setup (what changes vs. what stays the same)
- **What changes:** Number of clinical columns kept (`n_features=4` in `exp1_logreg_q4`, `n_features=6` in `exp1_logreg_q6`), compared against the `8`-column baseline (the first `5` rows of `results/logreg.csv`). At `4` and `6` columns, `PimaPreprocessor` ranks columns using ANOVA F-score (`f_classif`) on training rows only inside each exam.
- **What stays the same:** Identical first `5` stratified splits (`n_splits_to_run=EXPERIMENT_EVAL_SPLITS=5`, splits `0..4`), identical cleaning/clipping/`[0, pi]` scaling inside training rows, frozen `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)`, and the same 8-value `C` grid tuned by 3-fold inner `GridSearchCV` scored on `balanced_accuracy`.

### C. Quick Check & D. Time Estimate
- **Quick check:** Ran `evaluate("exp1_logreg_quick_check", make_model, PARAM_GRID, n_features=4, quick=True, n_splits_to_run=EXPERIMENT_EVAL_SPLITS)` (5 splits, reduced grid). Completed cleanly and deleted `results/exp1_logreg_quick_check.csv` immediately afterward (never used as results).
- **Time estimate:** Mean fit time in `results/logreg.csv` is `0.2266s` per split. Running `exp1_logreg_q4` (5 splits) + `exp1_logreg_q6` (5 splits) = `10` splits -> estimated **`~2.27 seconds`** total (well below 20 minutes).

### E. Selected Columns in Each Exam (`selected_features`)

| Split | `q4` (top 4 columns) | `q6` (top 6 columns) | `q8` baseline (all 8 columns) |
|---:|---|---|---|
| `0` | `plas,mass,age,insu` | `plas,mass,age,insu,preg,skin` | `preg,plas,pres,skin,insu,mass,pedi,age` |
| `1` | `plas,mass,age,preg` | `plas,mass,age,preg,skin,insu` | `preg,plas,pres,skin,insu,mass,pedi,age` |
| `2` | `plas,mass,age,preg` | `plas,mass,age,preg,pedi,skin` | `preg,plas,pres,skin,insu,mass,pedi,age` |
| `3` | `plas,mass,insu,age` | `plas,mass,insu,age,skin,preg` | `preg,plas,pres,skin,insu,mass,pedi,age` |
| `4` | `plas,mass,age,skin` | `plas,mass,age,skin,insu,preg` | `preg,plas,pres,skin,insu,mass,pedi,age` |

Every split ranked `plas` (glucose), `mass` (BMI), and `age` in the top 3, while the 4th and 6th slots rotated slightly across training folds.

### F. Results Table & Paired Comparisons (`balanced_accuracy`)

| Variant | Output CSV | Accuracy (`mean ± std`) | Balanced Acc (`mean ± std`) | Macro F1 (`mean ± std`) | Time / split |
|---|---|---:|---:|---:|---:|
| **4 columns (`q4`)** | `results/exp1_logreg_q4.csv` | `73.42% ± 5.90%` | `72.83% ± 5.21%` | `71.79% ± 5.73%` | `0.2428s ± 0.0216s` |
| **6 columns (`q6`)** | `results/exp1_logreg_q6.csv` | `73.30% ± 4.37%` | `72.48% ± 3.23%` | `71.58% ± 3.99%` | `0.2473s ± 0.0170s` |
| **8 columns (`q8` baseline)** | `results/logreg.csv` (first 5 rows) | `74.60% ± 4.82%` | `74.00% ± 3.83%` | `73.00% ± 4.55%` | `0.2287s ± 0.0047s` |

*Always-guess floor on full Pima: `65.10%`.*

**Paired `compare_models()` vs. 8-column baseline (Nadeau–Bengio t-test on `balanced_accuracy`, `df = 4`):**

| Comparison | Mean A (`k`) | Mean B (`q8`) | Difference (`A - B`) | Corrected SE | `t` | `p`-value | Plain-language result |
|---|---:|---:|---:|---:|---:|---:|---|
| `q4` vs `q8` | `72.83%` | `74.00%` | `-1.17%` | `0.0259` | `-0.4538` | `0.6735` | **no significant difference found** |
| `q6` vs `q8` | `72.48%` | `74.00%` | `-1.52%` | `0.0137` | `-1.1078` | `0.3301` | **no significant difference found** |

In plain words: dropping from 8 clinical columns to 6 or 4 lowered average balanced accuracy by only ~1.2–1.5 percentage points, which is small compared to how much scores bounce from split to split (`±3.2%` to `±5.2%`), so we found no statistically significant difference (`p = 0.6735` and `p = 0.3301`).

### G. Figure
- Saved at: `figures/exp1_logreg.png` (shows balanced accuracy and accuracy mean ± sample SD across `4`, `6`, and `8` columns, plus the `65.1%` always-guess floor).

### H. What Changed and Why (Plain Language)
Cutting Pima from 8 columns down to 6 or 4 barely changed Logistic Regression's performance (`74.00%` to `72.48%` and `72.83%` balanced accuracy, `p > 0.33`, no significant difference found). *Hypothesis:* most of the linear diabetes signal sits in the top clinical features (`plas`, `mass`, and `age`, which `f_classif` picked on all 5 splits), whereas columns like `insu` (`48.7%` imputed zeros) and `skin` (`29.6%` imputed zeros) add weaker or noisier linear signal once imputed (not directly tested with single-column ablation). Comparing these same `selected_features` against RBF SVM, Quantum Kernel, and VQC will show whether the quantum models gain or lose more when qubit count drops from 8 to 4.

---

## Experiment 2: Number of Training Patients (`30`, `60`, `120`, `200`)

### A. Prediction (Recorded BEFORE Any Run)
- **Date & time:** `2026-11-27 07:14 (+02:00)`
- **Member A's prediction (verbatim):**
  > "I think the average score will jump up early and flatten out as the patients increase, and I think it will be more consistent as patients increase."

### B. Setup (what changes vs. what stays the same)
- **What changes:** The number of training patients (`train_size` = `30`, `60`, `120`, `200`). For each size, `evaluate()` draws `10` different stratified random subsets (`n_draws=10`, seeds derived from `42`) from the 614-patient training pool (without passing `n_splits_to_run`).
- **What stays the same:** One untouched fixed `20%` (`154`-patient) stratified test set via `get_fixed_test_split()`, all `8` clinical columns (`n_features=8`), identical training-only preprocessing inside each draw, frozen `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)`, and one `fixed_params` setting tuned **once** on the 614-patient training pool using `tune_hyperparameters()` with the standard 8-value `C` grid (`3`-fold balanced-accuracy CV).
- **Important tuning-pool disclosure:** Tuning `C` on tiny subsets like `30` patients is too noisy, so `C` is tuned once on the full 614-patient pool and reused at every training size via `fixed_params`. Every model uses this exact same larger-pool tuning shortcut.

### C. Quick Check & D. Time Estimate
- **Quick check:** Ran `evaluate("exp2_logreg_quick_check", make_model, PARAM_GRID, n_features=8, train_size=30, quick=True, fixed_params={'C': 0.1}, n_draws=10)` (3 draws). Completed cleanly and deleted `results/exp2_logreg_quick_check.csv` immediately afterward (never used as results).
- **Time estimate:** From `results/logreg.csv` (`0.2266s` per tuned split), `1` tuning run on the 614-patient pool (`~0.23s`) plus `40` fixed-parameter small fits (`< 1s`) -> estimated **`< 2 seconds`** total (far below 20 minutes).

### E. Results Table (`10` Random Draws per Size on Fixed `154`-Patient Test Set)
- **Tuned settings from 614-patient pool:** `fixed_params = {'C': 0.1}` (with fixed `class_weight="balanced"`).
- **Fixed-test majority floor:** `64.94%` (`100` negative / `54` positive out of `154` test patients; full dataset floor remains `65.10%`).

| Training size | Output CSV | Balanced Acc (`mean ± std`) | Bal. Acc range (`min–max`) | Accuracy (`mean ± std`) | Macro F1 (`mean ± std`) | Time / draw |
|---:|---|---:|---:|---:|---:|---:|
| **`30`** | `results/exp2_logreg_n30.csv` | `69.19% ± 2.05%` | `64.35%–71.46%` | `70.39% ± 1.76%` | `68.37% ± 1.76%` | `0.0060s ± 0.0008s` |
| **`60`** | `results/exp2_logreg_n60.csv` | `69.64% ± 1.84%` | `66.98%–73.04%` | `70.58% ± 1.47%` | `68.77% ± 1.62%` | `0.0068s ± 0.0007s` |
| **`120`** | `results/exp2_logreg_n120.csv` | `70.07% ± 1.06%` | `68.13%–71.41%` | `70.97% ± 1.30%` | `69.18% ± 1.12%` | `0.0085s ± 0.0029s` |
| **`200`** | `results/exp2_logreg_n200.csv` | `69.94% ± 1.84%` | `67.91%–74.11%` | `70.97% ± 1.62%` | `69.13% ± 1.72%` | `0.0071s ± 0.0011s` |

### F. Note on Paired Comparisons
- `compare_models()` is **not** applied in Experiment 2 because the draws at `30`, `60`, `120`, and `200` are random training subsamples evaluated on one shared `154`-patient test set rather than matching k-fold cross-validation exams (the standard deviations above are descriptive spread across the 10 draws, not a significance test).

### G. Figure
- Saved at: `figures/exp2_logreg.png` (balanced accuracy and accuracy mean ± sample SD across `30`, `60`, `120`, and `200` training patients, plus the `64.9%` fixed-test floor).

### H. What Changed and Why (Plain Language)
Logistic Regression reached `69.19% ± 2.05%` balanced accuracy with just `30` training patients and flattened out near `70%` from `60` to `200` patients (`69.64%`, `70.07%`, `69.94%`), matching the "jump early and flatten out" part of the prediction. Spread tightened from `30` to `120` patients (`±2.05%` → `±1.84%` → `±1.06%`), though it bounced back up slightly at `200` (`±1.84%`, range `67.91%–74.11%`) instead of shrinking monotonically—and `30` patients had the widest min–max range (`7.1` percentage points, dipping once to `64.35%`). *Hypothesis:* because Logistic Regression only learns 9 parameters (8 weights + 1 intercept) with strong regularization (`C=0.1` tuned on the 614-patient pool), 30 patients already gets it close to its linear ceiling on this test split, while individual random draws (especially at small `n`, and even one lucky/unlucky draw at `200`) still shift the fitted line slightly (not proven without repeating with multiple fixed test sets). Comparing these numbers with RBF SVM, Quantum Kernel, and VQC will show whether the quantum models need more patients or hold up similarly when training data is scarce.

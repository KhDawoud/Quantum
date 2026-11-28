# Experiment Log — Member C (Quantum Kernel SVM)

## Step 0: Tooling & Model Check
- **Model file**: `src/model_qkernel.py` (real filename in repo; note that prompt listed `model_q_kernel.py` as placeholder).
- **Model configuration (frozen)**: `make_model(encoding='zz')` returns `QuantumKernelSVC(encoding='zz', class_weight='balanced', random_state=42)` backed by `SVC(kernel='precomputed', class_weight='balanced')` and `PARAM_GRID = {'C': [0.1, 1.0, 10.0, 100.0], 'gamma': [0.5, 1.0]}` (8 combinations).
- **Shared harness (`src/evaluate.py`) verified**: `evaluate()`, `EXPERIMENT_EVAL_SPLITS` (`5`), `n_splits_to_run`, `n_features`, `train_size`, `fixed_params`, `n_draws`, `get_fixed_test_split()`, `tune_hyperparameters()`, `compare_models()`, and `format_summary()` all exist and match these exact names.
- **Main 25-split baseline (`results/qkernel.csv`)**: 25 data rows (plus header), `73.93% ± 3.50%` accuracy, `73.97% ± 3.50%` balanced accuracy, `72.53% ± 3.50%` macro F1, `0.8274s ± 0.0185s` per split (`20.69s` total).
- **Plotting**: `matplotlib 3.11.2` installed and working.

---

## Experiment 1: Number of Measurements (Columns / Qubits: 4, 6, 8)

### B. Setup
- **What changes**: Number of top clinical columns kept by `PimaPreprocessor(n_features=k)` on training rows only (`k = 4`, `k = 6`), compared against the 8-column baseline (`k = 8`, first 5 rows of `results/qkernel.csv`).
- **What stays the same**: Exact same first 5 stratified splits (`n_splits_to_run=EXPERIMENT_EVAL_SPLITS = 5`, `seed=42`), same ZZ feature map (`2` repetitions), same fixed `class_weight='balanced'`, and same 8-combination `PARAM_GRID` tuned by 3-fold inner `GridSearchCV` on `balanced_accuracy`.
- **Prediction (written before inspecting results)**: Using only the top 4 or 6 features should reduce kernel concentration (keep off-diagonal kernel values more spread out) and run faster, which may let the 4- or 6-qubit kernel match or slightly beat 8 qubits if the weakest columns mostly add noise.

### C & D. Quick Check & Time Estimate
- **Quick check**: Ran `evaluate('exp1_qkernel_q4_quick', ..., n_features=4, quick=True, n_splits_to_run=EXPERIMENT_EVAL_SPLITS)` cleanly (5 splits) and immediately deleted `results/exp1_qkernel_q4_quick.csv`.
- **Time estimate**: From `results/qkernel.csv` (`0.8274s/split` at 8 qubits), running 10 splits (`5` at 4 qubits + `5` at 6 qubits) had an upper-bound estimate of **`8.27s` (`0.14 min`)**, well below 20 minutes.

### E. Results & Selected Columns

| Qubits (`n_features`) | Splits | Accuracy (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Train Time / Split |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **4** (`results/exp1_qkernel_q4.csv`) | 5 | 75.12% ± 5.21% | **75.53% ± 4.58%** | 73.92% ± 5.03% | 0.2057s ± 0.0051s |
| **6** (`results/exp1_qkernel_q6.csv`) | 5 | 72.65% ± 4.97% | **72.77% ± 3.83%** | 71.33% ± 4.59% | 0.3373s ± 0.0072s |
| **8** (`results/qkernel.csv`, rows 0–4) | 5 | 72.00% ± 3.90% | **72.01% ± 2.27%** | 70.57% ± 3.27% | 0.8235s ± 0.0135s |

*(Always-guess majority floor: `65.10%`.)*

**Columns picked by training-only `f_classif` (`selected_features`)**:
- **`n_features=4`**:
  - Split 0: `plas,mass,age,insu`
  - Split 1: `plas,mass,age,preg`
  - Split 2: `plas,mass,age,preg`
  - Split 3: `plas,mass,insu,age`
  - Split 4: `plas,mass,age,skin`
- **`n_features=6`**:
  - Split 0: `plas,mass,age,insu,preg,skin`
  - Split 1: `plas,mass,age,preg,skin,insu`
  - Split 2: `plas,mass,age,preg,pedi,skin`
  - Split 3: `plas,mass,insu,age,skin,preg`
  - Split 4: `plas,mass,age,skin,insu,preg`
- **`n_features=8`**: all 8 columns (`preg,plas,pres,skin,insu,mass,pedi,age`) in every split.

**Measured Kernel Spread (`kernel_stats()` off-diagonal values across the 5 training folds)**:

| Qubits | `gamma = 0.5` Mean ± Std | `gamma = 0.5` Variance ± Std | `gamma = 1.0` Mean ± Std | `gamma = 1.0` Variance ± Std |
| :---: | :---: | :---: | :---: | :---: |
| **4** | 0.6243 ± 0.0296 | 0.0535 ± 0.0029 | 0.2147 ± 0.0142 | 0.0491 ± 0.0053 |
| **6** | 0.5095 ± 0.0122 | 0.0578 ± 0.0023 | 0.1042 ± 0.0089 | 0.0241 ± 0.0027 |
| **8** | 0.4059 ± 0.0051 | 0.0574 ± 0.0010 | 0.0385 ± 0.0011 | 0.0070 ± 0.0003 |

### F. Paired Comparison vs. 8-Column Baseline (`compare_models` on `balanced_accuracy`)

| Comparison | Variant Mean ± Std | 8-Qubit Baseline Mean ± Std | Difference (`variant − baseline`) | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **4 qubits vs. 8 qubits** | 75.53% ± 4.58% | 72.01% ± 2.27% | +3.52% | 0.3673 | **no significant difference found** |
| **6 qubits vs. 8 qubits** | 72.77% ± 3.83% | 72.01% ± 2.27% | +0.76% | 0.8171 | **no significant difference found** |

### G. Figure
- Saved to [`figures/exp1_qkernel.png`](../figures/exp1_qkernel.png) (score with ± std error bars vs. qubits on the left; off-diagonal kernel variance vs. qubits on the right).

### H. Plain-Language Explanation
- On these 5 splits, balanced accuracy rose slightly from **72.01%** (8 qubits) to **75.53%** (4 qubits), and training time sped up **4×** (`0.824s` → `0.206s` per split), though with only 5 splits the paired test found **no significant difference found** (`p = 0.3673` for 4 vs. 8; `p = 0.8171` for 6 vs. 8).
- Our `kernel_stats()` measurements show clear **kernel concentration** at `gamma = 1.0`: as qubits grow from 4 to 8, average off-diagonal similarity falls from **0.215** to **0.039** and kernel variance shrinks **7.0×** (`0.0491` → `0.0070`), whereas `gamma = 0.5` keeps variance steady around `0.053–0.058`.
- **Hypothesis** (not directly tested in this single run): dropping weaker features (`pres`, `pedi`) and concentrating on the top medical predictors (`plas`, `mass`, `age`) alongside a less concentrated 4-qubit kernel may give the SVM a cleaner margin, though more splits would be needed to confirm whether that `+3.52%` gain is real or split-to-split noise.

---

## Experiment 2: Number of Training Patients (`30, 60, 120, 200`)

### B. Setup
- **What changes**: Training sample size per draw (`train_size` = `30`, `60`, `120`, `200`, with `10` random stratified draws each), plus a full training-pool (`614`-patient) reference.
- **What stays the same**: Same fixed 20% stratified test set (`154` patients from `get_fixed_test_split()`, seed `42`, majority floor = `64.94%`), all `8` columns (`8` qubits), same ZZ feature map (`2` repetitions), fixed `class_weight='balanced'`, and fixed settings `{'C': 1.0, 'gamma': 0.5}` tuned **once** on the `614`-patient training pool (`tune_hyperparameters()`).
- **Important disclosure (applies to every model)**: Tuning used the larger `614`-patient training pool and was reused at every size via `fixed_params={'C': 1.0, 'gamma': 0.5}`, because tuning separately on `30` patients is too noisy.
- **Prediction (written before inspecting results)**: Accuracy and balanced accuracy should start near the `64.94%` majority floor with large draw-to-draw spread at `30` patients, then rise steadily and tighten in spread as training data grows toward `200` and `614`.

### C & D. Quick Check & Time Estimate
- **Quick check**: Ran `evaluate('exp2_qkernel_n30_quick', ..., train_size=30, fixed_params={'C': 1.0, 'gamma': 0.5}, quick=True)` cleanly (`3` draws) and immediately deleted `results/exp2_qkernel_n30_quick.csv`.
- **Time estimate**: From `results/qkernel.csv` (`0.8274s` for a full tuned split), 1 pool tune (`~0.83s`) plus 40 small fixed-param fits (`< 2s`) was estimated at **`~2.8s` (`< 0.1 min`)**, far under 20 minutes.

### E. Results (10 Stratified Draws per Size on Fixed 154-Patient Test Set)

| Training Patients | Draws | Accuracy (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Train Time / Draw |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **30** (`results/exp2_qkernel_n30.csv`) | 10 | 68.83% ± 3.77% | **66.33% ± 4.20%** | 65.97% ± 4.01% | 0.0054s ± 0.0001s |
| **60** (`results/exp2_qkernel_n60.csv`) | 10 | 68.90% ± 2.13% | **67.19% ± 1.96%** | 66.65% ± 2.03% | 0.0070s ± 0.0018s |
| **120** (`results/exp2_qkernel_n120.csv`) | 10 | 68.90% ± 2.46% | **68.34% ± 2.81%** | 67.23% ± 2.57% | 0.0086s ± 0.0001s |
| **200** (`results/exp2_qkernel_n200.csv`) | 10 | 70.52% ± 2.06% | **69.72% ± 1.74%** | 68.78% ± 1.86% | 0.0115s ± 0.0001s |
| **614 (Full Pool Ref.)** | 1 | 71.43% | **72.04%** | 70.30% | 0.0312s |

*(Fixed 154-patient test set majority floor: `64.94%`.)*

### F. Paired Significance Check (`compare_models()`)
- Experiment 2 does **not** use the 5 outer CV exams—each size evaluates 10 random training draws against the single fixed 154-patient test set (`n_train` differs per size: `30, 60, 120, 200` vs. `614`), so `compare_models()` does **not** compare these rows against the 5-exam baseline or across different `n_train` sizes (`_resolve_split_size` rejects mismatched `n_train`). Instead, teammates can run `compare_models('results/exp2_qkernel_n30.csv', 'results/exp2_<other>_n30.csv')` at matching sizes once other models generate their per-size CSVs.

### G. Figure
- Saved to [`figures/exp2_qkernel.png`](../figures/exp2_qkernel.png) (shows balanced accuracy and accuracy with ± std spread across the 10 draws, alongside the 614-patient pool reference and 64.94% floor).

### H. Plain-Language Explanation
- As training patients increase from `30` to `200`, average balanced accuracy improves steadily from **66.33%** to **69.72%** (reaching **72.04%** with the full `614`-patient pool), while the draw-to-draw spread drops by more than half from **± 4.20%** at `30` patients down to **± 1.74%** at `200` patients.
- At `30` patients, results are noisy (`± 4.20%` spread) and barely clear the `64.94%` always-guess floor, showing that the quantum kernel does not magically learn diabetes structure from tiny samples alone.
- **Hypothesis** (not directly measured against classical models in this single-model run): because SVM decision margins depend on having enough representative support vectors in `256`-dimensional state space, larger training sets give the quantum kernel a cleaner, more stable boundary just like a classical RBF SVM.

---

## Experiment 3: How Data Enters the Circuit (Angle vs. ZZ Encoding)

### B. Setup
- **What changes**: Feature encoding switched from the 2-repetition entangled ZZ map (`encoding='zz'`) to unentangled simple angle encoding (`encoding='angle'`) via `lambda: make_model(encoding='angle')`.
- **Gate note**: The quantum kernel angle encoding uses **`RY`** per qubit, whereas the VQC uses **`RX`**; for a fidelity quantum kernel (`|<psi(x)|psi(y)>|^2`), `RY` and `RX` produce **identical** similarity scores (`prod_i cos^2(gamma * (x_i - y_i) / 2)`).
- **What stays the same**: Exact same first 5 stratified splits (`n_splits_to_run=EXPERIMENT_EVAL_SPLITS = 5`, `seed=42`), all `8` columns (`8` qubits), fixed `class_weight='balanced'`, and the same 8-combination `PARAM_GRID` (`C` in `[0.1, 1.0, 10.0, 100.0]`, `gamma` in `[0.5, 1.0]`). The ZZ baseline is the first 5 rows of `results/qkernel.csv`.
- **Prediction (written before inspecting results)**: Because unentangled angle encoding avoids the `6.5x` off-diagonal kernel concentration observed in 8-qubit ZZ at `gamma = 1.0`, angle encoding may match or slightly outperform ZZ on Pima while running faster—even though its kernel has no entanglement and is classically computable.

### C & D. Quick Check & Time Estimate
- **Quick check**: Ran `evaluate('exp3_qkernel_angle_quick', lambda: make_model(encoding='angle'), ..., quick=True, n_splits_to_run=5)` cleanly (`5` splits) and immediately deleted `results/exp3_qkernel_angle_quick.csv`.
- **Time estimate**: From `results/qkernel.csv` (`0.8274s/split` for ZZ as an upper bound), 5 splits of Angle was estimated at **`<= 4.14s` (`0.07 min`)**, well under 20 minutes.

### E. Results & Kernel Spread (First 5 Splits, 8 Qubits)

| Encoding | Splits | Accuracy (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Train Time / Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Angle (`RY`, `results/exp3_qkernel_angle.csv`)** | 5 | 73.56% ± 4.09% | **73.82% ± 3.08%** | 72.28% ± 3.79% | 0.2527s ± 0.0043s |
| **ZZ (`2 reps`, `results/qkernel.csv` rows 0–4)** | 5 | 72.00% ± 3.90% | **72.01% ± 2.27%** | 70.57% ± 3.27% | 0.8235s ± 0.0135s |

*(Always-guess majority floor: `65.10%`.)*

**Kernel spread (`kernel_stats()` off-diagonal values across the 5 training splits)**:

| Encoding | `gamma = 0.5` Mean ± Std | `gamma = 0.5` Variance ± Std | `gamma = 1.0` Mean ± Std | `gamma = 1.0` Variance ± Std |
| :--- | :---: | :---: | :---: | :---: |
| **Angle (`RY`)** | 0.6461 ± 0.0051 | 0.0307 ± 0.0006 | 0.2298 ± 0.0053 | 0.0456 ± 0.0010 |
| **ZZ (2 reps)** | 0.4059 ± 0.0051 | 0.0574 ± 0.0010 | 0.0385 ± 0.0011 | 0.0070 ± 0.0003 |

*(Measured check: Angle encoding matched the classical product formula `prod_i cos^2(gamma * (x_i - y_i) / 2)` within `4.885e-15` max difference across all 5 splits.)*

### F. Paired Comparison (`compare_models` on `balanced_accuracy`)

| Comparison | Angle Mean ± Std | ZZ Baseline Mean ± Std | Difference (`angle − zz`) | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Angle vs. ZZ (5 splits)** | 73.82% ± 3.08% | 72.01% ± 2.27% | +1.80% | 0.5006 | **no significant difference found** |

### G. Figure
- Saved to [`figures/exp3_qkernel.png`](../figures/exp3_qkernel.png) (Angle vs. ZZ scores with ± std error bars on the left; split-0 off-diagonal similarity distributions at `gamma = 1.0` on the right).

### H. Plain-Language Explanation
- Unentangled **Angle (`RY`)** encoding scored **73.82%** balanced accuracy vs. **72.01%** for entangled **ZZ** (`+1.80%`, `p = 0.5006` → **no significant difference found**) and trained over **3× faster** (`0.253s` vs. `0.824s` per split).
- Because Angle encoding has no CNOT gates (no entanglement), its kernel is mathematically identical to a simple classical product of cosines (verified to `4.89e-15`), and it avoids ZZ's `6.5×` variance collapse at `gamma = 1.0` (`0.0456` vs. `0.0070`).
- **Hypothesis** (not yet tested on curved artificial data in this model): adding pairwise ZZ entanglement does not help on Pima either because Pima's class boundary is mostly smooth/linear or because the classes overlap too heavily—the positive control (`make_moons`) is needed to confirm ZZ captures curves when they do exist.

---

## Experiment 5: Curved-Data Positive Control (`make_moons`, Artificial Data, 2 Qubits)

> **Data change (2026-11-28):** the moons data was changed from `400` to `768` points (`make_moons(n_samples=768, noise=0.2, random_state=42)` in `make_moons_dataset()`, `src/experiment_qkernel.py`) to match Pima's size, so every model trains on about `614` points per exam, as on Pima. Experiment 5 was rerun with the existing runner; every number below is from the 768-point run. The old 400-point numbers are kept only in the dated AI-use logs.
>
> **Saved reference files (2026-11-28):** `results/exp5_logreg_moons.csv`, `results/exp5_rbf_svm_moons.csv` and `results/exp5_qkernel_angle_moons.csv` were computed by Member D (`src/run_exp_d.py exp5-refs`) with each model's `make_model()` and `PARAM_GRID` imported unchanged, using C's `make_moons_dataset()`, the first 5 exams, `n_features=2` and `zero_as_missing_cols=()`. The saved logistic regression and angle files give the same numbers as the temporary references inside C's runner.

### B. Setup
- **What changes**: Dataset switched from Pima medical records to the **artificial** 2D curved two-moons benchmark `sklearn.datasets.make_moons(n_samples=768, noise=0.2, random_state=42)` (`384` class-0 points, `384` class-1 points, exact `50.00%` majority floor), using `n_features=2` (`2` qubits) and `zero_as_missing_cols=()`. Each exam has `614`-`615` training and `153`-`154` test points.
- **What stays the same**: Same shared harness (`src/evaluate.py`) with `n_splits_to_run=EXPERIMENT_EVAL_SPLITS` (`5` stratified splits, `seed=42`), leakage-free training-only `[0, pi]` rescaling, fixed `class_weight='balanced'`, and the exact same frozen `make_model()` + 8-combination `PARAM_GRID` (`C` in `[0.1, 1.0, 10.0, 100.0]`, `gamma` in `[0.5, 1.0]`). C's runner still computes matching-split references (`LogisticRegression` and `Angle` kernel) in a temporary directory; saved copies now also exist (see the note above).
- **Prediction (written before inspecting results)**: Straight-line Logistic Regression can only draw one flat cut through curved moons, whereas the 2-qubit ZZ quantum kernel can draw a smooth curved boundary and should significantly beat Logistic Regression, proving the quantum kernel detects nonlinear structure when it exists.

### C & D. Quick Check & Time Estimate
- **Quick check**: Ran `evaluate('exp5_qkernel_moons_quick', ..., n_features=2, quick=True, X=X_moons, y=y_moons, zero_as_missing_cols=(), n_splits_to_run=5)` cleanly (`5` splits) and immediately deleted `results/exp5_qkernel_moons_quick.csv`.
- **Time estimate**: From `results/qkernel.csv` (`0.8274s/split` at 8 qubits on 768 rows), 5 splits at 2 qubits on 400 rows had an upper-bound estimate of **`4.14s` (`0.0690 min`)**, far below 20 minutes. *(This estimate was for the original 400-point run. The 768-point rerun of the whole runner, including references, kernel spread and figure, took about 26 s.)*

### E. Results & Kernel Spread (Artificial `make_moons`, 768 points, First 5 Splits, 2 Qubits)

| Model | Splits | Accuracy (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Train Time / Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ZZ Quantum Kernel (`results/exp5_qkernel_moons.csv`)** | 5 | 93.88% ± 1.75% | **93.88% ± 1.74%** | 93.88% ± 1.75% | 0.3399s ± 0.0093s |
| **Angle Quantum Kernel (`results/exp5_qkernel_angle_moons.csv`)** | 5 | 86.46% ± 2.60% | **86.44% ± 2.60%** | 86.42% ± 2.61% | 0.3073s ± 0.0744s |
| **Logistic Regression (`results/exp5_logreg_moons.csv`)** | 5 | 85.68% ± 3.36% | **85.66% ± 3.35%** | 85.65% ± 3.35% | 0.1760s ± 0.0537s |
| **RBF SVM (`results/exp5_rbf_svm_moons.csv`)** | 5 | 96.10% ± 1.71% | **96.10% ± 1.71%** | 96.10% ± 1.71% | 0.1554s ± 0.0061s |

*(Artificial dataset always-guess floor: `50.00%`. Best ZZ Quantum Kernel settings were `{"C": 100.0, "gamma": 0.5}` on all 5 splits, scoring `92.21%`, `93.51%`, `94.16%`, `96.71%`, and `92.81%` balanced accuracy. RBF SVM chose `{"C": 10, "gamma": 1.0}` on all 5 splits. Test sets are not always exactly half-and-half (`76`/`77` in exams 3 and 4), so accuracy and balanced accuracy differ slightly.)*

**Kernel spread (`kernel_stats()` off-diagonal values across the 5 training splits, 2 qubits)**:

| Encoding | `gamma = 0.5` Mean ± Std | `gamma = 0.5` Variance ± Std | `gamma = 1.0` Mean ± Std | `gamma = 1.0` Variance ± Std |
| :--- | :---: | :---: | :---: | :---: |
| **ZZ (2 reps)** | 0.7246 ± 0.0019 | 0.0344 ± 0.0004 | 0.4120 ± 0.0032 | 0.0730 ± 0.0004 |
| **Angle (`RY`)** | 0.8473 ± 0.0019 | 0.0156 ± 0.0004 | 0.5545 ± 0.0037 | 0.0857 ± 0.0011 |

*(Measured check: 2-qubit ZZ kernel matched `hand_zz_2qubit_kernel()` within `2.665e-15` max difference on split 0.)*

### F. Paired Comparisons (`compare_models` on `balanced_accuracy`)

| Comparison (5 Matching `make_moons` Splits) | ZZ Kernel Mean ± Std | Reference Mean ± Std | Difference (`ZZ − ref`) | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ZZ Kernel vs. Logistic Regression (Linear)** | 93.88% ± 1.74% | 85.66% ± 3.35% | +8.21% | 0.0203 | **significant difference found (p < 0.05)** |
| **ZZ Kernel vs. Angle Kernel (`RY`, Unentangled)** | 93.88% ± 1.74% | 86.44% ± 2.60% | +7.44% | 0.0099 | **significant difference found (p < 0.05)** |
| **ZZ Kernel vs. RBF SVM (classical, curved)** | 93.88% ± 1.74% | 96.10% ± 1.71% | -2.22% | 0.1887 | no significant difference found |
| *(RBF SVM vs. Logistic Regression)* | *96.10% ± 1.71%* | *85.66% ± 3.35%* | *+10.43% (RBF − LogReg)* | *0.0113* | *significant difference found (p < 0.05)* |

### G. Figure
- Saved to [`figures/exp5_qkernel.png`](../figures/exp5_qkernel.png) (left panel: split-0 curved ZZ vs. straight-line Logistic Regression decision boundaries on the artificial `make_moons` points; right panel: 5-split balanced accuracy ± std for Logistic Regression, Angle kernel and ZZ kernel). Regenerated on 2026-11-28 from the 768-point run.
- The RBF SVM is **not** in this figure: adding it would need a change to the plotting function's inputs and to the runner, not just one extra bar, so the figure was left as C's code draws it. The RBF SVM numbers are in the tables above.

### H. Plain-Language Explanation
- On curved **artificial** `make_moons` data (`n = 768`, `noise = 0.2`), the frozen 2-qubit ZZ quantum kernel reached **93.88%** balanced accuracy, significantly beating both straight-line **Logistic Regression** (`85.66%`, `+8.21%`, `p = 0.0203`) and unentangled **Angle (`RY`)** encoding (`86.44%`, `+7.44%`, `p = 0.0099`).
- A classical RBF SVM, which also draws curved boundaries, scored `96.10% ± 1.71%`, `2.22` points above the ZZ kernel; no significant difference found (`p = 0.1887`, 5 exams), so on this data the quantum kernel matches a good classical curved model rather than beating it.
- Unlike Experiment 3 on 8-qubit Pima (where ZZ and Angle showed no significant difference), the pairwise `x0 * x1` ZZ interaction clearly helps here because the two crescent moons genuinely curve around each other, and 2 qubits avoid high-qubit kernel concentration (`var = 0.034–0.073`).
- **Interpretation vs. Hypothesis**: This positive control proves our `QuantumKernelSVC` circuit and evaluation pipeline can learn strong curved boundaries cleanly; our **hypothesis** for why ZZ did not beat Logistic Regression on Pima (`73.97%` vs. `74.20%`) is that real Pima diabetes features have a mostly smooth, overlapping boundary rather than sharp interlocking curves.

---

## Follow-Up 25-Split Run: Simple Angle (`RY`) Encoding (`results/qkernel_angle.csv`)

> **Disclosure:** `results/qkernel.csv` (ZZ, 25 splits) remains the planned primary quantum kernel run. `results/qkernel_angle.csv` is an additional 25-split comparison run requested afterwards to compare unentangled `RY` angle encoding at full 25-split power against all four main models.

### Setup & Reproducibility Check
- **Configuration**: Frozen `make_model(encoding='angle')` (`RY(gamma * x_i)` per qubit, no entanglement), `n_features=8`, `class_weight='balanced'`, exact same 8-combination `PARAM_GRID`, and all `25` standard Pima splits (`DEFAULT_EVAL_SPLITS = 25`, `seed = 42`) saved to `results/qkernel_angle.csv`.
- **Reproducibility check**: First 5 rows of `results/qkernel_angle.csv` (excluding `train_time_sec`) match `results/exp3_qkernel_angle.csv` identically.

### 25-Split Metrics (`results/qkernel_angle.csv`)

| Model | Splits | Accuracy (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Train Time / Split |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Angle Quantum Kernel (`RY`, `results/qkernel_angle.csv`)** | 25 | 74.42% ± 4.06% | **74.68% ± 3.99%** | 73.15% ± 4.08% | 0.2607s ± 0.0090s |
| **ZZ Quantum Kernel (planned main, `results/qkernel.csv`)** | 25 | 73.93% ± 3.50% | **73.97% ± 3.50%** | 72.53% ± 3.50% | 0.8274s ± 0.0185s |

*(Always-guess majority floor: `65.10%`. Inner 3-fold CV picked `{"C": 1.0, "gamma": 0.5}` on `12` splits, `{"C": 10.0, "gamma": 0.5}` on `8` splits, `{"C": 1.0, "gamma": 1.0}` on `4` splits, and `{"C": 0.1, "gamma": 0.5}` on `1` split.)*

### Paired Comparisons vs. All Main 25-Split Models (`compare_models` on `balanced_accuracy`)

| Comparison (25 Matching Pima Splits) | Angle Kernel | Comparison Model | Difference (`Angle − model`) | `t`-stat | `p`-value | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **vs. ZZ Quantum Kernel (`results/qkernel.csv`)** | 74.68% | 73.97% | +0.71% | +0.4213 | 0.6773 | **no significant difference found** |
| **vs. Logistic Regression (`results/logreg.csv`)** | 74.68% | 74.21% | +0.47% | +0.2745 | 0.7860 | **no significant difference found** |
| **vs. RBF SVM (`results/rbf_svm.csv`)** | 74.68% | 74.99% | -0.31% | -0.3745 | 0.7113 | **no significant difference found** |
| **vs. VQC ZZ (`results/vqc_zz_25.csv`)** | 74.68% | 63.07% | +11.61% | +3.7225 | 0.0011 | **significant difference found (p < 0.05)** |

### Plain-Language Takeaway
- Across all `25` Pima splits, unentangled **Angle (`RY`)** encoding reached **74.68%** balanced accuracy (`0.2607s/split`, `3.2x` faster than ZZ), landing statistically indistinguishable from both entangled **ZZ** (`+0.71%`, `p = 0.6773` → **no significant difference found**) and the classical baselines (`p = 0.7860` vs. Logistic Regression; `p = 0.7113` vs. RBF SVM).
- Because this Angle kernel has zero entanglement and equals a closed-form classical product of cosines, getting essentially the same `~74–75%` score as ZZ and RBF SVM reinforces that ZZ entanglement adds no measurable advantage on Pima diabetes data.

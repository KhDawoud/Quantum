# DESIGN_C: design choices for Person C (Quantum Kernel SVM)

Scope: `src/model_qkernel.py` (frozen quantum fidelity kernel classifier), `src/experiment_qkernel.py` (standalone Experiments 1, 2, 3, and 5), `src/experiments.py` (Experiment 3 integration and `notes/REVIEW_C.md` bug fixes), `tests/test_qkernel.py`, `results/qkernel.csv`, `results/exp{1,2,3,5}_qkernel_*.csv`, `figures/exp{1,2,3,5}_qkernel.png`, and `notes/EXPERIMENTS_C.md`.

---

## 1. What the model does, in plain language

1. **Encode**: turns each patient's `[0, pi]` numbers into a quantum state `|psi(x)>` on `n` qubits (`8` qubits for main Pima; `4`, `6`, or `2` qubits in Experiments 1 and 5).
2. **Measure similarity (fidelity kernel)**: computes how closely every pair of patients' quantum states overlap: `K[i, j] = |<psi(x_i) | psi(x_j)>|^2`, always between `0` (completely different) and `1` (identical).
3. **Draw the boundary**: feeds that similarity table into scikit-learn's `SVC(kernel='precomputed')`, which solves a standard convex optimization problem to separate healthy vs. diabetic patients cleanly.

---

## 2. Design choices and why

| Choice | Why |
|---|---|
| `default.qubit` + batched `qml.state()` (`compute_states()`) | Simulates every patient row in one vectorized pass and builds the full Gram matrix `K = |S @ S^H|^2` with matrix multiplication, running all `25` tuned outer splits in `~20.7s` total. |
| ZZ-style feature map (`encoding='zz'`, `reps=2`) | Matches `PROJECT_CONTEXT.md` and Member D's VQC: Hadamard + `RZ(gamma * x_i)` on every qubit, then linear nearest-neighbour pairs `(i, i+1)` with no wrap-around (`CNOT(i, i+1)`, `RZ((gamma * x_i) * (gamma * x_(i+1)))`, `CNOT(i, i+1)`). |
| Angle encoding (`encoding='angle'`, `RY`) | Unentangled baseline using one `RY(gamma * x_i)` rotation per qubit. Note vs. Member D's VQC (`RX`): for a fidelity kernel `|<psi(x)|psi(y)>|^2`, `RY` and `RX` give identical similarity scores (`prod_i cos^2(gamma * (x_i - y_i) / 2)`), verified within `4.885e-15`. |
| Fixed balanced class weighting (`class_weight='balanced'`) | All models use balanced class weighting, fixed and not tuned, because the classes are uneven and we evaluate on balanced accuracy. `make_model()` sets `class_weight='balanced'` inside `SVC(kernel='precomputed')`, computing `n_samples / (2 * class_count)` from training labels in `fit()` (~`0.77` healthy, ~`1.43` diabetic on full Pima) instead of spending tuning budget on weights. |
| `PARAM_GRID`: `C in [0.1, 1.0, 10.0, 100.0]`, `gamma in [0.5, 1.0]` | `8` combinations (fits the fair `<= 8` budget); `_validate_circuit_config()` enforces `0 < gamma <= 1.0` and `1 <= n_qubits <= 10` so single-qubit rotations stay inside `[0, pi]`. |
| Gram matrix polishing (`polish=True`) | Symmetrizes `0.5 * (K + K.T)`, clips tiny numerical overshoot to `[0, 1]`, and sets the training diagonal to `1.0` before `SVC`; sanity checks verify raw unpolished matrices. |
| Zero test-data leakage | `fit()` stores only training statevectors (`train_states_`); `_test_kernel()` evaluates test-vs-train overlaps without modifying training state; all splits, preprocessing, and 3-fold inner `GridSearchCV` run through `src/evaluate.py`. |
| Kernel concentration check (`kernel_stats()`) | Measures off-diagonal mean, variance, std, min, and max so Experiments 1, 3, and 5 quantify when high-dimensional quantum states begin concentrating near zero overlap. |
| `notes/REVIEW_C.md` fixes on merge | Preserved Member B's `run_exp2_trainsize_rbf` import in `src/experiments.py`, added `"train_time_sec"` to `_extract_metric_stats()`, and passed `n_splits_to_run=EXPERIMENT_EVAL_SPLITS` explicitly in `run_exp3_encoding_kernel()`. |

---

## 3. Official 25-split main results (`results/qkernel.csv`, 8 qubits)

| Metric | Value (`mean ± std`) | Notes |
|---|---|---|
| Always-guess floor | `65.10%` | Guessing `tested_negative` for every patient (`500 / 768`) |
| Accuracy | `73.93% ± 3.50%` | Beats majority floor by `+8.83` percentage points |
| Balanced accuracy | `73.97% ± 3.50%` | Mean of sensitivity and specificity across both classes |
| Macro F1 | `72.53% ± 3.50%` | Unweighted mean of class-0 and class-1 F1 scores |
| Train time / split | `0.8274s ± 0.0185s` | `20.69s` total across all 25 splits (tuning + refit) |

- **Selected settings across all 25 splits**: 3-fold inner CV picked **`gamma = 0.5` on every single split** (`{"C": 1.0, "gamma": 0.5}` in `21` splits, `{"C": 10.0, "gamma": 0.5}` in `3` splits, `{"C": 0.1, "gamma": 0.5}` in `1` split). This matches Experiment 1's finding that `gamma = 0.5` avoids both `x_i * x_(i+1) > pi` angle wrapping and the `7.0x` off-diagonal variance collapse measured at `gamma = 1.0`.

### Paired comparison vs. teammates (`compare_models()` on `balanced_accuracy`, 25 splits)

| Comparison | Quantum Kernel | Teammate Model | Mean Diff | `t` | `p`-value | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| vs. Logistic Regression (`results/logreg.csv`) | `73.97%` | `74.21%` | `-0.23%` | `-0.1374` | `0.8919` | **no significant difference found** |
| vs. RBF SVM (`results/rbf_svm.csv`) | `73.97%` | `74.99%` | `-1.02%` | `-0.6065` | `0.5499` | **no significant difference found** |
| vs. VQC ZZ (`results/vqc_zz_25.csv`) | `73.97%` | `63.07%` | `+10.90%` | `+3.8848` | `0.0007` | **significant difference found (p < 0.05)** |

---

## 4. Summary of standalone experiments

Full setups, predictions, per-split selected features, kernel spread tables, p-values, and plots are recorded in `notes/EXPERIMENTS_C.md`.

- **Experiment 1 — Qubits / Columns (`4`, `6`, `8`; first 5 splits)**: Balanced accuracy was `75.53% ± 4.58%` (`4` qubits), `72.77% ± 3.83%` (`6` qubits), and `72.01% ± 2.27%` (`8` qubits baseline). Paired tests vs. `8` qubits showed **no significant difference found** (`p = 0.3673` for `4q`, `p = 0.8171` for `6q`). Off-diagonal kernel variance at `gamma = 1.0` shrank **`7.0x`** from `0.0491` (`4q`) to `0.0070` (`8q`).
- **Experiment 2 — Training Size (`30`, `60`, `120`, `200`; 10 draws each on fixed 154-patient test set)**: Tuned once on the `614`-patient pool (`fixed_params={'C': 1.0, 'gamma': 0.5}`). Balanced accuracy rose steadily from `66.33% ± 4.20%` (`n=30`) -> `67.19% ± 1.96%` (`n=60`) -> `68.34% ± 2.81%` (`n=120`) -> `69.72% ± 1.74%` (`n=200`) -> `72.04%` (`n=614` ref.), while draw-to-draw spread tightened by more than half.
- **Experiment 3 — Encoding (`Angle RY` vs. `ZZ`; first 5 splits, 8 qubits)**: Angle scored `73.82% ± 3.08%` balanced accuracy vs. ZZ `72.01% ± 2.27%` (`+1.80%`, `p = 0.5006` -> **no significant difference found**), while running `>3x` faster (`0.2527s` vs. `0.8235s` per split) and avoiding ZZ's `6.5x` variance collapse at `gamma = 1.0`.
  - *Follow-up 25-split Angle run (`results/qkernel_angle.csv`, additional comparison run while `results/qkernel.csv` stays the planned primary run)*: `74.42% ± 4.06%` accuracy, **`74.68% ± 3.99%`** balanced accuracy, `73.15% ± 4.08%` macro F1 (`0.2607s/split`). On all `25` splits, `compare_models()` found **no significant difference found** vs. ZZ (`+0.71%`, `p = 0.6773`), Logistic Regression (`+0.47%`, `p = 0.7860`), or RBF SVM (`-0.31%`, `p = 0.7113`), and a **significant difference found (`p < 0.05`)** vs. VQC ZZ (`+11.61%`, `p = 0.0011`).
- **Experiment 5 — Artificial `make_moons` Positive Control (`n=400`, `noise=0.2`, 2 qubits, 5 splits)**: ZZ Quantum Kernel reached **`93.25% ± 3.38%`** balanced accuracy, significantly outperforming straight-line **Logistic Regression** (`83.50% ± 2.05%`, `+9.75%`, `p = 0.0095`) and unentangled **Angle (`RY`)** (`87.25% ± 3.58%`, `+6.00%`, `p = 0.0299`). This verifies our ZZ kernel captures curved structure cleanly when present, supporting the **hypothesis** that Pima did not benefit from ZZ entanglement because its class boundary is smoother and more heavily overlapping.

---

## 5. Validation, caveats & environment

- **Test suite**: `pytest tests/test_qkernel.py tests/test_data.py tests/test_evaluate.py -v` passed all **`42`** tests (`19` in `test_qkernel.py`, covering balanced class weights, 3-qubit circuit structure, 2-qubit hand ZZ formula `< 1e-12`, 8-qubit classical angle formula `< 1e-12`, PSD kernel checks, scikit-learn cloning/`Pipeline`/`GridSearchCV`, and mocked runner integration).
- **Exact formula checks on real runs**: 8-qubit Angle kernel matched `classical_angle_kernel()` within `4.885e-15` across Exp 3 splits; 2-qubit ZZ kernel matched `hand_zz_2qubit_kernel()` within `2.665e-15` on `make_moons` split 0.
- **Caveat (ZZ pair angle range)**: At `gamma = 1.0`, pairwise angles `x_i * x_(i+1)` can reach `pi^2 ≈ 9.87` radians (beyond `pi`), consistent with why inner CV selected `gamma = 0.5` on every Pima split and every `make_moons` split.
- **Environment**: Python `3.14.4`, `pennylane==0.45.1`, `scikit-learn==1.9.1`, `numpy==2.5.5`, `scipy==1.18.1`, `pandas==3.1.1`, `matplotlib==3.11.2`.
- **Adapted sources**: PennyLane `default.qubit`, `qml.state`, and `qml.RY` docs (`https://docs.pennylane.ai/`); IBM Quantum Learning QML course (`https://learning.quantum.ibm.com/course/quantum-machine-learning`); scikit-learn `SVC(kernel='precomputed')` and `BaseEstimator`/`ClassifierMixin` docs.

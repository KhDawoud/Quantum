# Review of teammate C's quantum kernel work (read-only)

> **Update (Nov 26, later):** the team switched the main table to 25 splits (`evaluate()` default is now 25) and experiments to the first 5 splits, passed explicitly with `n_splits_to_run=EXPERIMENT_EVAL_SPLITS`. Where this review says "the team rule is 5 splits" or "the default is 5", that is no longer true. C's `run_exp3_encoding_kernel` (`src/experiments.py` lines 105 and 112 on `feat/qkernel`) calls `evaluate()` without `n_splits_to_run`, so after merging `main` it would run 25 splits; C should add `n_splits_to_run=EXPERIMENT_EVAL_SPLITS` to both calls.

Reviewed: `origin/feat/qkernel`, one commit **919a9eb** by Khaled Dawoud, 2026-11-26 18:23 +0200, message "quantum kernel implementation and results". Based on the initial commit `2076017`, so it does **not** contain `main`'s later work (logistic regression, the real `evaluate.py`).
Method: `git fetch --all`, `git show`, a temporary worktree outside the working folder (removed), and C's own tests run once. Nothing merged, committed or pushed; the working folder stayed on `vqc`.

## 1. Files C added or changed (vs the initial commit)
- `src/model_qkernel.py` (skeleton -> 524 lines), `src/experiments.py` (+215 lines, only Experiment 3 for the kernel is implemented; the rest still raise `NotImplementedError`), `tests/test_qkernel.py` (new, 269 lines).
- **Not changed:** `src/evaluate.py`, `src/data.py`.
- **No `results/` files and no figures are on the branch**, although the commit message says "and results" (unverified why: maybe never run, maybe forgotten with `git add`; `.gitignore` does not hide `results/`).

## 2. Encodings
- **ZZ map matches** PROJECT_CONTEXT.md and my VQC: 2 repetitions; Hadamard then RZ(x_i) on every qubit; then for pairs (i, i+1), no wrap-around: CNOT, RZ(x_i * x_(i+1)) on qubit i+1, CNOT (literal gates, `model_qkernel.py`, `_apply_feature_map`). Mine uses the equivalent `IsingZZ` gate. C's loop applies H and RZ qubit by qubit instead of all H first; for different qubits that is the same circuit. A test compares a 2-qubit hand-written calculation with the circuit.
- **Difference: angle encoding uses `RY(x_i)` per qubit; mine uses `RX(x_i)`.** For the kernel this changes nothing (the squared overlap is the same, `prod cos^2((a-b)/2)`, and C checks it against that formula). It is just a naming difference to state in the report. PROJECT_CONTEXT does not say which rotation.
- Extra: C multiplies inputs by `gamma` before encoding, so the pair angle becomes `gamma^2 * x_i * x_(i+1)`. As specified, the pair angle can reach pi^2 (about 9.9) at gamma 1, beyond pi; the VQC notes already flag this.

## 3. The kernel
- Yes, a fidelity kernel: `K[i,j] = |<psi_i|psi_j>|^2`, computed from state vectors (`qml.state`, all rows in one batched call, then a matrix product), fed to `SVC(kernel='precomputed')`.
- Training states come from training rows only in `fit`; test rows are turned into states only in `predict` and compared with the stored training states. Test rows never enter the training kernel.
- Small cosmetic step: the training matrix is symmetrised, clipped to [0, 1] and its diagonal set to exactly 1 ("polish"). Harmless (differences are around 1e-16); the checks below use the unpolished matrix.

## 4. Settings
```
PARAM_GRID = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.5, 1.0]}
```
- gamma values are 0.5 and 1.0, and the code rejects any gamma above 1.0 (`ValueError`). Good.
- 4 x 2 = **8 combinations**, exactly at the limit of 8. Note: logistic regression also tunes `class_weight` (balanced or not); the kernel model does not. The weighting could favour logistic regression on balanced accuracy, so mention it when comparing.

## 5. Shared code
- C's runner (`run_exp3_encoding_kernel`) calls the shared `evaluate()` and `compare_models()` from `src.evaluate`, not a custom loop. Good.
- C did **not** pass `n_splits_to_run`, so it relies on the default of the evaluate version used. With `main`'s evaluate the default is **5 splits**. Never run in this review (no results exist), so the actual split count is unverified.
- **Two different `evaluate.py` files now exist:** `main`'s (added with the logistic regression commit: pipeline-based, 5 splits default, `n_splits_to_run`, own data via `X=`/`y=`, columns `macro_f1`, `train_time_sec`, returns `p_value`) and my `evaluate` branch version (columns `f1_macro`, `fit_seconds`, returns `p`). The existing `results/logreg.csv` and `results/vqc_zz.csv` follow `main`'s names. C's runner also expects `main`'s style (`macro_f1`, `p_value`), so **`main`'s version is the one to keep**; mine should not be merged as it would break C's code and column names.

## 6. Results
- C has **none** on the branch, so no row counts, scores, times, winning settings or split-id matching can be reported (unverified, nothing to check).
- For reference, the files that do exist: `results/logreg.csv` (main, 5 splits, balanced accuracy about 0.739) and `results/vqc_zz.csv` (vqc branch, 5 splits, balanced accuracy 0.58-0.65, about 400-490 s per split). Both have split ids 0-4 and n_train 614, 614, 614, 615, 615 (n_test 154, 154, 154, 153, 153).

## 7. Planned extras: `kernel_stats`, sanity checks, tests
- `kernel_stats(K)` exists: mean, variance, std, min, max of the off-diagonal entries (square matrix required).
- `run_sanity_checks()` checks (on unpolished matrices): symmetric (error < 1e-12), diagonal equal to 1 (< 1e-12), smallest eigenvalue not below -1e-10, the 2-qubit ZZ kernel against an independent hand formula, and the angle kernel against its closed formula. Weak point: done on 12-15 random points, not on real Pima rows.
- `tests/test_qkernel.py`: 18 test cases (grid contract, 3-qubit circuit structure for both encodings, hand calculation for reps 1 and 2 and gamma 0.5 and 1.0, classical-formula check, matrix properties, `kernel_stats`, validation errors, scikit-learn clone/GridSearchCV/Pipeline, "no mutation of stored training states", sanity checks, and the exp-3 runner with a mocked `evaluate`).
- **I ran them, in a temporary worktree with your `.venv`: 18 passed in 5.5 s (about 7 s with start-up).** The leakage-related test only checks that predicting does not change stored training data and that single-row and batch predictions agree. It is not a test of the evaluation harness.
- Note: your `.venv` lacked `matplotlib` (listed in `requirements.txt`), so I installed `matplotlib==3.11.2` there. Nothing else changed.

## 8. Anything stopping a 25-split rerun with the shared `evaluate()`?
- **No technical blocker.** The model follows the contract (`make_model()`, `fit`, `predict`, `PARAM_GRID` <= 8, gamma <= 1). Runtime is small: C's own benchmark (two 8-qubit fits, random data) gives about 0.2-0.25 s per fit, projecting about 24 s for 5 splits and about 2 minutes for 25 splits (estimate, not a full run).
- Things to fix first:
  1. **Bug in C's runner:** `_extract_metric_stats` looks for `train_time`/`training_time`/`fit_time`, but `main`'s evaluate writes `train_time_sec`. I tested it on `results/logreg.csv`: the time columns silently come out empty (NaN) in `exp3_encoding_kernel.csv`. Fix: add `"train_time_sec"` to both candidate lists.
  2. Use `n_splits_to_run` explicitly so the split count is visible. The team rule is one split count for all models (5, because of the VQC), so keep 5 and do not mix.
  3. Experiments 1, 2, 4, 5 are still not implemented in `experiments.py`.
  4. C's branch lacks `main`'s `evaluate.py` and logistic regression; merging `main` into it is needed before running (C's branch is based on the initial commit).

## 9. Paired test against logreg and VQC
- **Could not do:** C has no results file, so there are no split ids to match. Once C runs `evaluate("qkernel_zz", ...)` with `main`'s evaluate, the check is quick: `compare_models("results/qkernel_zz.csv", "results/logreg.csv")` and the same against `vqc_zz.csv`. A p-value above 0.05 would mean "no significant difference found" (not "equal"); with only 5 splits the test has little power, so expect large p-values unless gaps are big. I can run this if you want (about 25 s of computing, written to a scratch folder).

## Summary table

| Check | What you found | Evidence | Verdict |
|---|---|---|---|
| Branch | `origin/feat/qkernel`, 1 commit by Khaled Dawoud, 18:23, based on the initial commit | `git log --all` | Fine (needs merge of `main`) |
| 1 Files | 3 files changed (model, experiments, tests); evaluate.py and data.py untouched; **no results/figures committed** | `git diff --stat` | Needs fixing (results missing) |
| 2 ZZ map | Matches spec and my VQC | `_apply_feature_map`; hand-formula test | Fine |
| 2 Angle map | RY per qubit vs my RX; same kernel | `_apply_feature_map`; classical-formula test | Fine (document the difference) |
| 3 Kernel | Fidelity kernel from state vectors, precomputed SVC, train states from training rows only | `fit`, `_test_kernel` | Fine |
| 4 Settings | C 4 values x gamma {0.5, 1.0} = 8; gamma > 1 rejected; no `class_weight` tuned | `PARAM_GRID`, `_validate_circuit_config` | Fine (note fairness vs logreg) |
| 5 Shared code | Uses shared evaluate/compare, none changed; split count relies on default | `experiments.py` | Fine (set `n_splits_to_run`) |
| 5 Two evaluate versions | `main`'s and my `evaluate` branch differ in column and key names | `git show origin/main:src/evaluate.py` | Needs fixing (keep `main`'s) |
| 6 Results | None on the branch | `git ls-tree` | Could not tell |
| 7 Extras | `kernel_stats`, sanity checks, 18 tests exist; tests pass in 5.5 s | pytest run | Fine |
| 8 Rerun on 25 splits | No blocker; **runner drops training time (`train_time_sec` not recognised)**; about 2 min estimated for 25 splits | test of `_extract_metric_stats` | Needs fixing (small) |
| 9 compare_models | Not possible: no C results | n/a | Could not tell (unverified) |

## Cleanup
Temporary worktree removed (git had already unregistered it; an empty leftover folder `c-review` was deleted); `git worktree list` shows only the main folder. Scratch files deleted. No `results/` files were created. `git fetch --all` added the remote-tracking branches `origin/feat/qkernel` and `origin/evaluate-claude`. Installed `matplotlib==3.11.2` into `.venv`.

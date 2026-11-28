# MOONS_CHECK: has anyone run experiment 5 (the moons control)?

Checked 2026-11-28, read-only (git commands only; no model, no experiment, no merge, no branch switch). Everything below comes from `git fetch --all --prune` (nothing new came in), `git ls-tree`, `git grep`, `git show` and `git log` on every branch. Anything I could not confirm is marked **unverified**.

`git status` before starting: clean (nothing uncommitted).

## 1. Branches (author, date, last commit)

| Branch | Last commit author | Date | Message |
|---|---|---|---|
| main (= origin/main) | hamza hamza | 2026-11-28 08:35 | Complete RBF Experiment 1 and include it in the combined notebook |
| origin/feat/qkernel | Khaled Dawoud | 2026-11-26 19:12 | Balanced weightings implemented |
| vqc (local) | Yousef Khater | 2026-11-26 19:02 | VQC 5-split main run, design notes, C review |
| origin/vqc | Yousef Khater | 2026-11-26 18:22 | VQC main run: ZZ, 2 layers, 5 splits |
| evaluate-claude / origin/evaluate-claude | Yousef Khater | 2026-11-26 17:03 | Add alternative evaluate.py (Claude Code), tests and review notes |

## 2. Where "moons" / "exp5" appears

Only `main` has any moons run. `feat/qkernel`, `vqc`, `origin/vqc` and both `evaluate-claude` branches have the word "moons" only in the original `PROJECT_CONTEXT.md`, the `src/experiments.py` stub, and unit tests; no moons results, scripts or figures.

| Branch | Path | What it is | Who / when |
|---|---|---|---|
| main | `results/exp5_qkernel_moons.csv` | **C's ZZ quantum kernel moons result (5 rows)** | Khaled Dawoud, added in merge `9422e46`, 2026-11-26 23:46 (one of `9422e46` / `8f4fb06` for the other files below; the per-file commit is **unverified**) |
| main | `figures/exp5_qkernel.png` | C's figure (curved boundary + bars) | same commits |
| main | `src/experiment_qkernel.py` (`make_moons_dataset`, `run_exp5_qkernel_moons`, CLI `exp5`) | C's own data and runner | Khaled Dawoud, `9422e46` / `8f4fb06` (2026-11-27 01:22) |
| main | `notes/EXPERIMENTS_C.md` (section "Experiment 5"), `notes/DESIGN_C.md` line 61, `notes/AI_USE_C.md` line 19 | C's notes | same |
| main | `notebooks/main.ipynb` (section 6, "Experiment 5") | recomputes the moons table from C's CSV plus two temporary references | committed by mohamedgreida (`e11fd45`, 2026-11-27 07:48) and hamza (`63b5fa0`, 2026-11-28 08:35); who wrote section 6 is **unverified** (the text says Khaled ran it) |
| main | `notes/AI_USE_A.md` (Session 5) | A's log: A **did** run a logreg moons experiment (`results/exp5_moons_logreg.csv`, 83.50% +/- 2.05%), then **deleted it at A's request** (Session 6) | mohamedgreida |
| main | `src/experiments.py` line 269 `run_exp5_moons()` | still `raise NotImplementedError` (the original stub) | Khaled Dawoud, initial commit |
| main | `tests/test_data.py`, `tests/test_evaluate.py`, `tests/test_logreg.py` | unit tests using `make_moons(n_samples=200, noise=0.2)` to check the code works; not experiment results | various |
| main | `tests/test_rbf_svm.py` line 89 | unit test with `make_moons(n_samples=300, noise=0.12)`; not an experiment | hamza |

Not found anywhere: `src/moons_data.py`, any `run_exp5_*` script, `results/experiments/` moons files, `results/exp5_moons.csv`, `figures/exp5_moons.png`, or any moons note by B (hamza) or D (Yousef). A's `results/exp5_moons_logreg.csv` was never committed (no trace in `git log --all`), so it exists only, if at all, on A's own computer (**unverified**).

## 3. The one results file

`results/exp5_qkernel_moons.csv` belongs to **C, ZZ quantum kernel**.

| Item | Value |
|---|---|
| Rows / split ids | 5 / 0, 1, 2, 3, 4 |
| n_train / n_test | 320 / 80 in every split (so 400 points in total) |
| Balanced accuracy | 93.25% +/- 3.38 (spread = standard deviation across the 5 splits) |
| Accuracy | 93.25% +/- 3.38 (the same, because each test set is exactly 40 + 40) |
| Training time per split | 0.0834 s +/- 0.0075 |
| Winning settings | C = 100.0, gamma = 0.5 in all 5 splits |
| Columns used | `f0,f1` |

## 4. Who has done what

| Model | Moons run found? | Branch and path | Exams | Balanced accuracy (mean +/- spread) | Pushed to main? |
|---|---|---|---|---|---|
| A: logistic regression | Run once, then **deleted** at A's request; no file anywhere. The same number appears as C's temporary reference | none (A's log: `notes/AI_USE_A.md` Session 5/6) | first 5 | 83.50% +/- 2.05 (from C's temporary reference, not saved; matches A's logged value) | **No** (no file) |
| B: RBF SVM | **No** | none | n/a | n/a | No |
| C: quantum kernel, ZZ | **Yes** | main: `results/exp5_qkernel_moons.csv` | first 5 | 93.25% +/- 3.38 | **Yes** |
| C: quantum kernel, angle (RY) | Run only as a temporary reference, never saved (recomputed inside the notebook) | main: `src/experiment_qkernel.py`, `notebooks/main.ipynb` | first 5 | 87.25% +/- 3.58 | **No** (no file) |
| D: VQC, ZZ | **No** (not run yet) | none | n/a | n/a | No |
| D: VQC, angle | **No** (not run yet) | none | n/a | n/a | No |

## 5. Are the runs comparable? Data and call, side by side

| | Agreed setup | C (ZZ file + temporary logreg/angle references) | A (deleted run, from the AI log only) | B | D |
|---|---|---|---|---|---|
| Data | `make_moons(n_samples=768, noise=0.2, random_state=42)` | `make_moons(n_samples=400, noise=0.2, random_state=42)` in `make_moons_dataset()` in C's own file | `load_moons()`, n = 400, noise 0.20, seed 42 (own copy) | none | none |
| Column names | x1, x2 | plain NumPy array, so the shared cleaning names them `f0,f1` | unverified (file deleted) | n/a | n/a |
| Scaling | none stated (shared `[0, pi]` cleaning inside `evaluate()`) | shared cleaning only (inside `evaluate()`); C's figure code also scales with `PimaPreprocessor` | shared cleaning | n/a | n/a |
| Shared `src/moons_data.py`? | yes, agreed | **No: each person has an own copy** (C inside `src/experiment_qkernel.py`; A's was in `src/experiment_logreg.py`, now removed) | own copy | n/a | n/a |
| `n_features` | 2 | 2 | 2 | n/a | n/a |
| `zero_as_missing_cols` | None | `()` (empty): same effect as None, nothing is treated as missing | `()` | n/a | n/a |
| `n_splits_to_run` | first 5 (`EXPERIMENT_EVAL_SPLITS`) | `EXPERIMENT_EVAL_SPLITS` (= 5) | 5 | n/a | n/a |
| `results_dir` | `results/experiments` for D; others: team's own | ZZ: `results/`; references: temporary folders | `results/` (deleted) | n/a | n/a |
| Settings list | each model's normal list | ZZ and angle: C's normal 8-combination grid (C in 0.1/1/10/100, gamma 0.5/1.0); logreg reference: the shared logreg grid | logreg's normal grid | n/a | n/a |

**Differences to flag**
1. **Data size: 400 points, not 768.** Everyone who ran it used 400 points (this matches `PROJECT_CONTEXT.md`, which says "~400 points"). The "768" in the agreed setup does not appear anywhere in the repository; where it was agreed is **unverified**. Test sets are 80 patients (not about 154).
2. **Column names:** `f0,f1`, not `x1,x2` (unverified whether anyone expects the names to match; they do not affect the scores).
3. **No shared data file.** `src/moons_data.py` does not exist; C and A each built the data themselves. The arguments (noise 0.2, seed 42, 400) agree, so within the people who ran it the data should be identical, but this is a **same code logic, not same file** check (I did not run it, so identical rows are **unverified**).
4. **Only C's ZZ result is saved as a file.** The angle and logreg numbers exist only as temporary references computed inside C's script and the notebook.
5. **Split ids match** (0-4, same data, same seed) among C's own three runs, which is why C's comparison is valid. There is no second saved file, so I did **not** run `compare_models`.

## 6. Did anyone change `src/evaluate.py` or `src/data.py` for moons?

**No.** `src/data.py` has not changed since the initial commit (Khaled, 2026-11-24); it already supports a `zero_as_missing_cols` argument. `src/evaluate.py` changed twice: `668b608` (mohamedgreida, 2026-11-26 16:06, the logistic regression commit, which added `X=`, `y=`, `zero_as_missing_cols`, `n_splits_to_run`, `results_dir`) and `5bd1d52` (Yousef, 2026-11-26 19:29, the 25-split change). Neither mentions moons; the `X=`/`y=` arguments are what let moons run, and they were added for the whole team.

## 7. What can be compared today

Only C's three runs, all from one script on the same data and splits (400 points, first 5 splits, balanced test sets so balanced accuracy = accuracy). Only ZZ is a saved file; the other two come from C's notes and the notebook's recomputation (matches: 87.25% and 83.50%).

| Model (moons, 2 qubits, first 5 splits) | Balanced accuracy (mean +/- spread) | Above the 50% always-guess floor |
|---|---|---|
| Quantum kernel, ZZ (saved file) | 93.25% +/- 3.38 | +43.25 points |
| Quantum kernel, angle RY (temporary reference) | 87.25% +/- 3.58 | +37.25 points |
| Logistic regression (temporary reference) | 83.50% +/- 2.05 | +33.50 points |

C reports (from C's notes, `compare_models` run by C, not re-run here): ZZ vs logreg +9.75 points (p = 0.0095) and ZZ vs angle +6.00 points (p = 0.0299), both "significant difference found".

**Still missing:** RBF SVM (B); VQC ZZ and VQC angle (D); a saved file for logreg and for the angle kernel (A / C); one agreed data size (400 or 768) written down; a shared `src/moons_data.py`, or at least confirmation that everyone uses the same call.

## Plain-language summary

- **Done:** C ran the moons control for the ZZ quantum kernel and pushed it (93.25%). C's script also produced numbers for the angle kernel (87.25%) and logistic regression (83.50%), but only as temporary references.
- **Started, then removed:** A ran logistic regression on moons, got the same 83.50%, and deleted the files, so nothing of A's is on any branch.
- **Not done:** B (RBF SVM) and D (VQC ZZ and angle).
- **Not fully comparable yet:** everyone so far used 400 points, but your agreed setup says 768, there is no shared data file, and only one result is saved as a file. To compare D's VQC with C's numbers, D must use the same data as C (400 points, noise 0.2, seed 42, 2 columns, first 5 splits) or the team must rerun everyone on 768. That choice is the team's, so I did not make it.

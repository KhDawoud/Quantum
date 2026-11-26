# Read-only review of the `evaluate.py` work (2026-11-26)

Nothing in `src/` or `tests/` was edited, nothing committed, no branch switched. Other branches were read with `git show` / `git diff` / `git archive` into a scratch folder.

## Step 0: where are we?
- Working folder is on branch **`evaluate`** (made from `main`). `src/evaluate.py` here has the full implementation (0 `NotImplementedError` left, 3 functions + helpers).
- It is **uncommitted** (`git status`: `M src/evaluate.py`, `?? tests/test_evaluate.py`, `?? notes/EVALUATE_NOTES.md`, `AM notes/AI_USE_D.md`). Branches: `main`, `vqc`, `evaluate` (+ `origin/main`, `origin/vqc`).

## Concern 1: how many trainings (fits) per split
- **Real number for logistic regression: 19 per split** (6 combinations x 3 inner folds = 18, + 1 final fit). Measured: a wrapped `fit()` counted 95 calls over 5 splits = 19.0 each. Sizes: 18 fits on ~409-410 rows, 1 on 614 rows.
- **Your count (19) is right. My "about 7" was a mistake in my explanation, not in the code.** I wrote "6 inner-fold fits plus the refit" = 7. I used 6 (the number of combinations) and forgot to multiply by the 3 inner folds.
- Code: grid search at `src/evaluate.py:167-176`, final fit at `:185`. Quick mode skips the grid search (1 combination) so it is 1 fit per split.
- **VQC:** `PARAM_GRID` in `src/model_vqc.py` (on `vqc` branch) = `lr` [0.02, 0.05] x `epochs` [20, 40] = **4 combinations**. So 4 x 3 + 1 = **13 fits per split**, as you expected. 5 splits = 65 fits; 25 splits = 325 fits (matches PROJECT_CONTEXT).
- What that means (numbers from `notes/DESIGN_D.md` on `vqc`, one noisy laptop measurement): 5 splits about 1.2-1.35 h, 25 splits about 6-6.7 h, so the 5-split fallback applies. The VQC is far slower than the logreg run I showed (1.3 s).
- Also: my earlier "tune s/split" column for logreg (0.24 s) is the time for 18 fits, not 6.

## Concern 2: where the cleaning happens
- Fitted at `src/evaluate.py:153-155`, inside `_run_one_split`: `fit_transform(X_train, y_train)` on training rows, then `transform(X_test)` on test rows.
- (a) **Yes.** Fitted on training rows only, for every split; test rows are only transformed. Same in train-size mode (it fits on the small training draw only).
- (b) **No cleaning before the split** in `data.py` or `evaluate.py`: `load_pima()` only reads and checks the raw table. No notebooks exist. Outside the evaluation path: `data.py:264` (`summarize_pima`, a printed summary only) and, on the `vqc` branch, `src/vqc_sanity.py:79,89` (`PimaPreprocessor().fit_transform(X, y)` on all 768 rows) are dev checks that do not produce reported scores. They are not leaks into results, but **do not quote their numbers as results**.
- (c) **Confirmed.** Order is: fit preprocessor on the outer training rows -> inner 3-fold tuning on the transformed training rows -> final fit -> test.
  - Small effect in plain language: during tuning, the patients in an inner "practice exam" fold helped decide the median, the 1st/99th percentile limits and the min/max that were used to clean them. So practice-exam scores are very slightly optimistic, for every setting alike, which barely changes which setting wins.
  - No real test row ever reaches the preprocessor's `fit`, the column ranking, or the tuning (mutation test below confirms the fit part).
  - Stricter version: put preprocessor + model in one scikit-learn `Pipeline` and let each inner fold re-fit the cleaning. Needs grid keys like `model__C`, and `evaluate()` to set parameters on the pipeline. Extra run time: cleaning takes milliseconds, the number of model fits stays the same, so roughly **+0-5% for logreg and about nothing for the VQC** (estimate, not measured).

## Concern 3: number of data tests
- **No tests were lost.** `tests/test_data.py` is byte-identical on `main`, `vqc` and `evaluate` (`git diff main vqc -- tests/test_data.py` and `main evaluate` are empty).
- The difference is only how I counted: it has **8 test functions = 10 test cases**, because `test_top_k_feature_selection` runs 3 times (k = 4, 6, 8). pytest `--collect-only` says 10. I wrote "8 existing" earlier, which counted functions; you counted cases.
- The 8 functions: `test_load_pima_integrity` (shape, class counts, known zero counts), `test_hidden_zeros_imputed_but_preg_preserved` (hidden zeros imputed, real `preg` zeros kept, inputs not modified), `test_zero_test_data_leakage` (wildly different test sets leave the fitted numbers unchanged), `test_output_strictly_bounded_in_zero_to_pi`, `test_top_k_feature_selection[4|6|8]`, `test_feature_selection_validation_errors`, `test_sklearn_pipeline_and_moons_compatibility`, `test_summarize_pima_output`.
- `tests/test_vqc.py` exists **only on `vqc`**: 13 functions = **14 cases** (3 of them run for `zz` and `angle`). Not on `main` or `evaluate`.
- `tests/test_evaluate.py` exists only here (uncommitted): 18 cases. **Correction to my earlier message:** I said "20 new + 8 existing = 28". The total 28 is right, the split is **18 new + 10 existing**.
- Leakage tests lost: none. Leakage-related tests now: `test_zero_test_data_leakage` (data.py), `test_preprocessor_only_fits_on_training_rows` and `test_scores_do_not_depend_on_test_labels_of_other_rows` (evaluate).

## Concern 4: the data itself
- Only data file: `data/raw/pima_diabetes.csv` (768 rows + header; tracked in git). No other data files, no notebooks.
- It is **raw**: first row `6,148,72,35,0,33.6,0.627,50,tested_positive`; no blanks (0 NaN) but hidden zeros are still there: `plas` 5, `pres` 35, `skin` 227, `insu` 374, `mass` 11 (and 111 real zeros in `preg`); max `insu` 846, glucose 0-199, age 21-81. Not rescaled to 0..pi, not clipped, not filled. Zero counts match the numbers hard-coded in `test_load_pima_integrity`.
- `load_pima()` (`data.py:35-86`) reads that CSV if it exists; only if it is missing does it download OpenML dataset 37 and save it there. Class labels stay as text in the file and are converted to 0/1 in memory.
- **No file was cleaned with all 768 patients.** Your teammate may be thinking of `PimaPreprocessor`, which does the cleaning at run time, per split.
- Limit: I can only prove the file *looks* raw, not who produced it or when (unverified provenance).

## Concern 5
- **(a)** Yes. `test_first_5_of_25_equal_one_repeat` compares the index arrays of the first 5 of 25 with `n_repeats=1`, and the same test file checks every patient is tested 5x / 1x. Passes.
- **(b)** Two leakage tests, and they are **only partly strong**. I tested them by planting leaks in a throw-away copy (real `src/` untouched):
  - `test_preprocessor_only_fits_on_training_rows` records which patient rows the preprocessor is fitted on and requires them to be exactly the training rows. **Leak A (preprocessor fitted on all 768 rows): the test FAILED, good.**
  - `test_scores_do_not_depend_on_test_labels_of_other_rows` flips the test labels of split 0 and requires the chosen settings to stay the same. **Leak A: it still PASSED. Leak B (tuning grid search also sees the test rows): both tests PASSED.** So a leak inside the tuning step would go unnoticed, and the second test can pass by luck because there are only a few settings to choose from.
  - Verdict: catches the classic "clean everything first" leak, does not catch a peek during tuning or the model fit. See the fix below.
- **(c)** Hand calculation, 5 splits, n_train 600, n_test 150: scores A = .80 .78 .82 .79 .81, B = .75 .77 .78 .80 .74. d = .05 .01 .04 -.01 .07; mean .032; variance (ddof 1) .00102; correction 1/5 + 150/600 = .45; t = .032 / sqrt(.45 x .00102) = **1.4936**; p (4 degrees of freedom, two-sided) = **0.2096**. `compare_models` returned exactly mean_diff .032, t 1.4936, p 0.2096. For contrast, the uncorrected t-test would give t 2.24, p 0.089: the correction is what keeps us from over-claiming.
- **(d)** To run make_moons through `evaluate()` today, these would have to change (not changed):
  1. `evaluate.py:236` always calls `load_pima()`: needs a way to pass data in (e.g. a `dataset`/`loader` argument).
  2. `:153` `n_features < 8` is hard-coded to Pima's 8 columns (moons default 8 would crash on 2 columns); should compare with the real column count.
  3. `:149-150` use `X.iloc`, which needs a DataFrame (moons gives a NumPy array); convert or index both ways.
  4. `PimaPreprocessor` should get `zero_as_missing_cols=()` for non-Pima data (as `test_data.py` does) so no Pima rule is applied.
  - Everything else (splits, tuning, CSV, `compare_models`) is data-agnostic.
- **(e)** `git log --all --oneline -- src/evaluate.py` shows one commit only (`2076017 Inital commit`, the skeleton). `main` and `vqc` still hold the skeleton (3 x `NotImplementedError` each), so **no other local branch has its own evaluate.py**. Unverified: I did not run `git fetch`, so any branch your teammates pushed after your last fetch is invisible to me; run `git fetch` yourself (it changes only remote-tracking refs) and re-check.

## Summary table

| Concern | What you found | Evidence | Verdict | Suggested fix (NOT applied) |
|---|---|---|---|---|
| 1 Fits per split | 19 for logreg (not 7); VQC 13 per split, 65 for 5 splits | counter script: 95 calls / 5 splits; `evaluate.py:167-185`; `model_vqc.py` grid 2x2 | Code fine; my explanation was wrong | Correct the earlier message (done here); re-time VQC before final run |
| 2 Cleaning | Fitted on training rows only, test only transformed; tuning runs after, tiny optimism | `evaluate.py:153-155`; grep of repo | Fine (as agreed) | Optional: Pipeline inside inner folds (+0-5% time) |
| 2b Cleaning of all rows | Only dev scripts on `vqc` (`vqc_sanity.py:79,89`) and a summary print | grep | Fine, but don't quote those numbers | Label them "dev only" |
| 3 Test counts | Nothing lost: 8 functions = 10 cases; test_vqc 13 functions = 14 cases; 18 new here | `git diff` empty; `--collect-only` | Fine; my "20 new/8 existing" was wrong | Fixed in AI log |
| 4 Data | Raw (zeros, max 846, text labels); `load_pima` reads this CSV, downloads only if missing | file head and ranges | Fine; provenance unverified | None |
| 5a First 5 = 25 | Proven by test | `test_first_5_of_25_equal_one_repeat` | Fine | None |
| 5b Leakage test | Catches fit-on-all-rows, misses a peek inside tuning | mutation runs in scratch copy | Needs fixing (weak) | Add test: spy model records rows it sees; compare final coefficients with test labels flipped (must be identical); check `search.fit` input |
| 5c Hand calc | t 1.4936, p 0.2096 both ways | script output | Fine | None |
| 5d moons | 4 changes needed (data argument, `<8`, `.iloc`, zero rule) | `evaluate.py:149,153,236` | Needs follow-up | Small `dataset` argument change, team decision |
| 5e Other evaluate.py | Only the skeleton on `main`/`vqc` | `git log --all` | Fine locally; remote unverified | `git fetch`, re-run the log |

## Temporary files
Scratch scripts and the scratch copy of the repo (in the session scratchpad) were deleted; no `results/` folder exists (the tests write to a temp dir); the earlier pandas/scikit-learn/pytest install into `.venv` is unchanged.

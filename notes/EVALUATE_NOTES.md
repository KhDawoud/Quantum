# Notes: choices made in `src/evaluate.py`

Short, in plain language, so any team member can answer judges' questions.

## Splits
- `RepeatedStratifiedKFold(5 folds, 5 repeats, seed 42)` gives 25 splits. Stratified = each fold has the same share of diabetic patients.
- Splits come out in repeat order, so `n_repeats=1` is exactly the first 5 of the 25 (tested). Quick mode and the 5-split fallback therefore use the same patients as the full run.

## What happens in each split (and why nothing leaks)
1. `PimaPreprocessor` is fitted on the training rows only (medians, 1st/99th percentiles, min/max, and the column ranking when `n_features < 8`). Test rows are only transformed.
2. Tuning: `GridSearchCV`, 3 inner stratified folds (shuffled, seed 42), `balanced_accuracy`, on training rows only.
3. We refit ourselves with the winning settings (`refit=False` in the grid search) so `fit_seconds` is the time of exactly that final fit; `tune_seconds` is the grid search alone.
4. Only now are the test rows scored.
- Known simplification: the preprocessor is fitted once on the whole outer training set, *then* the inner 3-fold tuning runs. So an inner validation fold has influenced the scaling numbers it is scored with. No test row is involved, so the HARD RULE holds; the effect on tuning is tiny. A full pipeline per inner fold would be stricter but cost more (VQC fits are slow) and the task spec asked for this order.
- `error_score="raise"`: if a setting crashes, we want to know, not get a silent NaN.

## Quick mode
- `n_repeats=1` and only the first value of each parameter. If that leaves one combination, the inner folds are skipped (`tune_seconds = 0`): there is nothing to choose between.
- In training-size mode quick only shrinks the grid; the number of draws is still `n_draws`.

## `fixed_params`
Skips tuning completely and sets the given parameters on a fresh model. Intended for experiment 2 (tune once on the 614-patient pool, reuse).

## Training-size mode (experiment 2)
- One fixed stratified 20% test set (154 patients, seed 42), identical for all sizes and draws.
- Draw `i` picks `train_size` patients, stratified, from the other 614 with seed `42 + i`.
- If `fixed_params` is not given, tuning is repeated inside every draw (slow; normally pass `fixed_params`).
- The Nadeau-Bengio test assumes CV-style resampling. With one fixed test set it is only a rough guide; say so in the report if used there.

## Output
- CSV columns: `split, n_train, n_test, accuracy, balanced_accuracy, f1_macro, tune_seconds, fit_seconds, best_params` (best_params is JSON text). In training-size mode `split` is the draw number.
- Returned dict: `<metric>_mean` and `<metric>_std` for the three scores and both timings, `n_splits`, `csv_path`. Standard deviation is the sample one (`ddof=1`).

## `compare_models`
- Takes DataFrames or CSV paths, requires identical `split`, `n_train`, `n_test` columns, differences are A minus B.
- `t = mean(d) / sqrt((1/J + n_test/n_train) * var(d, ddof=1))`, `J-1` degrees of freedom, two-sided p.
- `n_train`/`n_test` default to the mean of the CSV columns (they differ by at most 1 between folds).
- Zero variance: identical results give `t=0, p=1.0`; a constant non-zero gap gives `t=±inf, p=0.0`.
- Wording rule from PROJECT_CONTEXT: if p is large, write "no significant difference found", never "equal".

## Open point for the team
Experiment 5 (make_moons) needs `evaluate()` to accept other data, but the agreed signature has no data argument and I was told not to add one. `evaluate()` currently always loads Pima. Options: add a `dataset` argument later (small change) or run moons with its own call to `get_splits()`.

## Environment note
The `.venv` was missing pandas and scikit-learn (only part of `requirements.txt` was installed); I installed `pandas==3.1.1`, `scikit-learn==1.9.1`, `pytest==9.2.0`. On Windows run tests with `.venv\Scripts\python -m pytest -v` (not `.venv/bin/pytest`).

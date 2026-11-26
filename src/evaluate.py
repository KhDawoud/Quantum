"""Shared cross-validation, tuning, evaluation, and significance testing.

Nobody writes custom evaluation loops inside model files; every model runs
through ``evaluate()`` and is compared using ``compare_models()``.

Big picture (plain language)
----------------------------
* ``get_splits()``      decides WHICH patients are used for training and
                        WHICH are held back for testing, the same way for
                        every model.
* ``evaluate()``        for each split: clean the data using training rows
                        only, pick the best settings (tuning) using training
                        rows only, train once more, then score on the held-back
                        test rows. Results are saved as one CSV row per split.
* ``compare_models()``  asks "is model A really better than model B, or could
                        the difference be luck?" (corrected paired t-test).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from src.data import PimaPreprocessor, load_pima

RANDOM_SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "f1_macro")
RESULT_COLUMNS: tuple[str, ...] = (
    "split",
    "n_train",
    "n_test",
    *METRICS,
    "tune_seconds",
    "fit_seconds",
    "best_params",
)

# Training-size mode (experiment 2): fraction of patients locked away as the
# one fixed test set.
TRAIN_SIZE_TEST_FRACTION = 0.2


def get_splits(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic repeated stratified train/test index pairs.

    Stratified = every fold keeps the same diabetic / non-diabetic ratio as
    the whole dataset. Repeated = the whole 5-fold shuffle is done
    ``n_repeats`` times with different shuffles, giving ``n_splits * n_repeats``
    splits. The same seed always gives the same splits, so every model is
    tested on exactly the same patients.

    Splits come in order: split 0-4 are repeat 1, split 5-9 are repeat 2, ...
    That is why ``n_repeats=1`` returns exactly the first 5 of the 25 splits.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return [(train, test) for train, test in splitter.split(X, y)]


def _train_size_splits(
    y: np.ndarray, train_size: int, n_draws: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Experiment 2: one fixed 20% test set + ``n_draws`` small training sets.

    The test set is chosen once (seed 42) and never changes. Each draw picks
    ``train_size`` random patients, stratified, from the remaining 80% (the
    "pool"). Draw ``i`` uses seed ``42 + i`` so draws differ from each other
    but are reproducible.
    """
    all_idx = np.arange(len(y))
    pool_idx, test_idx = train_test_split(
        all_idx,
        test_size=TRAIN_SIZE_TEST_FRACTION,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    pool_idx = np.sort(pool_idx)
    test_idx = np.sort(test_idx)
    if train_size > len(pool_idx):
        raise ValueError(
            f"train_size={train_size} is larger than the training pool "
            f"({len(pool_idx)} patients)."
        )
    draws = []
    for i in range(n_draws):
        if train_size == len(pool_idx):
            train_idx = pool_idx
        else:
            train_idx, _ = train_test_split(
                pool_idx,
                train_size=train_size,
                stratify=y[pool_idx],
                random_state=RANDOM_SEED + i,
            )
        draws.append((np.sort(train_idx), test_idx))
    return draws


def _first_value_grid(param_grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Quick mode: keep only the first value of every parameter."""
    return {key: [values[0]] for key, values in param_grid.items()}


def _n_combinations(param_grid: dict[str, list[Any]]) -> int:
    return math.prod(len(values) for values in param_grid.values())


def _json_default(obj: Any) -> Any:
    # numpy numbers are not JSON-serialisable by default
    return obj.item() if hasattr(obj, "item") else str(obj)


def _run_one_split(
    split_id: int,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    make_model: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    n_features: int,
    fixed_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Train on ``train_idx``, score on ``test_idx``. Test rows are used ONLY
    in the last scoring step."""
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # 1. Cleaning + scaling + column ranking: learned from training rows only.
    prep = PimaPreprocessor(n_features=n_features if n_features < 8 else None)
    Z_train = prep.fit_transform(X_train, y_train)
    Z_test = prep.transform(X_test)  # only applies what was learned above

    # 2. Tuning: inner 3-fold cross-validation on the training rows only.
    tune_start = time.perf_counter()
    if fixed_params is not None:
        best_params = dict(fixed_params)
        tune_seconds = 0.0
    elif _n_combinations(param_grid) <= 1:
        # Nothing to choose between, so skip the (pointless) inner folds.
        best_params = {key: values[0] for key, values in param_grid.items()}
        tune_seconds = 0.0
    else:
        search = GridSearchCV(
            make_model(),
            param_grid,
            scoring="balanced_accuracy",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            refit=False,  # we refit ourselves below so we can time it
            n_jobs=1,
            error_score="raise",  # never hide a crashing setting as NaN
        )
        search.fit(Z_train, y_train)
        best_params = dict(search.best_params_)
        tune_seconds = time.perf_counter() - tune_start

    # 3. Final fit on the whole training set with the chosen settings.
    model = make_model()
    if best_params:
        model.set_params(**best_params)
    fit_start = time.perf_counter()
    model.fit(Z_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    # 4. Scoring on the held-back test rows.
    pred = model.predict(Z_test)
    return {
        "split": split_id,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "tune_seconds": tune_seconds,
        "fit_seconds": fit_seconds,
        "best_params": json.dumps(best_params, sort_keys=True, default=_json_default),
    }


def evaluate(
    name: str,
    make_model: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    n_features: int = 8,
    train_size: int | None = None,
    quick: bool = False,
    fixed_params: dict[str, Any] | None = None,
    n_repeats: int = 5,
    n_draws: int = 10,
) -> dict[str, Any]:
    """Run the shared fair-comparison pipeline and save ``results/<name>.csv``.

    Args:
        name: Used for the output file ``results/<name>.csv``.
        make_model: Returns a fresh scikit-learn style estimator.
        param_grid: Settings to try (tuning). Ignored if ``fixed_params``.
        n_features: Keep the top-k columns (k < 8) ranked on training rows.
        train_size: If given, switch to training-size mode (experiment 2):
            fixed 20% test set, ``n_draws`` random training subsets of this
            size. Otherwise normal 5-fold CV x ``n_repeats``.
        quick: Development mode: ``n_repeats=1`` (first 5 splits) and only
            the first value of each parameter in ``param_grid``.
        fixed_params: Use these settings, skip tuning.
        n_repeats: Number of 5-fold repeats (default 5 -> 25 splits).
        n_draws: Number of random training subsets in training-size mode.

    Returns:
        Dict with ``<metric>_mean`` / ``<metric>_std`` for accuracy,
        balanced_accuracy, f1_macro, tune_seconds and fit_seconds, plus
        ``n_splits`` and ``csv_path``. ``_std`` is the sample standard
        deviation (ddof=1) across splits.
    """
    X, y = load_pima()

    if quick:
        n_repeats = 1
        param_grid = _first_value_grid(param_grid)

    if train_size is not None:
        splits = _train_size_splits(y, train_size, n_draws)
    else:
        splits = get_splits(X, y, n_splits=5, n_repeats=n_repeats)

    rows = [
        _run_one_split(
            i, X, y, train_idx, test_idx, make_model, param_grid, n_features, fixed_params
        )
        for i, (train_idx, test_idx) in enumerate(splits)
    ]
    df = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    summary: dict[str, Any] = {}
    for col in (*METRICS, "tune_seconds", "fit_seconds"):
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")
    summary["n_splits"] = int(len(df))
    summary["csv_path"] = str(csv_path)
    return summary


def compare_models(
    results_a: pd.DataFrame | str,
    results_b: pd.DataFrame | str,
    metric: str = "balanced_accuracy",
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, float]:
    """Compare two models on identical splits with Nadeau-Bengio resampled t-test.

    Plain-language explanation
    --------------------------
    Both models were scored on the same J splits. For each split we take the
    difference d = (score of A) - (score of B). If the models were truly equal,
    the average difference would be about 0. A normal paired t-test asks
    "is the mean difference far from 0 compared to its wobble?".

    The catch: the splits overlap (training sets share most patients), so the
    J differences are NOT independent, and the normal t-test is over-confident
    (it finds "significant" differences that are just luck). Nadeau & Bengio
    (2003) fixed this by enlarging the variance:

        t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) )

    The extra ``n_test/n_train`` term pays for the overlap. t is compared to a
    t-distribution with J-1 degrees of freedom; p is two-sided: "if the models
    were equal, how often would we see a gap at least this big?". A small p
    (e.g. < 0.05) suggests a real difference; a large p means "no significant
    difference found" (NOT "they are equal").

    Args:
        results_a, results_b: DataFrames or CSV paths written by ``evaluate()``.
        metric: Column to compare.
        n_train, n_test: Override the sizes used in the correction; by default
            the mean ``n_train`` / ``n_test`` columns of the results are used.

    Returns:
        ``mean_diff`` (A minus B), ``std_diff``, ``t``, ``p``, ``n_splits``.
    """
    df_a = pd.read_csv(results_a) if isinstance(results_a, (str, Path)) else results_a
    df_b = pd.read_csv(results_b) if isinstance(results_b, (str, Path)) else results_b

    for label, df in (("results_a", df_a), ("results_b", df_b)):
        for col in ("split", "n_train", "n_test", metric):
            if col not in df.columns:
                raise ValueError(f"{label} has no '{col}' column.")

    df_a = df_a.sort_values("split").reset_index(drop=True)
    df_b = df_b.sort_values("split").reset_index(drop=True)
    id_cols = ["split", "n_train", "n_test"]
    if len(df_a) != len(df_b) or not df_a[id_cols].equals(df_b[id_cols]):
        raise ValueError(
            "The two results do not cover identical splits "
            "(different split ids or train/test sizes)."
        )

    n_splits = len(df_a)
    if n_splits < 2:
        raise ValueError("Need at least 2 splits to compare models.")

    diffs = df_a[metric].to_numpy(dtype=float) - df_b[metric].to_numpy(dtype=float)
    mean_diff = float(diffs.mean())
    var_diff = float(diffs.var(ddof=1))
    std_diff = math.sqrt(var_diff)

    train_n = float(n_train if n_train is not None else df_a["n_train"].mean())
    test_n = float(n_test if n_test is not None else df_a["n_test"].mean())
    correction = 1.0 / n_splits + test_n / train_n

    if var_diff == 0.0:
        # All differences identical: no wobble to measure.
        if mean_diff == 0.0:
            t_stat, p_value = 0.0, 1.0  # models score identically everywhere
        else:
            # Same non-zero gap on every split: infinitely clear.
            t_stat, p_value = math.copysign(math.inf, mean_diff), 0.0
    else:
        t_stat = mean_diff / math.sqrt(correction * var_diff)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n_splits - 1))

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t": float(t_stat),
        "p": float(p_value),
        "n_splits": n_splits,
    }

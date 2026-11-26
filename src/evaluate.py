"""Shared cross-validation, tuning, evaluation, and significance testing.

Nobody writes custom evaluation loops inside model files; every model runs
through ``evaluate()`` and is compared using ``compare_models()``.

Sources adapted / cited:
  - scikit-learn ``RepeatedStratifiedKFold``, ``StratifiedKFold``,
    ``StratifiedShuffleSplit``, ``Pipeline``, and ``GridSearchCV`` docs:
    https://scikit-learn.org/stable/modules/cross_validation.html
  - Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error.
    Machine Learning, 52(3), 239-281 (corrected resampled paired t-test).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline

from src.data import ZERO_AS_MISSING_COLS, PimaPreprocessor, load_pima

RANDOM_SEED = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
# Timing fallback per PROJECT_CONTEXT.md and Member D's VQC timing in DESIGN_D.md:
# 8-qubit VQC-ZZ takes ~6.0-7.8 h for 25 splits (> 3 h threshold), so all models
# evaluate on the exact same first 5 splits (one complete stratified 5-fold repeat).
DEFAULT_EVAL_SPLITS = 5
INNER_CV_SPLITS = 3
MAX_PARAM_GRID_COMBOS = 8
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _slice_rows(
    X: pd.DataFrame | np.ndarray, indices: np.ndarray
) -> pd.DataFrame | np.ndarray:
    """Slice rows consistently whether ``X`` is a DataFrame or NumPy array."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _resolve_dataset(
    X: pd.DataFrame | np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
    """Return ``(X, y)``, loading Pima when both are omitted."""
    if X is None and y is None:
        return load_pima()
    if X is None or y is None:
        raise ValueError("Provide both X and y, or omit both to load Pima.")
    return X, np.asarray(y)


def _count_grid_combos(param_grid: dict[str, list[Any]]) -> int:
    """Validate parameter lists are non-empty and return total combination count."""
    if not param_grid:
        return 0
    counts: list[int] = []
    for key, values in param_grid.items():
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
            raise ValueError(f"param_grid[{key!r}] must be a non-empty sequence.")
        counts.append(len(values))
    return int(np.prod(counts))


def _shrink_param_grid(
    param_grid: dict[str, list[Any]], max_combos: int = 2
) -> dict[str, list[Any]]:
    """Reduce a parameter grid to at most ``max_combos`` combinations for quick mode."""
    shrunk: dict[str, list[Any]] = {}
    kept_multi = False
    for key, values in param_grid.items():
        vals = list(values)
        if not kept_multi and len(vals) > 1:
            shrunk[key] = vals[:max_combos]
            kept_multi = True
        else:
            shrunk[key] = vals[:1]
    return shrunk


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip ``clf__`` pipeline prefixes and coerce NumPy scalars for JSON output."""
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        short_key = key[5:] if key.startswith("clf__") else key
        if isinstance(value, np.generic):
            value = value.item()
        cleaned[short_key] = value
    return cleaned


def _build_pipeline(
    make_model: Callable[[], Any],
    n_features: int,
    zero_as_missing_cols: Iterable[str] | None,
) -> Pipeline:
    """Create the zero-leakage ``PimaPreprocessor -> classifier`` pipeline."""
    prep = PimaPreprocessor(
        n_features=n_features,
        zero_as_missing_cols=zero_as_missing_cols,
    )
    return Pipeline([("prep", prep), ("clf", make_model())])


def _run_grid_search(
    pipe: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """Fit 3-fold inner ``GridSearchCV`` scored on balanced accuracy."""
    prefixed_grid = {f"clf__{k}": list(v) for k, v in param_grid.items()}
    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    search = GridSearchCV(
        estimator=pipe,
        param_grid=prefixed_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def get_splits(
    X: pd.DataFrame | np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic repeated stratified train/test index pairs."""
    X_data, y_arr = _resolve_dataset(X, y)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(X_data, y_arr)
    ]


def get_fixed_test_split(
    X: pd.DataFrame | np.ndarray | None = None,
    y: np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed 80/20 stratified train-pool and test index split (Exp 2)."""
    X_data, y_arr = _resolve_dataset(X, y)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    pool_idx, test_idx = next(splitter.split(X_data, y_arr))
    return np.asarray(pool_idx), np.asarray(test_idx)


def get_trainsize_splits(
    X: pd.DataFrame | np.ndarray | None,
    y: np.ndarray | None,
    train_size: int,
    n_draws: int = 10,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Draw ``n_draws`` stratified training subsets against a fixed 20% test set."""
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}.")
    X_data, y_arr = _resolve_dataset(X, y)
    pool_idx, test_idx = get_fixed_test_split(
        X_data, y_arr, test_size=test_size, random_state=random_state
    )
    if train_size < 1 or train_size > len(pool_idx):
        raise ValueError(
            f"train_size must be between 1 and {len(pool_idx)}, got {train_size}."
        )
    if train_size == len(pool_idx):
        return [(pool_idx.copy(), test_idx.copy()) for _ in range(n_draws)]

    X_pool = _slice_rows(X_data, pool_idx)
    y_pool = y_arr[pool_idx]
    draws: list[tuple[np.ndarray, np.ndarray]] = []
    for draw_idx in range(n_draws):
        sub_splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            random_state=random_state + draw_idx,
        )
        rel_train_idx, _ = next(sub_splitter.split(X_pool, y_pool))
        draws.append((pool_idx[rel_train_idx], test_idx))
    return draws


def tune_hyperparameters(
    make_model: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    n_features: int = 8,
    zero_as_missing_cols: Iterable[str] | None = ZERO_AS_MISSING_COLS,
) -> dict[str, Any]:
    """Tune a model once on a training pool (e.g. 614 patients) and return best params."""
    n_combos = _count_grid_combos(param_grid)
    if n_combos > MAX_PARAM_GRID_COMBOS:
        raise ValueError(
            f"param_grid has {n_combos} combinations; max fair budget is {MAX_PARAM_GRID_COMBOS}."
        )
    if not param_grid:
        return {}
    pipe = _build_pipeline(make_model, n_features, zero_as_missing_cols)
    search = _run_grid_search(pipe, param_grid, X_train, np.asarray(y_train))
    return _normalize_params(search.best_params_)


def evaluate(
    name: str,
    make_model: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    n_features: int = 8,
    train_size: int | None = None,
    quick: bool = False,
    fixed_params: dict[str, Any] | None = None,
    *,
    X: pd.DataFrame | np.ndarray | None = None,
    y: np.ndarray | None = None,
    zero_as_missing_cols: Iterable[str] | None = ZERO_AS_MISSING_COLS,
    n_splits_to_run: int | None = None,
    n_draws: int = 10,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the shared fair-comparison pipeline and save ``results/<name>.csv``."""
    n_combos = _count_grid_combos(param_grid)
    if n_combos > MAX_PARAM_GRID_COMBOS:
        raise ValueError(
            f"param_grid has {n_combos} combinations; max fair budget is {MAX_PARAM_GRID_COMBOS}."
        )

    X_data, y_arr = _resolve_dataset(X, y)
    effective_grid = (
        _shrink_param_grid(param_grid)
        if (quick and fixed_params is None and param_grid)
        else param_grid
    )

    if train_size is not None:
        effective_draws = min(3, n_draws) if quick else n_draws
        splits = get_trainsize_splits(
            X_data, y_arr, train_size=train_size, n_draws=effective_draws
        )
    else:
        all_splits = get_splits(X_data, y_arr)
        limit = DEFAULT_EVAL_SPLITS if n_splits_to_run is None else n_splits_to_run
        if quick:
            limit = min(limit, DEFAULT_EVAL_SPLITS)
        if limit < 1 or limit > len(all_splits):
            raise ValueError(
                f"n_splits_to_run must be between 1 and {len(all_splits)}, got {limit}."
            )
        splits = all_splits[:limit]

    rows: list[dict[str, Any]] = []
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = _slice_rows(X_data, train_idx)
        y_train = y_arr[train_idx]
        X_test = _slice_rows(X_data, test_idx)
        y_test = y_arr[test_idx]

        pipe = _build_pipeline(make_model, n_features, zero_as_missing_cols)

        start = time.perf_counter()
        if fixed_params is not None:
            chosen_params = _normalize_params(fixed_params)
            if chosen_params:
                pipe.set_params(**{f"clf__{k}": v for k, v in chosen_params.items()})
            fitted_pipe = pipe.fit(X_train, y_train)
        elif effective_grid:
            search = _run_grid_search(pipe, effective_grid, X_train, y_train)
            fitted_pipe = search.best_estimator_
            chosen_params = _normalize_params(search.best_params_)
        else:
            fitted_pipe = pipe.fit(X_train, y_train)
            chosen_params = {}
        train_time_sec = float(time.perf_counter() - start)

        y_pred = fitted_pipe.predict(X_test)
        selected_cols = ",".join(
            str(col) for col in fitted_pipe.named_steps["prep"].get_feature_names_out()
        )
        rows.append(
            {
                "split": split_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(
                    f1_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "train_time_sec": train_time_sec,
                "best_params": json.dumps(chosen_params, sort_keys=True),
                "selected_features": selected_cols,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{name}.csv"
    df.to_csv(results_path, index=False)

    _, class_counts = np.unique(y_arr, return_counts=True)
    majority_floor = float(class_counts.max() / len(y_arr))
    ddof = 1 if len(df) > 1 else 0

    return {
        "name": name,
        "results_path": str(results_path),
        "n_splits": int(len(df)),
        "n_features": int(n_features),
        "train_size": train_size,
        "quick": bool(quick),
        "fixed_params": _normalize_params(fixed_params) if fixed_params else None,
        "majority_floor": majority_floor,
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=ddof)),
        "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(df["balanced_accuracy"].std(ddof=ddof)),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std(ddof=ddof)),
        "train_time_sec_mean": float(df["train_time_sec"].mean()),
        "train_time_sec_std": float(df["train_time_sec"].std(ddof=ddof)),
        "results": df,
    }


def _load_results_df(results: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    return pd.read_csv(Path(results))


def _resolve_split_size(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col: str,
    override: int | float | None,
) -> float:
    if override is not None:
        val = float(override)
    elif col in df_a.columns:
        if col in df_b.columns and not np.allclose(
            df_a[col].to_numpy(dtype=np.float64),
            df_b[col].to_numpy(dtype=np.float64),
        ):
            raise ValueError(
                f"Mismatched '{col}' sizes between results_a and results_b."
            )
        val = float(df_a[col].mean())
    elif col in df_b.columns:
        val = float(df_b[col].mean())
    else:
        raise ValueError(
            f"'{col}' was not supplied and column '{col}' is missing from results."
        )
    if val <= 0.0:
        raise ValueError(f"'{col}' must be positive, got {val}.")
    return val


def compare_models(
    results_a: pd.DataFrame | str | Path,
    results_b: pd.DataFrame | str | Path,
    metric: str = "balanced_accuracy",
    n_train: int | None = None,
    n_test: int | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compare two models on identical splits with the Nadeau-Bengio resampled t-test."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    df_a = _load_results_df(results_a)
    df_b = _load_results_df(results_b)

    for label, df in (("results_a", df_a), ("results_b", df_b)):
        if metric not in df.columns:
            raise ValueError(f"Metric {metric!r} not found in {label}.")

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Split counts must match for paired comparison ({len(df_a)} vs {len(df_b)})."
        )
    j = len(df_a)
    if j < 2:
        raise ValueError(f"Need at least 2 splits for paired t-test, got {j}.")

    if "split" in df_a.columns and "split" in df_b.columns:
        if not np.array_equal(df_a["split"].to_numpy(), df_b["split"].to_numpy()):
            raise ValueError(
                "Split IDs differ between results_a and results_b; cannot pair results."
            )

    n_train_val = _resolve_split_size(df_a, df_b, "n_train", n_train)
    n_test_val = _resolve_split_size(df_a, df_b, "n_test", n_test)

    scores_a = df_a[metric].to_numpy(dtype=np.float64)
    scores_b = df_b[metric].to_numpy(dtype=np.float64)
    diff = scores_a - scores_b

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    var_diff = float(np.var(diff, ddof=1))
    corrected_var = (1.0 / j + n_test_val / n_train_val) * var_diff
    corrected_se = float(np.sqrt(max(0.0, corrected_var)))
    df_val = j - 1

    if corrected_se == 0.0:
        if mean_diff == 0.0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = float(np.copysign(np.inf, mean_diff))
            p_value = 0.0
    else:
        t_stat = float(mean_diff / corrected_se)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df_val))

    significant = bool(p_value < alpha)
    conclusion = (
        f"significant difference found (p < {alpha:g})"
        if significant
        else "no significant difference found"
    )

    return {
        "metric": metric,
        "n_splits": j,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "corrected_se": corrected_se,
        "t_stat": t_stat,
        "df": df_val,
        "p_value": p_value,
        "significant": significant,
        "conclusion": conclusion,
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Render a plain-language results table for a completed ``evaluate()`` run."""
    floor_pct = 100.0 * summary["majority_floor"]
    lines = [
        f"Results for '{summary['name']}' "
        f"({summary['n_splits']} splits, {summary['n_features']} features, "
        f"quick={summary['quick']})",
        f"  Always-guess floor : {floor_pct:.2f}%",
        f"  Accuracy           : {100.0 * summary['accuracy_mean']:.2f}% "
        f"+/- {100.0 * summary['accuracy_std']:.2f}%",
        f"  Balanced accuracy  : {100.0 * summary['balanced_accuracy_mean']:.2f}% "
        f"+/- {100.0 * summary['balanced_accuracy_std']:.2f}%",
        f"  Macro F1           : {100.0 * summary['macro_f1_mean']:.2f}% "
        f"+/- {100.0 * summary['macro_f1_std']:.2f}%",
        f"  Train time / split : {summary['train_time_sec_mean']:.4f}s "
        f"+/- {summary['train_time_sec_std']:.4f}s",
        f"  Saved CSV          : {summary['results_path']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.model_logreg import PARAM_GRID, make_model

    logreg_summary = evaluate("logreg", make_model, PARAM_GRID)
    print(format_summary(logreg_summary))

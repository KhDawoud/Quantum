"""Shared cross-validation, tuning, evaluation, and significance testing.

Nobody writes custom evaluation loops inside model files; every model runs
through ``evaluate()`` and is compared using ``compare_models()``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def get_splits(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic repeated stratified train/test index pairs."""
    raise NotImplementedError(
        "Implement get_splits() using RepeatedStratifiedKFold(random_state=42)."
    )


def evaluate(
    name: str,
    make_model: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    n_features: int = 8,
    train_size: int | None = None,
    quick: bool = False,
    fixed_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the shared fair-comparison pipeline and save ``results/<name>.csv``."""
    raise NotImplementedError("Implement evaluate() per PROJECT_CONTEXT.md.")


def compare_models(
    results_a: pd.DataFrame | str,
    results_b: pd.DataFrame | str,
    metric: str = "balanced_accuracy",
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, float]:
    """Compare two models on identical splits with Nadeau-Bengio resampled t-test."""
    raise NotImplementedError(
        "Implement compare_models() per PROJECT_CONTEXT.md."
    )

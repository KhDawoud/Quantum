"""Dataset loading and leakage-free preprocessing for Q//Horizon 2026."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted

RANDOM_SEED = 42
PIMA_OPENML_ID = 37
PIMA_COLUMNS: tuple[str, ...] = (
    "preg",
    "plas",
    "pres",
    "skin",
    "insu",
    "mass",
    "pedi",
    "age",
)
# Zeros in these five clinical columns mean "not measured" -> treat as missing.
# Never include 'preg' because 0 pregnancies is a real measurement.
ZERO_AS_MISSING_COLS: tuple[str, ...] = ("plas", "pres", "skin", "insu", "mass")
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "pima_diabetes.csv"
)


def load_pima(
    cache_path: str | Path | None = None,
    prefer_cache: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the Pima Indians Diabetes dataset (OpenML ID 37).

    Args:
        cache_path: Optional CSV path to read from / write to. Defaults to
            ``data/raw/pima_diabetes.csv`` inside the repository root.
        prefer_cache: When ``True`` and ``cache_path`` exists, load the cached
            CSV directly for offline reproducibility.

    Returns:
        ``(X, y)`` where ``X`` is a ``(768, 8)`` ``float64`` DataFrame with
        ``PIMA_COLUMNS`` and ``y`` is a ``(768,)`` ``int64`` NumPy array with
        ``tested_positive = 1`` (268 rows) and ``tested_negative = 0`` (500 rows).
    """
    resolved_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH

    if prefer_cache and resolved_path.exists():
        df = pd.read_csv(resolved_path)
    else:
        bunch = fetch_openml(data_id=PIMA_OPENML_ID, as_frame=True, parser="auto")
        df = bunch.data[list(PIMA_COLUMNS)].copy()
        df["class"] = bunch.target.astype(str).to_numpy()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_path, index=False)

    missing_cols = [col for col in (*PIMA_COLUMNS, "class") if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing expected columns: {missing_cols}")

    X = df[list(PIMA_COLUMNS)].astype(np.float64).copy()
    labels = df["class"].astype(str).str.strip()
    valid_labels = {"tested_positive", "tested_negative"}
    unexpected = set(labels.unique()) - valid_labels
    if unexpected:
        raise ValueError(f"Unexpected target labels found: {unexpected}")

    y = (labels == "tested_positive").astype(np.int64).to_numpy()

    if X.shape != (768, 8):
        raise ValueError(f"Expected X shape (768, 8), got {X.shape}")
    if y.shape != (768,) or np.isnan(y).any():
        raise ValueError(f"Expected valid y shape (768,), got {y.shape}")
    if int((y == 0).sum()) != 500 or int((y == 1).sum()) != 268:
        raise ValueError(
            "Unexpected class counts: "
            f"neg={int((y == 0).sum())}, pos={int((y == 1).sum())}"
        )

    return X, y


class PimaPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-free preprocessor for Pima Indians Diabetes (outputs in [0, pi]).

    Steps learned strictly from training rows inside ``fit(X, y)``:
      1. Convert impossible zeros in ``zero_as_missing_cols`` to ``NaN``.
      2. Fill missing values with training column medians.
      3. Clip each column to its training 1st and 99th percentiles.
      4. Min-max rescale each column into ``[0, pi]`` (clipping test values).
      5. Optionally keep the top ``n_features`` columns ranked by ``f_classif``.
    """

    def __init__(
        self,
        n_features: int | None = None,
        zero_as_missing_cols: Iterable[str] | None = ZERO_AS_MISSING_COLS,
    ) -> None:
        self.n_features = n_features
        self.zero_as_missing_cols = zero_as_missing_cols

    def _to_array_and_names(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float64, copy=True)
            names = [str(col) for col in X.columns]
        else:
            arr = np.array(X, dtype=np.float64, copy=True)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
            if arr.shape[1] == len(PIMA_COLUMNS):
                names = list(PIMA_COLUMNS)
            else:
                names = [f"f{idx}" for idx in range(arr.shape[1])]
        return arr, names

    def _resolve_zero_missing_indices(self, names: list[str]) -> list[int]:
        if self.zero_as_missing_cols is None:
            return []
        cols = tuple(self.zero_as_missing_cols)
        if not cols:
            return []
        return [idx for idx, name in enumerate(names) if name in cols]

    def _apply_imputation_clip_scale(self, X_arr: np.ndarray) -> np.ndarray:
        X_work = np.array(X_arr, dtype=np.float64, copy=True)
        for col_idx in self.zero_missing_indices_:
            zero_mask = X_work[:, col_idx] == 0.0
            X_work[zero_mask, col_idx] = np.nan

        nan_mask = np.isnan(X_work)
        if nan_mask.any():
            X_work = np.where(nan_mask, self.medians_, X_work)

        X_work = np.clip(X_work, self.p01_, self.p99_)

        span = self.max_ - self.min_
        safe_span = np.where(span > 0.0, span, 1.0)
        X_scaled = np.where(
            span > 0.0,
            (X_work - self.min_) / safe_span * np.pi,
            0.0,
        )
        return np.clip(X_scaled, 0.0, np.pi)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | None = None,
    ) -> "PimaPreprocessor":
        """Learn medians, percentiles, min/max bounds, and feature ranking from training rows."""
        X_work, names = self._to_array_and_names(X)
        n_samples, n_cols = X_work.shape
        if n_samples == 0:
            raise ValueError("Cannot fit PimaPreprocessor on empty input.")

        self.n_features_in_ = n_cols
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.zero_missing_indices_ = self._resolve_zero_missing_indices(names)

        for col_idx in self.zero_missing_indices_:
            zero_mask = X_work[:, col_idx] == 0.0
            X_work[zero_mask, col_idx] = np.nan

        medians = np.nanmedian(X_work, axis=0)
        self.medians_ = np.where(np.isnan(medians), 0.0, medians)

        nan_mask = np.isnan(X_work)
        if nan_mask.any():
            X_work = np.where(nan_mask, self.medians_, X_work)

        self.p01_ = np.percentile(X_work, 1, axis=0)
        self.p99_ = np.percentile(X_work, 99, axis=0)
        X_clipped = np.clip(X_work, self.p01_, self.p99_)

        self.min_ = X_clipped.min(axis=0)
        self.max_ = X_clipped.max(axis=0)

        X_scaled = self._apply_imputation_clip_scale(X_clipped)

        if self.n_features is None or self.n_features == n_cols:
            self.selected_indices_ = np.arange(n_cols, dtype=int)
            self.f_scores_ = None
        else:
            if not isinstance(self.n_features, (int, np.integer)) or (
                self.n_features <= 0 or self.n_features > n_cols
            ):
                raise ValueError(
                    f"n_features must be between 1 and {n_cols}, got {self.n_features}"
                )
            if y is None:
                raise ValueError(
                    "Target vector y is required during fit() when n_features < total columns."
                )
            y_arr = np.asarray(y)
            f_scores, _ = f_classif(X_scaled, y_arr)
            safe_scores = np.where(np.isnan(f_scores), -np.inf, f_scores)
            ranked = np.lexsort((np.arange(n_cols), -safe_scores))
            self.f_scores_ = f_scores
            self.selected_indices_ = ranked[: int(self.n_features)]

        self.selected_feature_names_ = [
            names[int(idx)] for idx in self.selected_indices_
        ]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform rows using only parameters learned during ``fit()``."""
        check_is_fitted(
            self,
            [
                "n_features_in_",
                "zero_missing_indices_",
                "medians_",
                "p01_",
                "p99_",
                "min_",
                "max_",
                "selected_indices_",
            ],
        )
        X_work, _ = self._to_array_and_names(X)
        if X_work.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns, got {X_work.shape[1]}"
            )
        X_scaled = self._apply_imputation_clip_scale(X_work)
        return np.asarray(X_scaled[:, self.selected_indices_], dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names after optional top-k selection."""
        check_is_fitted(self, ["selected_feature_names_"])
        return np.asarray(self.selected_feature_names_, dtype=object)


def summarize_pima(X: pd.DataFrame, y: np.ndarray) -> str:
    """Format a plain-language summary of raw Pima data and cleaned bounds."""
    n_rows, n_cols = X.shape
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    floor_pct = 100.0 * n_neg / n_rows

    lines = [
        "Pima Indians Diabetes (OpenML ID 37)",
        f"Shape: {n_rows} patients x {n_cols} columns",
        f"Classes: {n_neg} negative (0), {n_pos} positive (1) | Always-guess floor: {floor_pct:.2f}%",
        "Zero-value counts in raw data:",
    ]
    for col in PIMA_COLUMNS:
        zero_count = int((X[col] == 0).sum())
        zero_pct = 100.0 * zero_count / n_rows
        note = "real count (kept)" if col == "preg" else (
            "hidden missing -> imputed" if col in ZERO_AS_MISSING_COLS else "no hidden zeros"
        )
        lines.append(f"  - {col:<4s}: {zero_count:3d} ({zero_pct:5.1f}%) [{note}]")

    prep = PimaPreprocessor()
    X_clean = prep.fit_transform(X, y)
    lines.append(
        "After PimaPreprocessor.fit_transform(X, y): "
        f"shape={X_clean.shape}, NaNs={int(np.isnan(X_clean).sum())}, "
        f"range=[{X_clean.min():.4f}, {X_clean.max():.4f}] (pi={np.pi:.4f})"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    X_raw, y_raw = load_pima()
    print(summarize_pima(X_raw, y_raw))

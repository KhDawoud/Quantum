"""Classical linear baseline: Logistic Regression.

Contract: expose ``make_model(**kwargs)`` and ``PARAM_GRID`` (<= 8 combinations).
"""

from __future__ import annotations

from typing import Any

# Placeholder grid (at most 8 combinations, e.g. C values and class_weight).
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def make_model(**kwargs: Any):
    """Return a scikit-learn LogisticRegression estimator (random_state=42)."""
    raise NotImplementedError(
        "Implement LogisticRegression in src/model_logreg.py."
    )

"""Classical nonlinear baseline: RBF Support Vector Machine.

Contract: expose ``make_model(**kwargs)`` and ``PARAM_GRID`` (<= 8 combinations).
"""

from __future__ import annotations

from typing import Any

# Placeholder grid (at most 8 combinations, calibrated for inputs in [0, pi]).
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.5, 2.0],
    "gamma": [0.01, 0.05, 0.2, "scale"],
}


def make_model(**kwargs: Any):
    """Return a scikit-learn SVC(kernel='rbf', random_state=42)."""
    raise NotImplementedError("Implement RBF SVC in src/model_rbf_svm.py.")

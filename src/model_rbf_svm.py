"""Classical nonlinear baseline: RBF Support Vector Machine.

Contract (PROJECT_CONTEXT.md):
  - Expose ``make_model(**kwargs)`` returning a scikit-learn fit/predict estimator.
  - Expose ``PARAM_GRID`` with <= 8 hyperparameter combinations.
  - Leave preprocessing, splitting, and scoring to ``src/evaluate.py``.

Plain-language picture of the model:
  1. SIMILARITY : K(x, z) = exp(-gamma * ||x - z||^2) compares scaled patients.
  2. MARGIN     : support vectors define a nonlinear separating boundary.
  3. CONTROLS   : C controls the penalty for margin violations; gamma controls
     how quickly similarity falls with distance (larger means more local).

Source: scikit-learn SVC API:
  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

from __future__ import annotations

from typing import Any

from sklearn.svm import SVC

from src import RANDOM_SEED

# 2 x 4 = 8 combinations, fixed before inspecting outer-test performance.
# Gamma is an RBF distance coefficient, not the quantum encoding angle scale.
PARAM_GRID: dict[str, list[Any]] = {
    "C": [1, 10],
    "gamma": [0.01, 0.1, 0.5, 1.0],
}


def make_model(**kwargs: Any) -> SVC:
    """Return a fresh RBF SVC, with optional estimator-parameter overrides.

    Probability calibration is off: the shared harness only needs class labels.
    This avoids SVC's additional internal calibration CV. The seed is retained
    for the project contract (it affects SVC only when probability=True).
    """
    params: dict[str, Any] = {
        "kernel": "rbf",
        "C": 1,
        "gamma": 0.1,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }
    params.update(kwargs)
    return SVC(**params)

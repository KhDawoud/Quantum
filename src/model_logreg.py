"""Classical linear baseline: Logistic Regression.

Contract (PROJECT_CONTEXT.md):
  - Expose ``make_model(**kwargs)`` returning a scikit-learn estimator with
    ``fit(X, y)``, ``predict(X)``, and ``predict_proba(X)``.
  - Expose ``PARAM_GRID`` with <= 8 hyperparameter combinations.
  - Never put train/test splitting or evaluation loops inside model files;
    every classifier plugs into ``src/evaluate.py``.

Plain-language picture of the model:
  1. WEIGHTED SUM : multiply each scaled clinical feature (in [0, pi]) by a
     learned weight and add a bias offset (intercept).
  2. SIGMOID      : squash that linear score into a probability in (0, 1).
  3. BOUNDARY     : predicting class 1 when P >= 0.5 draws a flat (linear)
     decision boundary through the 8-feature space, serving as our straight-line
     reference against RBF SVM and the quantum models.

Sources adapted:
  - scikit-learn ``LogisticRegression`` API and L2 default configuration:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from src import RANDOM_SEED

# 4 x 2 = 8 combinations (uses the full fair <= 8 tuning budget).
# Spans 4 orders of magnitude of inverse regularization strength C for [0, pi]
# inputs, and lets 3-fold balanced-accuracy tuning choose between unweighted and
# class-balanced loss on the 65/35 Pima split.
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def make_model(**kwargs: Any) -> LogisticRegression:
    """Return a scikit-learn LogisticRegression estimator (random_state=42).

    Default configuration uses ``solver='lbfgs'`` with standard L2
    regularization (default in scikit-learn without passing the deprecated
    ``penalty`` argument) and ``max_iter=1000`` for clean convergence.
    """
    params: dict[str, Any] = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
    }
    params.update(kwargs)
    return LogisticRegression(**params)

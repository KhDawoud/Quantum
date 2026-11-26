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

# 8 combinations (uses the full fair <= 8 tuning budget).
# Matches the Nov 26 team decision: balanced class weighting is fixed inside
# make_model() across all models rather than tuned, leaving the full budget
# to search inverse regularization strength C on [0, pi]-scaled inputs.
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
}


def make_model(**kwargs: Any) -> LogisticRegression:
    """Return a scikit-learn LogisticRegression estimator (random_state=42).

    Default configuration fixes ``class_weight='balanced'`` (matching B, C, and
    D), uses ``solver='lbfgs'`` with standard L2 regularization (omitting the
    deprecated ``penalty`` argument), and sets ``max_iter=1000``.
    """
    params: dict[str, Any] = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }
    params.update(kwargs)
    return LogisticRegression(**params)

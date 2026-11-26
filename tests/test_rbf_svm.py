"""Tests for the nonlinear classical baseline (src/model_rbf_svm.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src import RANDOM_SEED
from src.data import PimaPreprocessor, load_pima
from src.evaluate import evaluate
from src.model_rbf_svm import PARAM_GRID, make_model


def test_param_grid_budget_and_factory_contract():
    model = make_model()
    assert isinstance(model, SVC)
    assert model.kernel == "rbf"
    assert model.random_state == RANDOM_SEED
    assert model.get_params()["class_weight"] == "balanced"
    assert "class_weight" not in PARAM_GRID
    assert not hasattr(model, "predict_proba")
    assert PARAM_GRID == {"C": [1, 10], "gamma": [0.01, 0.1, 0.5, 1.0]}
    assert int(np.prod([len(values) for values in PARAM_GRID.values()])) == 8
    assert set(PARAM_GRID) <= set(model.get_params())
    overridden = make_model(C=10, gamma=0.5)
    assert overridden.C == 10
    assert overridden.gamma == 0.5
    assert clone(overridden).get_params() == overridden.get_params()
    assert make_model() is not model


def test_balanced_weights_are_computed_from_fit_labels():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0] * 7 + [1] * 3)
    model = make_model().fit(X, y)
    np.testing.assert_allclose(model.class_weight_, [10 / (2 * 7), 10 / (2 * 3)])
    # Reusing the estimator with different training labels must recompute weights.
    model.fit(X, np.array([0] * 5 + [1] * 5))
    np.testing.assert_allclose(model.class_weight_, [1.0, 1.0])


def test_rbf_svm():
    """Standalone synthetic smoke test; scaling is learned inside each fold."""
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, n_redundant=2,
        weights=[0.65, 0.35], random_state=RANDOM_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED,
    )
    pipe = Pipeline([
        ("prep", MinMaxScaler(feature_range=(0, np.pi), clip=True)),
        ("clf", make_model()),
    ])
    search = GridSearchCV(
        clone(pipe),
        {f"clf__{key}": values for key, values in PARAM_GRID.items()},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
        scoring="balanced_accuracy", error_score="raise",
    ).fit(X_train, y_train)
    predictions = search.predict(X_test)
    print("Synthetic data only (not competition results)")
    print("Best parameters:", search.best_params_)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, predictions))
    print("Macro F1:", f1_score(y_test, predictions, average="macro"))
    assert predictions.shape == y_test.shape
    assert set(np.unique(predictions)) == {0, 1}
    assert len(search.cv_results_["params"]) == 8
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    prep = search.best_estimator_.named_steps["prep"]
    np.testing.assert_allclose(prep.data_min_, X_train.min(axis=0))
    np.testing.assert_allclose(prep.data_max_, X_train.max(axis=0))
    assert prep.transform(X_test).min() >= 0
    assert prep.transform(X_test).max() <= np.pi + 1e-12


def test_nonlinear_learning_and_rbf_decision_function():
    X, y = make_moons(n_samples=300, noise=0.12, random_state=RANDOM_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=RANDOM_SEED,
    )
    prep = MinMaxScaler(feature_range=(0, np.pi), clip=True).fit(X_train)
    X_train, X_test = prep.transform(X_train), prep.transform(X_test)
    model = make_model(C=10, gamma=1.0).fit(X_train, y_train)
    assert balanced_accuracy_score(y_test, model.predict(X_test)) > 0.90

    # Independently reconstruct sum_i(alpha_i*y_i*K(x, sv_i)) + intercept.
    squared_distances = ((X_test[:, None, :] - model.support_vectors_) ** 2).sum(axis=2)
    kernel = np.exp(-model.gamma * squared_distances)
    expected = kernel @ model.dual_coef_[0] + model.intercept_[0]
    np.testing.assert_allclose(model.decision_function(X_test), expected, atol=1e-10)
    repeated = make_model(C=10, gamma=1.0).fit(X_train, y_train)
    np.testing.assert_array_equal(model.predict(X_test), repeated.predict(X_test))


@pytest.mark.parametrize("n_features", [4, 6, 8])
def test_shared_preprocessor_with_reduced_features(n_features):
    X, y = load_pima()
    train_idx, test_idx = next(
        StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(X, y)
    )
    pipe = Pipeline([
        ("prep", PimaPreprocessor(n_features=n_features)),
        ("clf", make_model()),
    ]).fit(X.iloc[train_idx], y[train_idx])
    assert pipe.named_steps["clf"].n_features_in_ == n_features
    assert pipe.predict(X.iloc[test_idx]).shape == (len(test_idx),)


def test_shared_evaluate_csv_contract(tmp_path):
    summary = evaluate(
        "rbf_svm_test", make_model, PARAM_GRID,
        n_splits_to_run=1, results_dir=tmp_path,
    )
    results = pd.read_csv(summary["results_path"])
    assert summary["n_splits"] == 1
    assert summary["quick"] is False
    assert list(results.columns) == [
        "split", "n_train", "n_test", "accuracy", "balanced_accuracy",
        "macro_f1", "train_time_sec", "best_params", "selected_features",
    ]
    assert results.loc[0, "n_train"] + results.loc[0, "n_test"] == 768
    chosen = json.loads(results.loc[0, "best_params"])
    assert set(chosen) == set(PARAM_GRID)
    assert all(chosen[key] in values for key, values in PARAM_GRID.items())
    assert results.loc[0, "balanced_accuracy"] == pytest.approx(
        summary["balanced_accuracy_mean"]
    )


if __name__ == "__main__":
    test_rbf_svm()

"""Tests for the classical linear baseline (src/model_logreg.py)."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import PimaPreprocessor, load_pima
from src.model_logreg import PARAM_GRID, make_model


def test_param_grid_budget_and_keys():
    assert PARAM_GRID
    assert all(len(values) > 0 for values in PARAM_GRID.values())
    assert "class_weight" not in PARAM_GRID
    assert PARAM_GRID == {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
    n_combos = int(np.prod([len(values) for values in PARAM_GRID.values()]))
    assert 1 <= n_combos <= 8
    assert n_combos == 8
    assert set(PARAM_GRID) <= set(make_model().get_params())


def test_make_model_defaults_and_overrides():
    model = make_model()
    assert isinstance(model, LogisticRegression)
    assert model.random_state == 42
    assert model.solver == "lbfgs"
    assert model.max_iter == 1000
    assert model.class_weight == "balanced"

    overridden = make_model(C=0.5, class_weight=None)
    assert isinstance(overridden, LogisticRegression)
    assert overridden.C == 0.5
    assert overridden.class_weight is None
    assert overridden.random_state == 42


def test_balanced_class_weighting_adapts_to_fit_labels():
    rng = np.random.default_rng(42)
    X_base = rng.normal(0.0, 1.0, size=(100, 4))

    y_imb = np.array([0] * 70 + [1] * 30)
    X_imb_raw = X_base.copy()
    X_imb_raw[y_imb == 1, 0] += 0.8
    X_imb = (
        (X_imb_raw - X_imb_raw.min(axis=0))
        / (X_imb_raw.max(axis=0) - X_imb_raw.min(axis=0))
        * np.pi
    )
    bal_imb = make_model().fit(X_imb, y_imb)
    unw_imb = make_model(class_weight=None).fit(X_imb, y_imb)
    assert bal_imb.intercept_[0] > unw_imb.intercept_[0]
    assert bal_imb.predict_proba(X_imb)[:, 1].mean() > unw_imb.predict_proba(X_imb)[:, 1].mean()

    y_eq = np.array([0] * 50 + [1] * 50)
    X_eq_raw = X_base.copy()
    X_eq_raw[y_eq == 1, 0] += 0.8
    X_eq = (
        (X_eq_raw - X_eq_raw.min(axis=0))
        / (X_eq_raw.max(axis=0) - X_eq_raw.min(axis=0))
        * np.pi
    )
    bal_eq = make_model().fit(X_eq, y_eq)
    unw_eq = make_model(class_weight=None).fit(X_eq, y_eq)
    np.testing.assert_allclose(bal_eq.coef_, unw_eq.coef_, atol=1e-10)
    np.testing.assert_allclose(bal_eq.intercept_, unw_eq.intercept_, atol=1e-10)


def test_fit_predict_and_predict_proba_on_linearly_separable_data():
    rng = np.random.default_rng(42)
    X_raw = rng.normal(0.0, 1.0, size=(120, 8))
    y = (X_raw[:, 0] + X_raw[:, 1] > 0.0).astype(int)
    X_raw[y == 1, :2] += 1.2
    X_raw[y == 0, :2] -= 1.2
    X_scaled = (
        (X_raw - X_raw.min(axis=0))
        / (X_raw.max(axis=0) - X_raw.min(axis=0))
        * np.pi
    )

    model = make_model().fit(X_scaled[:90], y[:90])
    preds = model.predict(X_scaled[90:])
    proba = model.predict_proba(X_scaled[90:])

    assert preds.shape == (30,)
    assert set(np.unique(preds)) <= {0, 1}
    assert float((preds == y[90:]).mean()) >= 0.90

    assert proba.shape == (30, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_deterministic_fit_with_seed_42():
    X, y = load_pima()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))

    prep = PimaPreprocessor().fit(X.iloc[train_idx], y[train_idx])
    X_train = prep.transform(X.iloc[train_idx])
    X_test = prep.transform(X.iloc[test_idx])

    model_a = make_model().fit(X_train, y[train_idx])
    model_b = make_model().fit(X_train, y[train_idx])

    np.testing.assert_array_equal(model_a.coef_, model_b.coef_)
    np.testing.assert_array_equal(model_a.intercept_, model_b.intercept_)
    np.testing.assert_array_equal(
        model_a.predict_proba(X_test), model_b.predict_proba(X_test)
    )


def test_pipeline_clone_and_gridsearch_on_pima_no_warnings():
    X, y = load_pima()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(outer.split(X, y))

    pipe = Pipeline([("prep", PimaPreprocessor()), ("clf", make_model())])
    cloned = clone(pipe)
    prefixed_grid = {f"clf__{key}": values for key, values in PARAM_GRID.items()}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings("error", category=FutureWarning, module=r"sklearn\.")
        warnings.filterwarnings("error", category=DeprecationWarning, module=r"sklearn\.")
        search = GridSearchCV(
            cloned,
            prefixed_grid,
            cv=inner,
            scoring="balanced_accuracy",
        )
        search.fit(X.iloc[train_idx], y[train_idx])

    assert set(search.best_params_) == set(prefixed_grid)
    preds = search.predict(X.iloc[test_idx])
    assert preds.shape == (len(test_idx),)
    assert set(np.unique(preds)) == {0, 1}


def test_reduced_features_and_moons_compatibility():
    X, y = load_pima()
    pipe_pima_4 = Pipeline(
        [("prep", PimaPreprocessor(n_features=4)), ("clf", make_model())]
    )
    pipe_pima_4.fit(X.iloc[:500], y[:500])
    preds_4 = pipe_pima_4.predict(X.iloc[500:])
    assert preds_4.shape == (268,)
    assert set(np.unique(preds_4)) <= {0, 1}

    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    pipe_moons = Pipeline(
        [("prep", PimaPreprocessor(zero_as_missing_cols=())), ("clf", make_model())]
    )
    pipe_moons.fit(X_moons[:150], y_moons[:150])
    preds_moons = pipe_moons.predict(X_moons[150:])
    assert preds_moons.shape == (50,)
    assert set(np.unique(preds_moons)) <= {0, 1}

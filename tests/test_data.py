"""Tests for dataset integrity and leakage-free preprocessing (src/data.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import (
    PIMA_COLUMNS,
    ZERO_AS_MISSING_COLS,
    PimaPreprocessor,
    load_pima,
    summarize_pima,
)


def test_load_pima_integrity():
    X, y = load_pima()
    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == list(PIMA_COLUMNS)
    assert X.shape == (768, 8)
    assert y.shape == (768,)
    assert set(np.unique(y)) == {0, 1}
    assert int((y == 0).sum()) == 500
    assert int((y == 1).sum()) == 268
    # Known raw zero counts from the Build Guide / dataset screening
    assert int((X["skin"] == 0).sum()) == 227
    assert int((X["insu"] == 0).sum()) == 374
    assert int((X["preg"] == 0).sum()) == 111


def test_hidden_zeros_imputed_but_preg_preserved():
    X, y = load_pima()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_test = X.iloc[test_idx].copy()
    raw_train_snapshot = X_train.copy()

    prep = PimaPreprocessor()
    X_train_scaled = prep.fit_transform(X_train, y_train)
    X_test_scaled = prep.transform(X_test)

    # Caller DataFrames must never be mutated in place
    pd.testing.assert_frame_equal(X_train, raw_train_snapshot)

    # Every clinical column median must be strictly positive (zeros were ignored)
    for col in ZERO_AS_MISSING_COLS:
        col_idx = list(PIMA_COLUMNS).index(col)
        assert prep.medians_[col_idx] > 0.0

    # 'preg' had real zeros (min == 0), so zero pregnancies stay at the 0.0 lower bound
    preg_idx = list(PIMA_COLUMNS).index("preg")
    assert prep.min_[preg_idx] == 0.0
    zero_preg_rows = (X_test["preg"] == 0).to_numpy()
    assert zero_preg_rows.any()
    assert np.allclose(X_test_scaled[zero_preg_rows, preg_idx], 0.0)

    # Hidden zeros in 'insu' are NOT left at 0.0 — they get imputed to the median (> 0)
    insu_idx = list(PIMA_COLUMNS).index("insu")
    zero_insu_rows = (X_test["insu"] == 0).to_numpy()
    assert zero_insu_rows.any()
    assert np.all(X_test_scaled[zero_insu_rows, insu_idx] > 0.0)
    assert X_train_scaled.shape == (len(train_idx), 8)


def test_zero_test_data_leakage():
    X, y = load_pima()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test_clean = X.iloc[test_idx].copy()

    X_test_extreme = X_test_clean.copy()
    X_test_extreme.iloc[:, :] = 1e9
    X_test_zeros = X_test_clean.copy()
    X_test_zeros.iloc[:, :] = 0.0

    prep_a = PimaPreprocessor(n_features=6).fit(X_train, y_train)
    prep_b = PimaPreprocessor(n_features=6).fit(X_train, y_train)
    prep_c = PimaPreprocessor(n_features=6).fit(X_train, y_train)

    # Transform wildly different test sets (must not alter learned state)
    _ = prep_a.transform(X_test_clean)
    _ = prep_b.transform(X_test_extreme)
    _ = prep_c.transform(X_test_zeros)

    for attr in ("medians_", "p01_", "p99_", "min_", "max_", "selected_indices_"):
        np.testing.assert_array_equal(getattr(prep_a, attr), getattr(prep_b, attr))
        np.testing.assert_array_equal(getattr(prep_a, attr), getattr(prep_c, attr))

    np.testing.assert_array_equal(prep_a.transform(X_train), prep_b.transform(X_train))


def test_output_strictly_bounded_in_zero_to_pi():
    X, y = load_pima()
    X_train, y_train = X.iloc[:600], y[:600]

    X_extreme_test = X.iloc[600:].copy()
    X_extreme_test.iloc[0, :] = -1e6
    X_extreme_test.iloc[1, :] = 1e6
    X_extreme_test.iloc[2, :] = 0.0
    X_extreme_test.iloc[3, :] = np.nan

    prep = PimaPreprocessor().fit(X_train, y_train)
    X_out = prep.transform(X_extreme_test)

    assert not np.isnan(X_out).any()
    assert not np.isinf(X_out).any()
    assert np.all(X_out >= 0.0)
    assert np.all(X_out <= np.pi)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_top_k_feature_selection(k: int):
    X, y = load_pima()
    prep = PimaPreprocessor(n_features=k)
    X_out = prep.fit_transform(X, y)
    assert X_out.shape == (len(X), k)
    assert len(prep.get_feature_names_out()) == k
    assert np.all(X_out >= 0.0) and np.all(X_out <= np.pi)


def test_feature_selection_validation_errors():
    X, y = load_pima()
    with pytest.raises(ValueError, match="Target vector y is required"):
        PimaPreprocessor(n_features=4).fit(X)
    with pytest.raises(ValueError, match="n_features must be between"):
        PimaPreprocessor(n_features=0).fit(X, y)
    with pytest.raises(ValueError, match="n_features must be between"):
        PimaPreprocessor(n_features=99).fit(X, y)


def test_sklearn_pipeline_and_moons_compatibility():
    X, y = load_pima()
    pipe = Pipeline(
        [
            ("prep", PimaPreprocessor(n_features=6)),
            ("clf", LogisticRegression(random_state=42, max_iter=500)),
        ]
    )
    cloned = clone(pipe)
    cloned.fit(X.iloc[:500], y[:500])
    preds = cloned.predict(X.iloc[500:])
    assert preds.shape == (268,)

    # Positive control compatibility (2D artificial make_moons dataset)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    moons_prep = PimaPreprocessor(zero_as_missing_cols=())
    X_moons_scaled = moons_prep.fit_transform(X_moons, y_moons)
    assert X_moons_scaled.shape == (200, 2)
    assert np.all(X_moons_scaled >= 0.0) and np.all(X_moons_scaled <= np.pi)


def test_summarize_pima_output():
    X, y = load_pima()
    summary = summarize_pima(X, y)
    assert "768 patients x 8 columns" in summary
    assert "65.10%" in summary

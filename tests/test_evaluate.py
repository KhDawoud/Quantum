"""Tests for the shared evaluation harness (src/evaluate.py)."""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

import src.evaluate as ev
from src.data import PimaPreprocessor, load_pima
from src.evaluate import RESULT_COLUMNS, compare_models, evaluate, get_splits

GRID = {"C": [0.1, 1.0]}


def make_lr() -> LogisticRegression:
    return LogisticRegression(max_iter=500, random_state=42)


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch):
    """Never write test CSVs into the real results/ folder."""
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)


# ---------------------------------------------------------------- get_splits
def test_splits_identical_on_every_call():
    X, y = load_pima()
    a = get_splits(X, y)
    b = get_splits(X, y)
    assert len(a) == len(b) == 25
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_every_patient_tested_exactly_5_times_in_25_splits():
    X, y = load_pima()
    counts = Counter(i for _, te in get_splits(X, y) for i in te)
    assert len(counts) == 768
    assert set(counts.values()) == {5}


def test_every_patient_tested_once_with_one_repeat():
    X, y = load_pima()
    splits = get_splits(X, y, n_repeats=1)
    assert len(splits) == 5
    counts = Counter(i for _, te in splits for i in te)
    assert len(counts) == 768
    assert set(counts.values()) == {1}


def test_first_5_of_25_equal_one_repeat():
    X, y = load_pima()
    full = get_splits(X, y, n_repeats=5)[:5]
    one = get_splits(X, y, n_repeats=1)
    for (tr_a, te_a), (tr_b, te_b) in zip(full, one):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_train_and_test_never_overlap_and_are_stratified():
    X, y = load_pima()
    for tr, te in get_splits(X, y):
        assert not set(tr) & set(te)
        assert abs(y[te].mean() - y.mean()) < 0.02


# ------------------------------------------------------------------ evaluate
def test_evaluate_writes_expected_csv():
    out = evaluate("t_basic", make_lr, GRID, n_repeats=1)
    df = pd.read_csv(out["csv_path"])
    assert list(df.columns) == list(RESULT_COLUMNS)
    assert len(df) == 5 == out["n_splits"]
    assert (df["n_train"] + df["n_test"] == 768).all()
    assert df["balanced_accuracy"].between(0, 1).all()
    for params in df["best_params"]:
        assert json.loads(params)["C"] in GRID["C"]
    for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
        assert out[f"{metric}_mean"] == pytest.approx(df[metric].mean())
        assert out[f"{metric}_std"] == pytest.approx(df[metric].std(ddof=1))


def test_evaluate_default_gives_25_rows():
    out = evaluate("t_full", make_lr, {"C": [1.0]})
    assert out["n_splits"] == 25
    assert len(pd.read_csv(out["csv_path"])) == 25


def test_quick_mode_uses_first_5_splits_and_first_grid_values():
    out = evaluate("t_quick", make_lr, GRID, quick=True)
    df = pd.read_csv(out["csv_path"])
    assert len(df) == 5
    assert all(json.loads(p) == {"C": 0.1} for p in df["best_params"])
    assert (df["tune_seconds"] == 0).all()
    # The same 5 splits as the first 5 of the full run.
    X, y = load_pima()
    first5 = get_splits(X, y)[:5]
    assert list(df["n_test"]) == [len(te) for _, te in first5]


def test_fixed_params_skip_tuning():
    out = evaluate("t_fixed", make_lr, GRID, fixed_params={"C": 10.0}, n_repeats=1)
    df = pd.read_csv(out["csv_path"])
    assert all(json.loads(p) == {"C": 10.0} for p in df["best_params"])
    assert (df["tune_seconds"] == 0).all()


def test_train_size_mode_fixed_test_set():
    out = evaluate(
        "t_size", make_lr, GRID, train_size=60, fixed_params={"C": 1.0}, n_draws=4
    )
    df = pd.read_csv(out["csv_path"])
    assert len(df) == 4
    assert (df["n_train"] == 60).all()
    assert (df["n_test"] == 154).all()

    X, y = load_pima()
    draws = ev._train_size_splits(y, 60, 4)
    tests = [tuple(te) for _, te in draws]
    assert len(set(tests)) == 1  # one fixed test set
    test_set = set(draws[0][1])
    pool = set(range(768)) - test_set
    for tr, _ in draws:
        assert set(tr) <= pool  # drawn only from the other 80%
        assert abs(y[tr].mean() - y[list(pool)].mean()) < 0.1
    assert len({tuple(tr) for tr, _ in draws}) == 4  # draws differ
    # reproducible
    again = ev._train_size_splits(y, 60, 4)
    for (a, _), (b, _) in zip(draws, again):
        np.testing.assert_array_equal(a, b)


def test_train_size_too_big_rejected():
    with pytest.raises(ValueError, match="larger than the training pool"):
        evaluate("t_big", make_lr, GRID, train_size=700, fixed_params={"C": 1.0})


# ------------------------------------------------------------------- leakage
def test_preprocessor_only_fits_on_training_rows(monkeypatch):
    """Record every patient row the preprocessor and the model ever fit on."""
    prep_fits: list[set[int]] = []
    model_fit_sizes: list[int] = []

    class SpyPreprocessor(PimaPreprocessor):
        def fit(self, X, y=None):
            prep_fits.append(set(X.index))  # patient ids 0..767
            return super().fit(X, y)

    class SpyModel(LogisticRegression):
        def fit(self, X, y, *args, **kwargs):
            model_fit_sizes.append(len(X))
            return super().fit(X, y, *args, **kwargs)

    monkeypatch.setattr(ev, "PimaPreprocessor", SpyPreprocessor)
    X, y = load_pima()
    splits = get_splits(X, y, n_repeats=1)

    # n_features=4 also exercises the column ranking (uses labels).
    evaluate(
        "t_leak",
        lambda: SpyModel(max_iter=500, random_state=42),
        GRID,
        n_features=4,
        n_repeats=1,
    )

    assert len(prep_fits) == len(splits)  # exactly one fit per split
    for fitted_rows, (train_idx, test_idx) in zip(prep_fits, splits):
        assert fitted_rows == set(train_idx)
        assert not fitted_rows & set(test_idx)
    # Final model fit never sees more than the training rows either.
    assert max(model_fit_sizes) <= max(len(tr) for tr, _ in splits)


def test_scores_do_not_depend_on_test_labels_of_other_rows(monkeypatch):
    """Scramble labels of rows that are in the test set of split 0: the
    chosen settings and the final fit for split 0 must not change."""
    X, y = load_pima()
    _, test0 = get_splits(X, y, n_repeats=1)[0]
    y_bad = y.copy()
    y_bad[test0] = 1 - y_bad[test0]

    base = evaluate("t_clean", make_lr, GRID, n_repeats=1)
    monkeypatch.setattr(ev, "load_pima", lambda: (X, y_bad))
    bad = evaluate("t_bad", make_lr, GRID, n_repeats=1)

    df_ok, df_bad = pd.read_csv(base["csv_path"]), pd.read_csv(bad["csv_path"])
    # Split 0 trains on rows whose labels are untouched -> same tuned settings.
    assert df_ok.loc[0, "best_params"] == df_bad.loc[0, "best_params"]
    # ... but its test score changes because the test labels changed.
    assert df_ok.loc[0, "accuracy"] != df_bad.loc[0, "accuracy"]


# ------------------------------------------------------------ compare_models
def _fake_results(scores, n_train=600, n_test=150, metric="balanced_accuracy"):
    return pd.DataFrame(
        {
            "split": range(len(scores)),
            "n_train": n_train,
            "n_test": n_test,
            metric: scores,
        }
    )


def test_compare_identical_results():
    df = _fake_results([0.7, 0.72, 0.68, 0.75, 0.71])
    res = compare_models(df, df.copy())
    assert res["mean_diff"] == 0.0
    assert res["p"] == 1.0
    assert res["t"] == 0.0
    assert res["n_splits"] == 5


def test_compare_matches_hand_calculation():
    a = _fake_results([0.80, 0.78, 0.82, 0.79, 0.81])
    b = _fake_results([0.75, 0.77, 0.78, 0.80, 0.74])
    # Differences: .05 .01 .04 -.01 .07 -> mean .032
    # Sum of squared deviations = .00408 -> variance (ddof=1) = .00102
    # Correction factor = 1/5 + 150/600 = .45
    # t = .032 / sqrt(.45 * .00102)
    expected_t = 0.032 / math.sqrt(0.45 * 0.00102)
    expected_p = 2 * stats.t.sf(abs(expected_t), df=4)
    res = compare_models(a, b)
    assert res["mean_diff"] == pytest.approx(0.032)
    assert res["std_diff"] == pytest.approx(math.sqrt(0.00102))
    assert res["t"] == pytest.approx(expected_t)
    assert res["p"] == pytest.approx(expected_p)
    assert 0.1 < res["p"] < 0.3  # sanity: not significant
    # Swapping the models flips the sign but not p.
    swapped = compare_models(b, a)
    assert swapped["t"] == pytest.approx(-expected_t)
    assert swapped["p"] == pytest.approx(expected_p)


def test_compare_zero_variance_nonzero_gap():
    a = _fake_results([0.8] * 5)
    b = _fake_results([0.7] * 5)
    res = compare_models(a, b)
    assert res["p"] == 0.0 and res["t"] == math.inf


def test_compare_reads_csv_paths_and_rejects_different_splits(tmp_path):
    a = _fake_results([0.7, 0.8, 0.75, 0.77, 0.72])
    b = _fake_results([0.6, 0.7, 0.65, 0.71, 0.66])
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    assert compare_models(str(pa), str(pb))["n_splits"] == 5

    with pytest.raises(ValueError, match="identical splits"):
        compare_models(a, b.iloc[:4])
    with pytest.raises(ValueError, match="identical splits"):
        compare_models(a, _fake_results([0.6] * 5, n_test=151))
    with pytest.raises(ValueError, match="no 'f1_macro'"):
        compare_models(a, b, metric="f1_macro")


def test_compare_end_to_end_with_real_results():
    a = evaluate("t_cmp_a", make_lr, {"C": [1.0]}, n_repeats=2)
    b = evaluate("t_cmp_b", make_lr, {"C": [0.001]}, n_repeats=2)
    res = compare_models(a["csv_path"], b["csv_path"])
    assert res["n_splits"] == 10
    assert 0.0 <= res["p"] <= 1.0

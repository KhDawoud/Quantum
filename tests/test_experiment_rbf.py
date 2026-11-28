"""Check training-size protocol, especially tuning/test separation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import experiment_rbf as experiment
from src.evaluate import get_trainsize_splits


def test_exp1_matches_team_splits_and_features_without_rewriting_baseline(monkeypatch, tmp_path):
    baseline = Path("results/rbf_svm.csv")
    before = baseline.read_bytes()
    calls = []
    real_evaluate = experiment.evaluate

    def checked_evaluate(name, factory, grid, **kwargs):
        calls.append((name, kwargs))
        assert grid == {"C": [1, 10], "gamma": [0.01, 0.1, 0.5, 1.0]}
        return real_evaluate(name, factory, grid, **kwargs)

    monkeypatch.setattr(experiment, "evaluate", checked_evaluate)
    result = experiment.run_exp1_qubits_rbf(tmp_path, tmp_path, baseline)
    assert [name for name, _ in calls] == ["exp1_rbf_svm_q4", "exp1_rbf_svm_q6"]
    assert all(kw["n_splits_to_run"] == 5 and kw["quick"] is False for _, kw in calls)
    for k in (4, 6):
        actual = pd.read_csv(tmp_path / f"exp1_rbf_svm_q{k}.csv")
        reference = pd.read_csv(f"results/exp1_logreg_q{k}.csv")
        pd.testing.assert_frame_equal(
            actual[["split", "n_train", "n_test", "selected_features"]],
            reference[["split", "n_train", "n_test", "selected_features"]],
        )
        assert result["comparisons"][f"q{k}_vs_q8"]["n_splits"] == 5
    pd.testing.assert_frame_equal(result["q8"], pd.read_csv(baseline).iloc[:5])
    assert baseline.read_bytes() == before
    assert not (tmp_path / "exp1_rbf_svm_q8.csv").exists()
    assert result["figure_path"].is_file()


def test_exp1_rejects_unpaired_baseline_before_running(monkeypatch, tmp_path):
    baseline = pd.read_csv("results/rbf_svm.csv").iloc[1:6]
    path = tmp_path / "baseline.csv"
    baseline.to_csv(path, index=False)
    monkeypatch.setattr(experiment, "evaluate", lambda *a, **kw: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="first five splits"):
        experiment.run_exp1_qubits_rbf(tmp_path, tmp_path, path)


def test_trainsize_tunes_only_once_outside_fixed_test(monkeypatch, tmp_path):
    X, y = experiment.load_pima()
    pool, test = experiment.get_fixed_test_split(X, y)
    calls = []
    actual_tune = experiment.tune_hyperparameters

    def checked_tune(factory, grid, X_train, y_train):
        np.testing.assert_array_equal(X_train.index, X.iloc[pool].index)
        np.testing.assert_array_equal(y_train, y[pool])
        assert set(X_train.index).isdisjoint(X.iloc[test].index)
        params = actual_tune(factory, grid, X_train, y_train)
        calls.append(params)
        return params

    monkeypatch.setattr(experiment, "tune_hyperparameters", checked_tune)
    monkeypatch.setattr(experiment, "plot_trainsize", lambda *args: None)
    result = experiment.run_exp2_trainsize_rbf(tmp_path, tmp_path)
    assert len(calls) == 1
    assert len(result) == 40
    assert set(result.n_test) == {154}
    assert set(result.tuning_pool_size) == {614}
    assert all(json.loads(value) == calls[0] for value in result.best_params)
    for size in (30, 60, 120, 200):
        assert len(result[result.n_train == size]) == 10
        for train_idx, test_idx in get_trainsize_splits(X, y, size):
            np.testing.assert_array_equal(test_idx, test)
            assert set(train_idx) <= set(pool)
            assert set(train_idx).isdisjoint(test_idx)
    assert (tmp_path / "exp2_trainsize_rbf.csv").exists()

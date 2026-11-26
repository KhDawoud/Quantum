"""Check training-size protocol, especially tuning/test separation."""

import json
import numpy as np

from src import experiment_rbf as experiment
from src.evaluate import get_trainsize_splits


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

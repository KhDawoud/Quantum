"""Tests for Person A's Experiment 1 and Experiment 2 Logistic Regression runners."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.data import PIMA_COLUMNS, load_pima
from src.evaluate import EXPERIMENT_EVAL_SPLITS, get_fixed_test_split


def test_exp1_logreg_uses_shared_evaluate_and_experiment_splits(monkeypatch, tmp_path):
    import src.experiment_logreg as experiment

    calls: list[dict[str, Any]] = []
    real_evaluate = experiment.evaluate

    def spy_evaluate(name, *args, **kwargs):
        calls.append(
            {
                "name": name,
                "n_features": kwargs.get("n_features"),
                "n_splits_to_run": kwargs.get("n_splits_to_run"),
                "quick": kwargs.get("quick"),
            }
        )
        return real_evaluate(name, *args, **kwargs)

    monkeypatch.setattr(experiment, "evaluate", spy_evaluate)

    result = experiment.run_exp1_qubits_logreg(
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )

    assert len(calls) == 2
    assert [call["name"] for call in calls] == ["exp1_logreg_q4", "exp1_logreg_q6"]
    assert [call["n_features"] for call in calls] == [4, 6]
    assert all(call["n_splits_to_run"] == EXPERIMENT_EVAL_SPLITS == 5 for call in calls)
    assert all(call["quick"] is False for call in calls)

    assert set(result) == {"q4", "q6", "q8", "comparisons", "figure_path"}
    assert set(result["comparisons"]) == {"q4_vs_q8", "q6_vs_q8"}

    for k, csv_name in ((4, "exp1_logreg_q4.csv"), (6, "exp1_logreg_q6.csv")):
        csv_path = tmp_path / csv_name
        assert csv_path.exists()
        sub = pd.read_csv(csv_path)
        assert len(sub) == 5
        assert list(sub["split"]) == [0, 1, 2, 3, 4]
        assert (sub["n_train"] + sub["n_test"] == 768).all()
        for sel in sub["selected_features"]:
            cols = str(sel).split(",")
            assert len(cols) == k
            assert set(cols) <= set(PIMA_COLUMNS)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert ((sub[metric] >= 0.0) & (sub[metric] <= 1.0)).all()

    baseline_first_5 = pd.read_csv("results/logreg.csv").iloc[:5].reset_index(drop=True)
    pd.testing.assert_frame_equal(result["q8"].reset_index(drop=True), baseline_first_5)

    png_path = tmp_path / "exp1_logreg.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_exp2_trainsize_logreg_tunes_once_and_saves_per_size_csvs(
    monkeypatch, tmp_path
):
    import src.experiment_logreg as experiment

    X, y = load_pima()
    pool_idx, test_idx = get_fixed_test_split(X, y)

    tune_calls: list[dict[str, Any]] = []
    real_tune = experiment.tune_hyperparameters

    def spy_tune(make_fn, grid, X_train, y_train, *args, **kwargs):
        np.testing.assert_array_equal(X_train.index, X.iloc[pool_idx].index)
        np.testing.assert_array_equal(y_train, y[pool_idx])
        assert set(X_train.index).isdisjoint(X.iloc[test_idx].index)
        best = real_tune(make_fn, grid, X_train, y_train, *args, **kwargs)
        tune_calls.append(best)
        return best

    eval_calls: list[dict[str, Any]] = []
    real_evaluate = experiment.evaluate

    def spy_evaluate(name, *args, **kwargs):
        eval_calls.append(
            {
                "name": name,
                "train_size": kwargs.get("train_size"),
                "n_draws": kwargs.get("n_draws"),
                "n_features": kwargs.get("n_features"),
                "quick": kwargs.get("quick"),
                "fixed_params": kwargs.get("fixed_params"),
                "has_n_splits_to_run": (
                    "n_splits_to_run" in kwargs
                    and kwargs["n_splits_to_run"] is not None
                ),
            }
        )
        return real_evaluate(name, *args, **kwargs)

    monkeypatch.setattr(experiment, "tune_hyperparameters", spy_tune)
    monkeypatch.setattr(experiment, "evaluate", spy_evaluate)

    out = experiment.run_exp2_trainsize_logreg(
        results_dir=tmp_path, figures_dir=tmp_path
    )

    assert len(tune_calls) == 1
    assert len(eval_calls) == 4
    assert [c["name"] for c in eval_calls] == [
        "exp2_logreg_n30",
        "exp2_logreg_n60",
        "exp2_logreg_n120",
        "exp2_logreg_n200",
    ]
    assert [c["train_size"] for c in eval_calls] == [30, 60, 120, 200]
    assert all(c["n_draws"] == 10 for c in eval_calls)
    assert all(c["n_features"] == 8 for c in eval_calls)
    assert all(c["quick"] is False for c in eval_calls)
    assert all(c["fixed_params"] == tune_calls[0] for c in eval_calls)
    assert all(c["has_n_splits_to_run"] is False for c in eval_calls)

    assert out["tuned_params"] == tune_calls[0]
    for size in (30, 60, 120, 200):
        csv_path = tmp_path / f"exp2_logreg_n{size}.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 10
        assert (df["n_train"] == size).all()
        assert (df["n_test"] == 154).all()
        assert all(json.loads(val) == tune_calls[0] for val in df["best_params"])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert ((df[metric] >= 0.0) & (df[metric] <= 1.0)).all()

    png_path = tmp_path / "exp2_logreg.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0

"""Tests for shared evaluation harness and Nadeau-Bengio significance testing."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

from src.data import load_pima
from src.evaluate import (
    DEFAULT_EVAL_SPLITS,
    EXPERIMENT_EVAL_SPLITS,
    QUICK_EVAL_SPLITS,
    compare_models,
    evaluate,
    get_splits,
)
from src.model_logreg import PARAM_GRID, make_model


class _RecordingLogReg(LogisticRegression):
    """Real LogisticRegression that records inputs and fitted weights without mutating params."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict | None = None,
        recorder: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            C=C,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.recorder = recorder

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.min() >= -1e-9
        assert X_arr.max() <= np.pi + 1e-9
        super().fit(X_arr, y, sample_weight=sample_weight)
        if self.recorder is not None:
            self.recorder.append(
                {
                    "C": self.C,
                    "class_weight": self.class_weight,
                    "n_rows": len(X_arr),
                    "coef": self.coef_.copy(),
                    "intercept": self.intercept_.copy(),
                }
            )
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.min() >= -1e-9
        assert X_arr.max() <= np.pi + 1e-9
        return super().predict(X_arr)


def test_get_splits_deterministic_and_stratified():
    X, y = load_pima()
    splits_a = get_splits(X, y)
    splits_b = get_splits(X, y)
    splits_default = get_splits()

    assert len(splits_a) == 25
    assert len(splits_default) == 25
    overall_pos_rate = float(np.mean(y))

    for (tr_a, te_a), (tr_b, te_b), (tr_d, te_d) in zip(
        splits_a, splits_b, splits_default
    ):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)
        np.testing.assert_array_equal(tr_a, tr_d)
        np.testing.assert_array_equal(te_a, te_d)

        assert len(np.intersect1d(tr_a, te_a)) == 0
        assert len(tr_a) + len(te_a) == len(y)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([tr_a, te_a])), np.arange(len(y))
        )
        assert abs(float(np.mean(y[te_a])) - overall_pos_rate) < 0.02


def test_every_patient_in_exactly_five_test_sets():
    X, y = load_pima()
    splits = get_splits(X, y)

    all_test_counts = np.bincount(
        np.concatenate([te for _, te in splits]), minlength=len(y)
    )
    assert np.all(all_test_counts == 5)

    first_five_counts = np.bincount(
        np.concatenate([te for _, te in splits[:5]]), minlength=len(y)
    )
    assert np.all(first_five_counts == 1)


def test_evaluate_zero_test_data_leakage(tmp_path):
    X, y = load_pima()
    splits = get_splits(X, y)
    _, test_idx_0 = splits[0]

    X_perturbed = X.copy()
    X_perturbed.iloc[test_idx_0, :] = 1e6

    clean_records: list[dict[str, Any]] = []
    perturbed_records: list[dict[str, Any]] = []

    evaluate(
        "leak_clean",
        lambda: _RecordingLogReg(recorder=clean_records),
        PARAM_GRID,
        fixed_params={"C": 1.0, "class_weight": "balanced"},
        X=X,
        y=y,
        n_splits_to_run=1,
        results_dir=tmp_path,
    )
    evaluate(
        "leak_perturbed",
        lambda: _RecordingLogReg(recorder=perturbed_records),
        PARAM_GRID,
        fixed_params={"C": 1.0, "class_weight": "balanced"},
        X=X_perturbed,
        y=y,
        n_splits_to_run=1,
        results_dir=tmp_path,
    )

    assert len(clean_records) == 1
    assert len(perturbed_records) == 1
    np.testing.assert_array_equal(clean_records[0]["coef"], perturbed_records[0]["coef"])
    np.testing.assert_array_equal(
        clean_records[0]["intercept"], perturbed_records[0]["intercept"]
    )


def test_evaluate_runs_logreg_saves_csv_and_quick_mode(tmp_path):
    from src.evaluate import DEFAULT_EVAL_SPLITS, format_summary

    summary = evaluate("logreg_full", make_model, PARAM_GRID, results_dir=tmp_path)
    csv_path = tmp_path / "logreg_full.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    expected_cols = [
        "split",
        "n_train",
        "n_test",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "train_time_sec",
        "best_params",
        "selected_features",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == DEFAULT_EVAL_SPLITS == 25
    assert list(df["split"]) == list(range(25))
    assert ((df["accuracy"] >= 0.0) & (df["accuracy"] <= 1.0)).all()
    assert ((df["balanced_accuracy"] >= 0.0) & (df["balanced_accuracy"] <= 1.0)).all()
    assert ((df["macro_f1"] >= 0.0) & (df["macro_f1"] <= 1.0)).all()
    assert (df["train_time_sec"] >= 0.0).all()

    assert summary["n_splits"] == 25
    assert summary["accuracy_mean"] == pytest.approx(float(df["accuracy"].mean()))
    assert summary["balanced_accuracy_mean"] == pytest.approx(
        float(df["balanced_accuracy"].mean())
    )
    assert summary["majority_floor"] == pytest.approx(500.0 / 768.0)

    rendered = format_summary(summary)
    assert "logreg_full" in rendered
    assert "65.10%" in rendered

    quick_calls: list[dict[str, Any]] = []
    quick_summary = evaluate(
        "logreg_quick",
        lambda: _RecordingLogReg(recorder=quick_calls),
        PARAM_GRID,
        quick=True,
        results_dir=tmp_path,
    )
    quick_csv = tmp_path / "logreg_quick.csv"
    assert quick_csv.exists()
    quick_df = pd.read_csv(quick_csv)
    assert len(quick_df) == 5
    assert quick_summary["quick"] is True
    distinct_combos = {(c["C"], c["class_weight"]) for c in quick_calls}
    assert len(distinct_combos) <= 2
    assert len(quick_calls) <= 35


FAST_FIXED = {"C": 1.0, "class_weight": "balanced"}


def test_split_count_constants():
    assert DEFAULT_EVAL_SPLITS == 25
    assert QUICK_EVAL_SPLITS == 5
    assert EXPERIMENT_EVAL_SPLITS == 5


def test_evaluate_runs_25_splits_by_default(tmp_path):
    summary = evaluate(
        "default25", make_model, PARAM_GRID, fixed_params=FAST_FIXED, results_dir=tmp_path
    )
    df = pd.read_csv(tmp_path / "default25.csv")
    assert summary["n_splits"] == 25
    assert list(df["split"]) == list(range(25))


def test_quick_mode_still_runs_exactly_5_splits(tmp_path):
    # Quick mode stays at 5 even with the default (25) or a larger explicit request.
    for name, kwargs in (("quick_default", {}), ("quick_ask25", {"n_splits_to_run": 25})):
        summary = evaluate(
            name, make_model, PARAM_GRID, quick=True, results_dir=tmp_path, **kwargs
        )
        assert summary["n_splits"] == 5
        assert list(pd.read_csv(tmp_path / f"{name}.csv")["split"]) == [0, 1, 2, 3, 4]


def test_experiment_split_count_runs_5_splits(tmp_path):
    summary = evaluate(
        "exp5",
        make_model,
        PARAM_GRID,
        fixed_params=FAST_FIXED,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=tmp_path,
    )
    assert summary["n_splits"] == 5


def test_first_5_of_25_identical_to_5_split_run(tmp_path):
    X, y = load_pima()
    all_25 = get_splits(X, y)
    assert len(all_25) == 25
    # The 5-split run uses exactly the first 5 index pairs of the 25 ...
    for (tr_a, te_a), (tr_b, te_b) in zip(all_25[:5], get_splits(X, y, n_repeats=1)):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)

    # ... so its per-split results equal the first 5 rows of the 25-split run.
    evaluate("run25", make_model, PARAM_GRID, results_dir=tmp_path)
    evaluate(
        "run5",
        make_model,
        PARAM_GRID,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=tmp_path,
    )
    df25 = pd.read_csv(tmp_path / "run25.csv").drop(columns="train_time_sec")
    df5 = pd.read_csv(tmp_path / "run5.csv").drop(columns="train_time_sec")
    pd.testing.assert_frame_equal(df25.iloc[:5].reset_index(drop=True), df5)


def test_evaluate_rejects_oversized_param_grid(tmp_path):
    bad_grid = {"C": [0.01, 0.1, 1.0], "max_iter": [100, 500, 1000]}
    with pytest.raises(ValueError, match="8"):
        evaluate("too_many", make_model, bad_grid, results_dir=tmp_path)


def test_evaluate_n_features_and_make_moons(tmp_path):
    # Experiments 1 (qubits) and 5 (moons) use the first 5 splits, passed explicitly.
    summary_k4 = evaluate(
        "logreg_k4",
        make_model,
        PARAM_GRID,
        n_features=4,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=tmp_path,
    )
    df_k4 = summary_k4["results"]
    assert len(df_k4) == 5
    assert all(len(str(row).split(",")) == 4 for row in df_k4["selected_features"])

    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    summary_moons = evaluate(
        "logreg_moons",
        make_model,
        PARAM_GRID,
        n_features=2,
        X=X_moons,
        y=y_moons,
        zero_as_missing_cols=(),
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=tmp_path,
    )
    moons_csv = tmp_path / "logreg_moons.csv"
    assert moons_csv.exists()
    assert len(summary_moons["results"]) == 5
    assert summary_moons["majority_floor"] == pytest.approx(0.5)


def test_evaluate_train_size_fixed_params_and_tune_on_pool(tmp_path):
    from src.evaluate import get_fixed_test_split, tune_hyperparameters

    X, y = load_pima()
    pool_idx, test_idx = get_fixed_test_split(X, y)
    pool_idx_2, test_idx_2 = get_fixed_test_split(X, y)

    np.testing.assert_array_equal(pool_idx, pool_idx_2)
    np.testing.assert_array_equal(test_idx, test_idx_2)
    assert len(pool_idx) == 614
    assert len(test_idx) == 154
    assert len(np.intersect1d(pool_idx, test_idx)) == 0

    best_params = tune_hyperparameters(
        make_model, PARAM_GRID, X.iloc[pool_idx], y[pool_idx]
    )
    assert set(best_params) == {"C", "class_weight"}
    assert best_params["C"] in PARAM_GRID["C"]
    assert best_params["class_weight"] in PARAM_GRID["class_weight"]

    fit_calls: list[dict[str, Any]] = []
    fixed = {"C": 1.0, "class_weight": "balanced"}
    summary = evaluate(
        "logreg_sz30",
        lambda: _RecordingLogReg(recorder=fit_calls),
        PARAM_GRID,
        train_size=30,
        fixed_params=fixed,
        n_draws=4,
        results_dir=tmp_path,
    )
    df = pd.read_csv(tmp_path / "logreg_sz30.csv")
    assert len(df) == 4
    assert summary["n_splits"] == 4
    assert (df["n_train"] == 30).all()
    assert (df["n_test"] == 154).all()
    assert len(fit_calls) == 4
    assert all(json.loads(val) == fixed for val in df["best_params"])


def test_compare_models_nadeau_bengio_and_edge_cases(tmp_path):
    scores_a = np.array([0.75, 0.78, 0.72, 0.76, 0.74])
    scores_b = np.array([0.70, 0.72, 0.70, 0.73, 0.70])
    df_a = pd.DataFrame(
        {
            "split": [0, 1, 2, 3, 4],
            "n_train": [614, 614, 614, 615, 615],
            "n_test": [154, 154, 154, 153, 153],
            "balanced_accuracy": scores_a,
        }
    )
    df_b = pd.DataFrame(
        {
            "split": [0, 1, 2, 3, 4],
            "n_train": [614, 614, 614, 615, 615],
            "n_test": [154, 154, 154, 153, 153],
            "balanced_accuracy": scores_b,
        }
    )

    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)

    res_df = compare_models(df_a, df_b, metric="balanced_accuracy")
    res_csv = compare_models(path_a, str(path_b), metric="balanced_accuracy")

    d = scores_a - scores_b
    j = 5
    n_train_mean = float(df_a["n_train"].mean())
    n_test_mean = float(df_a["n_test"].mean())
    expected_se = float(np.sqrt((1.0 / j + n_test_mean / n_train_mean) * np.var(d, ddof=1)))
    expected_t = float(np.mean(d) / expected_se)
    expected_p = float(2.0 * stats.t.sf(abs(expected_t), df=j - 1))

    for res in (res_df, res_csv):
        assert res["df"] == 4
        assert res["n_splits"] == 5
        assert res["mean_diff"] == pytest.approx(float(np.mean(d)), abs=1e-12)
        assert res["corrected_se"] == pytest.approx(expected_se, abs=1e-12)
        assert res["t_stat"] == pytest.approx(expected_t, abs=1e-12)
        assert res["p_value"] == pytest.approx(expected_p, abs=1e-12)
        assert res["significant"] is bool(expected_p < 0.05)

    # Small noisy difference -> not significant, never labelled "equal"
    df_noisy_a = df_a.copy()
    df_noisy_b = df_a.copy()
    df_noisy_b["balanced_accuracy"] = scores_a + np.array([0.02, -0.02, 0.01, -0.01, 0.005])
    res_ns = compare_models(df_noisy_a, df_noisy_b, metric="balanced_accuracy")
    assert res_ns["significant"] is False
    assert res_ns["conclusion"] == "no significant difference found"
    assert "equal" not in res_ns["conclusion"].lower()

    # Identical scores -> zero difference, p = 1.0
    res_same = compare_models(df_a, df_a, metric="balanced_accuracy")
    assert res_same["mean_diff"] == 0.0
    assert res_same["t_stat"] == 0.0
    assert res_same["p_value"] == 1.0
    assert res_same["significant"] is False
    assert res_same["conclusion"] == "no significant difference found"

    # Mismatched split count or IDs -> ValueError
    with pytest.raises(ValueError):
        compare_models(df_a, df_b.iloc[:4])
    df_b_reordered = df_b.copy()
    df_b_reordered["split"] = [0, 1, 2, 3, 99]
    with pytest.raises(ValueError):
        compare_models(df_a, df_b_reordered)

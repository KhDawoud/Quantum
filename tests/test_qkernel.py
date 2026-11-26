"""Comprehensive tests for Quantum Kernel SVM and Exp 3 (kernel encoding)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import src.experiments as experiments
from src.data import PimaPreprocessor, load_pima
from src.model_qkernel import (
    MAX_QUBITS,
    PARAM_GRID,
    QuantumKernelSVC,
    build_state_qnode,
    classical_angle_kernel,
    compute_kernel_matrix,
    compute_states,
    draw_feature_map,
    hand_zz_2qubit_kernel,
    kernel_stats,
    make_model,
    run_sanity_checks,
)


def test_param_grid_contract():
    n_combos = 1
    for vals in PARAM_GRID.values():
        n_combos *= len(vals)
    assert n_combos <= 8
    assert "C" in PARAM_GRID and "gamma" in PARAM_GRID
    assert all(c > 0 for c in PARAM_GRID["C"])
    assert all(0.0 < g <= 1.0 for g in PARAM_GRID["gamma"])


def test_zz_circuit_structure_3_qubits():
    qnode = build_state_qnode(n_qubits=3, encoding="zz", reps=2)
    tape = qml.workflow.construct_tape(qnode)(np.array([0.4, 0.8, 1.2]))
    ops = tape.operations

    # Per rep: 3 H + 3 RZ + 2*(CNOT + RZ + CNOT) = 12 gates -> 24 total for 2 reps.
    assert len(ops) == 24
    op_names = [op.name for op in ops]
    assert op_names.count("Hadamard") == 6
    assert op_names.count("RZ") == 10
    assert op_names.count("CNOT") == 8

    # Confirm strictly linear pairs (0, 1) and (1, 2) with NO (2, 0) wrap-around.
    cnot_wires = [tuple(op.wires.tolist()) for op in ops if op.name == "CNOT"]
    assert set(cnot_wires) == {(0, 1), (1, 2)}

    diagram = draw_feature_map(n_qubits=3, encoding="zz", reps=2)
    assert "H" in diagram and "RZ" in diagram


def test_angle_circuit_structure_3_qubits():
    qnode = build_state_qnode(n_qubits=3, encoding="angle", reps=2)
    tape = qml.workflow.construct_tape(qnode)(np.array([0.4, 0.8, 1.2]))
    ops = tape.operations
    assert [op.name for op in ops] == ["RY", "RY", "RY"]
    assert [op.wires.tolist() for op in ops] == [[0], [1], [2]]

    diagram = draw_feature_map(n_qubits=3, encoding="angle")
    assert "RY" in diagram


@pytest.mark.parametrize("reps", [1, 2])
@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_zz_2qubit_matches_hand_calculation(reps: int, gamma: float):
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, np.pi, size=(8, 2))
    Y = rng.uniform(0.0, np.pi, size=(5, 2))

    k_qml_train = compute_kernel_matrix(X, encoding="zz", reps=reps, gamma=gamma, polish=False)
    k_hand_train = hand_zz_2qubit_kernel(X, reps=reps, gamma=gamma)
    np.testing.assert_allclose(k_qml_train, k_hand_train, atol=1e-12, rtol=0.0)

    k_qml_cross = compute_kernel_matrix(X, Y, encoding="zz", reps=reps, gamma=gamma, polish=False)
    k_hand_cross = hand_zz_2qubit_kernel(X, Y, reps=reps, gamma=gamma)
    np.testing.assert_allclose(k_qml_cross, k_hand_cross, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_qubits", [2, 8])
@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_angle_kernel_matches_classical_formula(n_qubits: int, gamma: float):
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, np.pi, size=(10, n_qubits))
    Y = rng.uniform(0.0, np.pi, size=(4, n_qubits))

    k_qml = compute_kernel_matrix(X, Y, encoding="angle", gamma=gamma, polish=False)
    k_cls = classical_angle_kernel(X, Y, gamma=gamma)
    np.testing.assert_allclose(k_qml, k_cls, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("encoding", ["zz", "angle"])
def test_raw_kernel_matrix_properties_8_qubits(encoding: str):
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, np.pi, size=(15, 8))
    K = compute_kernel_matrix(X, encoding=encoding, gamma=1.0, polish=False)

    assert K.shape == (15, 15)
    assert np.max(np.abs(K - K.T)) < 1e-12
    assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-12
    assert np.all(K >= -1e-12) and np.all(K <= 1.0 + 1e-12)
    eigs = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert np.min(eigs) >= -1e-10


def test_kernel_stats_off_diagonal_only():
    K = np.array(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.6],
            [0.4, 0.6, 1.0],
        ]
    )
    stats = kernel_stats(K)
    off_diag = np.array([0.2, 0.4, 0.2, 0.6, 0.4, 0.6])
    assert stats["mean"] == pytest.approx(float(np.mean(off_diag)))
    assert stats["var"] == pytest.approx(float(np.var(off_diag)))
    assert stats["std"] == pytest.approx(float(np.std(off_diag)))
    assert stats["min"] == pytest.approx(0.2)
    assert stats["max"] == pytest.approx(0.6)

    with pytest.raises(ValueError):
        kernel_stats(np.array([[1.0]]))
    with pytest.raises(ValueError):
        kernel_stats(np.ones((2, 3)))


def test_validation_guards():
    X = np.ones((5, 4), dtype=np.float64)
    y = np.array([0, 1, 0, 1, 0])

    with pytest.raises(ValueError, match="gamma"):
        make_model(gamma=1.5).fit(X, y)
    with pytest.raises(ValueError, match="gamma"):
        make_model(gamma=0.0).fit(X, y)
    with pytest.raises(ValueError, match="encoding"):
        make_model(encoding="iqp").fit(X, y)
    with pytest.raises(ValueError, match="reps"):
        make_model(reps=0).fit(X, y)
    with pytest.raises(ValueError, match="C must be"):
        make_model(C=-1.0).fit(X, y)

    X_too_wide = np.ones((4, MAX_QUBITS + 1), dtype=np.float64)
    with pytest.raises(ValueError, match="n_qubits"):
        compute_states(X_too_wide)


def test_sklearn_contract_pipeline_gridsearch_and_no_test_leakage():
    X, y = load_pima()
    X_tr, y_tr = X.iloc[:80], y[:80]
    X_te = X.iloc[80:95]

    pipe = Pipeline(
        [
            ("prep", PimaPreprocessor(n_features=4)),
            ("clf", make_model(encoding="zz")),
        ]
    )
    cloned = clone(pipe)
    cloned.fit(X_tr, y_tr)

    preds_batch = cloned.predict(X_te)
    scores_batch = cloned.decision_function(X_te)
    assert preds_batch.shape == (15,)
    assert scores_batch.shape == (15,)

    # Predicting a single row must match the batch prediction exactly (no batch/test leakage).
    pred_single = cloned.predict(X_te.iloc[[0]])
    assert pred_single.shape == (1,)
    assert pred_single[0] == preds_batch[0]

    # Test-time prediction must never mutate stored training states or training kernel.
    clf_step: QuantumKernelSVC = cloned.named_steps["clf"]
    train_states_before = clf_step.train_states_.copy()
    train_kernel_before = clf_step.train_kernel_.copy()
    _ = clf_step.predict(np.zeros((3, 4)))
    np.testing.assert_array_equal(clf_step.train_states_, train_states_before)
    np.testing.assert_array_equal(clf_step.train_kernel_, train_kernel_before)

    # GridSearchCV compatibility on synthetic [0, pi] data
    X_syn, y_syn = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=1, random_state=42
    )
    X_syn = PimaPreprocessor(zero_as_missing_cols=()).fit_transform(X_syn, y_syn)
    gs = GridSearchCV(
        make_model(encoding="zz"),
        {"C": [0.1, 1.0], "gamma": [0.5, 1.0]},
        cv=3,
        scoring="balanced_accuracy",
    )
    gs.fit(X_syn, y_syn)
    assert gs.best_params_["gamma"] in (0.5, 1.0)
    assert gs.predict(X_syn[:5]).shape == (5,)


def test_run_sanity_checks_passes():
    report = run_sanity_checks(verbose=False)
    assert report["hand_zz_2q_err"] < 1e-12
    assert report["classical_angle_8q_err"] < 1e-12
    assert report["zz_min_eig"] >= -1e-10


def test_run_exp3_encoding_kernel_with_mocked_evaluate(tmp_path, monkeypatch):
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    res_dir.mkdir()
    fig_dir.mkdir()

    def fake_evaluate(name, make_model_fn, param_grid, n_features=8, train_size=None, quick=False, fixed_params=None):
        assert n_features == 8
        assert quick is True
        est = make_model_fn()
        assert isinstance(est, QuantumKernelSVC)
        assert param_grid == PARAM_GRID

        offset = 0.02 if name == "qkernel_zz" else 0.0
        df = pd.DataFrame(
            {
                "split": range(5),
                "accuracy": [0.73 + offset] * 5,
                "balanced_accuracy": [0.70 + offset] * 5,
                "macro_f1": [0.69 + offset] * 5,
                "train_time": [0.15 + offset] * 5,
            }
        )
        df.to_csv(res_dir / f"{name}.csv", index=False)
        return {
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": 0.01,
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": 0.01,
            "macro_f1_mean": float(df["macro_f1"].mean()),
            "macro_f1_std": 0.01,
            "train_time_mean": float(df["train_time"].mean()),
            "train_time_std": 0.002,
            "n_splits": 5,
        }

    def fake_compare_models(path_a, path_b):
        assert Path(path_a).exists() and Path(path_b).exists()
        return {"mean_diff": -0.02, "p_value": 0.31}

    monkeypatch.setattr(experiments, "evaluate", fake_evaluate)
    monkeypatch.setattr(experiments, "compare_models", fake_compare_models)

    out_df = experiments.run_exp3_encoding_kernel(
        quick=True,
        results_dir=res_dir,
        figures_dir=fig_dir,
    )

    assert list(out_df["encoding"]) == ["angle", "zz"]
    assert list(out_df["classically_simulable"]) == [True, False]
    assert out_df.loc[out_df["encoding"] == "angle", "classical_formula_max_diff"].iloc[0] < 1e-12
    assert out_df["paired_p_value"].iloc[0] == pytest.approx(0.31)
    assert (res_dir / "exp3_encoding_kernel.csv").exists()
    fig_file = fig_dir / "exp3_encoding_kernel.png"
    assert fig_file.exists() and fig_file.stat().st_size > 1024

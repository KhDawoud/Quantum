"""Quantum Kernel SVM classifier (PennyLane default.qubit + precomputed SVC).

Adapted from PennyLane quantum kernel and IBM Quantum Learning QML patterns:
  - PennyLane ``default.qubit``, ``qml.state``, ``qml.RY``:
    https://docs.pennylane.ai/
  - IBM Quantum Learning, *Introduction to Quantum Machine Learning*
    (ZZ feature map and fidelity quantum kernels):
    https://learning.quantum.ibm.com/course/quantum-machine-learning

Contract:
  - ``make_model(encoding='zz', **kwargs)`` returning an estimator with fit/predict.
  - ``PARAM_GRID`` with <= 8 combinations and ``gamma <= 1.0`` (e.g. {0.5, 1.0}).
  - ``kernel_stats(K)`` returning off-diagonal mean and variance for Experiment 1/3.
"""

from __future__ import annotations

from functools import lru_cache
import time
from typing import Any

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

RANDOM_SEED = 42
MAX_QUBITS = 10
VALID_ENCODINGS: tuple[str, ...] = ("zz", "angle")

# At most 8 combinations; gamma must stay <= 1.0 so scaled [0, pi] inputs never wrap past pi.
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": [0.5, 1.0],
}


def _validate_circuit_config(
    n_qubits: int,
    encoding: str,
    reps: int,
    gamma: float,
) -> None:
    """Validate qubit limit, encoding choice, repetition count, and angle scaling."""
    if (
        not isinstance(n_qubits, (int, np.integer))
        or isinstance(n_qubits, bool)
        or n_qubits < 1
        or n_qubits > MAX_QUBITS
    ):
        raise ValueError(
            f"n_qubits must be an integer in [1, {MAX_QUBITS}], got {n_qubits}."
        )
    if encoding not in VALID_ENCODINGS:
        raise ValueError(
            f"encoding must be one of {VALID_ENCODINGS}, got {encoding!r}."
        )
    if not isinstance(reps, (int, np.integer)) or isinstance(reps, bool) or reps < 1:
        raise ValueError(f"reps must be a positive integer, got {reps}.")
    gamma_val = float(gamma)
    if not np.isfinite(gamma_val) or gamma_val <= 0.0 or gamma_val > 1.0:
        raise ValueError(
            f"gamma must be in (0, 1.0] so angles stay within [0, pi], got {gamma}."
        )


def _apply_feature_map(
    x: np.ndarray,
    n_qubits: int,
    encoding: str,
    reps: int,
) -> None:
    """Apply either the entangled ZZ-style feature map or simple RY angle encoding."""
    if encoding == "angle":
        for i in range(n_qubits):
            qml.RY(x[..., i], wires=i)
        return

    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[..., i], wires=i)
        # Linear nearest-neighbour pairs only (no wrap-around from n_qubits - 1 to 0).
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(x[..., i] * x[..., i + 1], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


@lru_cache(maxsize=16)
def build_state_qnode(
    n_qubits: int,
    encoding: str = "zz",
    reps: int = 2,
):
    """Build a PennyLane ``default.qubit`` QNode returning the encoded quantum statevector."""
    _validate_circuit_config(n_qubits, encoding, reps, gamma=1.0)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_circuit(x: np.ndarray):
        _apply_feature_map(x, n_qubits=n_qubits, encoding=encoding, reps=reps)
        return qml.state()

    return state_circuit


def compute_states(
    X: np.ndarray,
    encoding: str = "zz",
    reps: int = 2,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute statevectors for every row in ``X`` in a single batched simulation pass."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] < 1:
        raise ValueError(f"Expected non-empty 2D array X, got shape {X_arr.shape}.")
    if not np.isfinite(X_arr).all():
        raise ValueError("Input X contains NaN or infinite values.")

    n_samples, n_qubits = X_arr.shape
    _validate_circuit_config(n_qubits, encoding, reps, gamma)

    X_scaled = X_arr * float(gamma)
    qnode = build_state_qnode(n_qubits=n_qubits, encoding=encoding, reps=reps)
    states = np.asarray(qnode(X_scaled), dtype=np.complex128)
    if states.ndim == 1:
        states = states[np.newaxis, :]

    expected_shape = (n_samples, 2**n_qubits)
    if states.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected statevector batch shape {states.shape}; expected {expected_shape}."
        )
    return states


def fidelity_kernel_from_states(
    states_a: np.ndarray,
    states_b: np.ndarray | None = None,
    *,
    polish: bool = True,
) -> np.ndarray:
    """Compute exact fidelity kernel ``K[i, j] = |<psi_a[i] | psi_b[j]>|^2``."""
    a = np.asarray(states_a, dtype=np.complex128)
    b = a if states_b is None else np.asarray(states_b, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Incompatible statevector shapes {a.shape} and {b.shape} for overlap."
        )

    overlaps = a @ b.conj().T
    kernel = np.asarray(overlaps.real**2 + overlaps.imag**2, dtype=np.float64)

    if polish:
        if states_b is None:
            kernel = 0.5 * (kernel + kernel.T)
            kernel = np.clip(kernel, 0.0, 1.0)
            np.fill_diagonal(kernel, 1.0)
        else:
            kernel = np.clip(kernel, 0.0, 1.0)
    return kernel


def compute_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    encoding: str = "zz",
    reps: int = 2,
    gamma: float = 1.0,
    polish: bool = True,
) -> np.ndarray:
    """Compute the quantum fidelity kernel matrix between ``X`` and ``Y`` (or ``X``)."""
    states_x = compute_states(X, encoding=encoding, reps=reps, gamma=gamma)
    states_y = (
        None
        if Y is None
        else compute_states(Y, encoding=encoding, reps=reps, gamma=gamma)
    )
    return fidelity_kernel_from_states(states_x, states_y, polish=polish)


def classical_angle_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """Exact closed-form product-state kernel for unentangled RY angle encoding.

    For ``|psi(x)> = tensor_i RY(gamma * x_i) |0>``, the overlap squared is
    ``prod_i cos^2(gamma * (x_i - y_i) / 2)``, which is classically efficient.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)
    if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(
            f"Incompatible shapes {X_arr.shape} and {Y_arr.shape} for classical angle kernel."
        )
    _validate_circuit_config(X_arr.shape[1], "angle", reps=1, gamma=gamma)
    diffs = (X_arr[:, np.newaxis, :] - Y_arr[np.newaxis, :, :]) * (0.5 * float(gamma))
    return np.asarray(np.prod(np.cos(diffs) ** 2, axis=2), dtype=np.float64)


def hand_zz_2qubit_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    reps: int = 2,
    gamma: float = 1.0,
) -> np.ndarray:
    """Independent 2-qubit linear-algebra reference for the ZZ feature map."""
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)
    if X_arr.ndim != 2 or Y_arr.ndim != 2 or X_arr.shape[1] != 2 or Y_arr.shape[1] != 2:
        raise ValueError("hand_zz_2qubit_kernel requires 2D arrays with exactly 2 columns.")
    _validate_circuit_config(2, "zz", reps=reps, gamma=gamma)

    h1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    h2 = np.kron(h1, h1)
    # Pauli-Z eigenvalues (s0, s1) for computational basis |00>, |01>, |10>, |11>.
    s0 = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    s1 = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    s01 = s0 * s1

    def _encode_batch(arr: np.ndarray) -> np.ndarray:
        out = np.empty((arr.shape[0], 4), dtype=np.complex128)
        for idx, row in enumerate(arr):
            z0, z1 = float(gamma) * row[0], float(gamma) * row[1]
            phases = np.exp(-0.5j * (z0 * s0 + z1 * s1 + (z0 * z1) * s01))
            psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
            for _ in range(reps):
                psi = phases * (h2 @ psi)
            out[idx] = psi
        return out

    states_x = _encode_batch(X_arr)
    states_y = states_x if Y is None else _encode_batch(Y_arr)
    return fidelity_kernel_from_states(states_x, states_y, polish=False)


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Scikit-learn SVM classifier using a precomputed quantum fidelity kernel."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 1.0,
        encoding: str = "zz",
        reps: int = 2,
        class_weight: str | dict[Any, float] | None = None,
        random_state: int | None = RANDOM_SEED,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.encoding = encoding
        self.reps = reps
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVC":
        """Compute training quantum states, build Gram matrix, and fit precomputed SVC."""
        X_arr, y_arr = check_X_y(X, y, dtype=np.float64)
        c_val = float(self.C)
        if not np.isfinite(c_val) or c_val <= 0.0:
            raise ValueError(f"C must be finite and > 0, got {self.C}.")

        _validate_circuit_config(
            n_qubits=X_arr.shape[1],
            encoding=self.encoding,
            reps=self.reps,
            gamma=self.gamma,
        )

        self.n_features_in_ = int(X_arr.shape[1])
        self.train_states_ = compute_states(
            X_arr,
            encoding=self.encoding,
            reps=self.reps,
            gamma=float(self.gamma),
        )
        self.train_kernel_ = fidelity_kernel_from_states(
            self.train_states_, polish=True
        )
        self.svc_ = SVC(
            kernel="precomputed",
            C=c_val,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(self.train_kernel_, y_arr)
        self.classes_ = self.svc_.classes_
        return self

    def _test_kernel(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["train_states_", "svc_", "classes_", "n_features_in_"])
        X_arr = check_array(X, dtype=np.float64)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}."
            )
        test_states = compute_states(
            X_arr,
            encoding=self.encoding,
            reps=self.reps,
            gamma=float(self.gamma),
        )
        return fidelity_kernel_from_states(test_states, self.train_states_, polish=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed SVM decision scores against training support vectors."""
        return self.svc_.decision_function(self._test_kernel(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for rows in ``X`` using stored training states."""
        return self.svc_.predict(self._test_kernel(X))

    def compute_kernel(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        """Convenience method to compute the kernel matrix with this estimator's config."""
        return compute_kernel_matrix(
            X,
            Y,
            encoding=self.encoding,
            reps=self.reps,
            gamma=float(self.gamma),
            polish=True,
        )


def make_model(encoding: str = "zz", **kwargs: Any) -> QuantumKernelSVC:
    """Return a scikit-learn compatible Quantum Kernel SVM estimator."""
    return QuantumKernelSVC(encoding=encoding, **kwargs)


def kernel_stats(K: np.ndarray) -> dict[str, float]:
    """Return summary statistics of off-diagonal entries of a square kernel matrix ``K``."""
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1] or K_arr.shape[0] < 2:
        raise ValueError(
            f"kernel_stats expects a square 2D matrix with size >= 2, got shape {K_arr.shape}."
        )
    if not np.isfinite(K_arr).all():
        raise ValueError("Kernel matrix K contains NaN or infinite values.")

    mask = ~np.eye(K_arr.shape[0], dtype=bool)
    off_diag = K_arr[mask]
    return {
        "mean": float(np.mean(off_diag)),
        "var": float(np.var(off_diag)),
        "std": float(np.std(off_diag)),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
    }


def draw_feature_map(
    n_qubits: int = 3,
    encoding: str = "zz",
    reps: int = 2,
    gamma: float = 1.0,
    sample_input: np.ndarray | None = None,
) -> str:
    """Render an ASCII diagram of the quantum feature map using ``qml.draw``."""
    _validate_circuit_config(n_qubits, encoding, reps, gamma)
    if sample_input is None:
        x = np.linspace(0.4, 1.6, n_qubits, dtype=np.float64) * float(gamma)
    else:
        x = np.asarray(sample_input, dtype=np.float64).ravel() * float(gamma)
        if x.shape != (n_qubits,):
            raise ValueError(f"Expected sample_input of length {n_qubits}, got {x.shape}.")
    qnode = build_state_qnode(n_qubits=n_qubits, encoding=encoding, reps=reps)
    return str(qml.draw(qnode)(x))


def run_sanity_checks(verbose: bool = True) -> dict[str, Any]:
    """Run mathematical sanity checks on unpolished kernel matrices and hand formulas."""
    rng = np.random.default_rng(RANDOM_SEED)

    X_8q = rng.uniform(0.0, np.pi, size=(12, 8))
    K_zz_raw = compute_kernel_matrix(X_8q, encoding="zz", reps=2, gamma=1.0, polish=False)
    K_angle_raw = compute_kernel_matrix(X_8q, encoding="angle", gamma=1.0, polish=False)

    zz_sym_err = float(np.max(np.abs(K_zz_raw - K_zz_raw.T)))
    zz_diag_err = float(np.max(np.abs(np.diag(K_zz_raw) - 1.0)))
    zz_min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (K_zz_raw + K_zz_raw.T))))

    angle_sym_err = float(np.max(np.abs(K_angle_raw - K_angle_raw.T)))
    angle_diag_err = float(np.max(np.abs(np.diag(K_angle_raw) - 1.0)))
    angle_min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (K_angle_raw + K_angle_raw.T))))

    X_2q = rng.uniform(0.0, np.pi, size=(6, 2))
    K_zz_2q_qml = compute_kernel_matrix(X_2q, encoding="zz", reps=2, gamma=0.5, polish=False)
    K_zz_2q_hand = hand_zz_2qubit_kernel(X_2q, reps=2, gamma=0.5)
    hand_zz_err = float(np.max(np.abs(K_zz_2q_qml - K_zz_2q_hand)))

    K_angle_8q_cls = classical_angle_kernel(X_8q, gamma=1.0)
    classical_angle_err = float(np.max(np.abs(K_angle_raw - K_angle_8q_cls)))

    assert zz_sym_err < 1e-12 and angle_sym_err < 1e-12
    assert zz_diag_err < 1e-12 and angle_diag_err < 1e-12
    assert zz_min_eig >= -1e-10 and angle_min_eig >= -1e-10
    assert hand_zz_err < 1e-12
    assert classical_angle_err < 1e-12

    report = {
        "zz_sym_err": zz_sym_err,
        "zz_diag_err": zz_diag_err,
        "zz_min_eig": zz_min_eig,
        "angle_sym_err": angle_sym_err,
        "angle_diag_err": angle_diag_err,
        "angle_min_eig": angle_min_eig,
        "hand_zz_2q_err": hand_zz_err,
        "classical_angle_8q_err": classical_angle_err,
    }
    if verbose:
        print("=== Quantum Kernel Sanity Checks ===")
        for key, value in report.items():
            print(f"  {key:<24s}: {value:.3e}")
    return report


def estimate_evaluate_runtime(
    n_samples: int = 768,
    n_qubits: int = 8,
    n_splits: int = 25,
    n_inner_folds: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
    encoding: str = "zz",
    reps: int = 2,
) -> dict[str, float]:
    """Benchmark 8-qubit QuantumKernelSVC and project full ``evaluate()`` runtime."""
    grid = PARAM_GRID if param_grid is None else param_grid
    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)

    n_outer_train = int(round(0.8 * n_samples))
    n_outer_test = n_samples - n_outer_train
    n_inner_train = int(round(n_outer_train * (n_inner_folds - 1) / n_inner_folds))
    n_inner_val = n_outer_train - n_inner_train

    rng = np.random.default_rng(RANDOM_SEED)
    X_outer_tr = rng.uniform(0.0, np.pi, size=(n_outer_train, n_qubits))
    y_outer_tr = rng.integers(0, 2, size=n_outer_train)
    X_outer_te = rng.uniform(0.0, np.pi, size=(n_outer_test, n_qubits))

    X_inner_tr = X_outer_tr[:n_inner_train]
    y_inner_tr = y_outer_tr[:n_inner_train]
    X_inner_va = X_outer_tr[n_inner_train : n_inner_train + n_inner_val]

    # Warm up QNode compilation once, then time inner-fold and outer-refit cycles.
    _ = make_model(encoding=encoding, reps=reps, gamma=1.0).fit(
        X_inner_tr[:16], y_inner_tr[:16]
    )

    t0 = time.perf_counter()
    inner_clf = make_model(encoding=encoding, reps=reps, C=1.0, gamma=1.0).fit(
        X_inner_tr, y_inner_tr
    )
    _ = inner_clf.predict(X_inner_va)
    inner_fit_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    outer_clf = make_model(encoding=encoding, reps=reps, C=1.0, gamma=1.0).fit(
        X_outer_tr, y_outer_tr
    )
    _ = outer_clf.predict(X_outer_te)
    outer_fit_sec = time.perf_counter() - t1

    total_fits = n_splits * (n_combos * n_inner_folds + 1)
    est_seconds = n_splits * (n_combos * n_inner_folds * inner_fit_sec + outer_fit_sec)
    return {
        "n_qubits": float(n_qubits),
        "n_splits": float(n_splits),
        "n_combos": float(n_combos),
        "total_fits": float(total_fits),
        "inner_fit_sec": float(inner_fit_sec),
        "outer_fit_sec": float(outer_fit_sec),
        "estimated_seconds": float(est_seconds),
        "estimated_minutes": float(est_seconds / 60.0),
        "exceeds_30_min": bool(est_seconds > 1800.0),
    }


if __name__ == "__main__":
    print("=== 3-Qubit ZZ Feature Map ===")
    print(draw_feature_map(n_qubits=3, encoding="zz", reps=2, gamma=1.0))
    print("\n=== 3-Qubit Angle Encoding ===")
    print(draw_feature_map(n_qubits=3, encoding="angle", gamma=1.0))
    print()

    run_sanity_checks(verbose=True)

    rng = np.random.default_rng(RANDOM_SEED)
    print("\n=== Kernel Spread vs. Qubits (Synthetic [0, pi] data, n=120) ===")
    for k in (4, 6, 8):
        X_syn = rng.uniform(0.0, np.pi, size=(120, k))
        stats_zz = kernel_stats(compute_kernel_matrix(X_syn, encoding="zz", gamma=1.0))
        print(
            f"  ZZ ({k} qubits): mean={stats_zz['mean']:.4f}, "
            f"var={stats_zz['var']:.4e}, std={stats_zz['std']:.4f}"
        )
    X_8 = rng.uniform(0.0, np.pi, size=(120, 8))
    stats_ang = kernel_stats(compute_kernel_matrix(X_8, encoding="angle", gamma=1.0))
    print(
        f"  Angle (8 qubits): mean={stats_ang['mean']:.4f}, "
        f"var={stats_ang['var']:.4e}, std={stats_ang['std']:.4f}"
    )

    X_cls, y_cls = make_classification(
        n_samples=120, n_features=8, n_informative=5, random_state=RANDOM_SEED
    )
    X_cls = (X_cls - X_cls.min(axis=0)) / (X_cls.max(axis=0) - X_cls.min(axis=0)) * np.pi
    gs = GridSearchCV(
        make_model(encoding="zz"),
        PARAM_GRID,
        cv=3,
        scoring="balanced_accuracy",
    ).fit(X_cls, y_cls)
    print(
        f"\n=== GridSearchCV Smoke Test (n=120, 8q ZZ) ===\n"
        f"  Best params: {gs.best_params_} | Best 3-fold balanced_acc: {gs.best_score_:.4f}"
    )

    rt = estimate_evaluate_runtime()
    print(
        f"\n=== 25-Split Full evaluate() Runtime Estimate (8 qubits ZZ) ===\n"
        f"  inner_fit={rt['inner_fit_sec']*1000:.1f}ms, outer_fit={rt['outer_fit_sec']*1000:.1f}ms, "
        f"total_fits={int(rt['total_fits'])}, estimated={rt['estimated_seconds']:.2f}s "
        f"({rt['estimated_minutes']:.2f} min), exceeds_30_min={rt['exceeds_30_min']}"
    )

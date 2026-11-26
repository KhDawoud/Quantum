"""Variational Quantum Classifier (PennyLane default.qubit + StronglyEntanglingLayers).

Contract (PROJECT_CONTEXT.md):
  - ``make_model(encoding='zz', n_layers=2, **kwargs)`` returns an estimator with
    ``fit(X, y)``, ``predict(X)``, ``predict_proba(X)`` and ``loss_history_``.
  - ``PARAM_GRID`` has <= 4 combinations (VQC training is the runtime bottleneck).
  - Inputs must already be scaled to [0, pi] by ``PimaPreprocessor`` (hard rule).

Plain-language picture of the model
  1. ENCODE  : write one patient's 8 numbers into 8 qubits (angle or ZZ-style map).
  2. TRAIN   : a stack of trainable rotations + entangling gates ("the dials").
  3. MEASURE : read qubit 0 -> a number z in [-1, 1] -> turn it into a probability.
  Training = nudging the dials (Adam) so the probabilities match the labels.

Sources adapted (also cited in notes/DESIGN_D.md):
  - PennyLane "Variational classifier" demo (weights as a (layers, wires, 3) array,
    cost function = loss over a batch, optimiser.step): https://pennylane.ai/qml/demos/tutorial_variational_classifier
  - PennyLane docs for qml.StronglyEntanglingLayers (weights shape (L, M, 3)),
    qml.qnode (diff_method='backprop', parameter broadcasting) and qml.AdamOptimizer,
    all checked against the installed version 0.45.1.
  - Feature map structure: PROJECT_CONTEXT.md "ZZ-style feature map" (the same one the
    Quantum Kernel SVM uses), after Havlicek et al., Nature 567, 209 (2019).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

RANDOM_SEED = 42
ZZ_REPS = 2  # repetitions of the ZZ feature-map block (PROJECT_CONTEXT.md)
EPS = 1e-7  # keeps log() finite when a probability hits exactly 0 or 1
ANGLE_TOL = 1e-9

# Small grid (<= 4 combinations) because VQC training is the runtime bottleneck.
PARAM_GRID: dict[str, list[Any]] = {
    "lr": [0.02, 0.05],
    "epochs": [20, 40],
}


# --------------------------------------------------------------------------- #
# Circuit pieces
# --------------------------------------------------------------------------- #
def zz_feature_map(x, n_qubits: int, reps: int = ZZ_REPS) -> None:
    """ZZ-style feature map (PROJECT_CONTEXT.md), ``x`` has shape (batch, n_qubits).

    Per repetition: H on every qubit; RZ(x_i) on qubit i; then for each neighbour
    pair (i, i+1), no wrap-around: CNOT(i, i+1), RZ(x_i * x_(i+1)) on i+1, CNOT(i, i+1).
    The CNOT-RZ-CNOT sandwich is exactly the two-qubit gate ``IsingZZ(phi)``
    (verified numerically in tests/test_vqc.py), which PennyLane simulates as ONE
    operation instead of three, so we use it for speed. Same unitary, same results.
    """
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[:, i], wires=i)
        for i in range(n_qubits - 1):
            qml.IsingZZ(x[:, i] * x[:, i + 1], wires=[i, i + 1])


def angle_feature_map(x, n_qubits: int) -> None:
    """Simple angle encoding: one RX(x_i) rotation per qubit."""
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")


def build_qnode(n_qubits: int, encoding: str):
    """Return ``circuit(weights, x) -> <Z_0>`` for a batch ``x`` of shape (B, n_qubits).

    Batching trick (PennyLane "parameter broadcasting"): because ``x[:, i]`` is a whole
    vector of angles, ONE circuit execution evaluates every patient in the mini-batch.
    ``diff_method='backprop'`` differentiates the simulator itself (autograd), which is
    far faster than the parameter-shift rule on a simulator.
    """
    if encoding not in ("zz", "angle"):
        raise ValueError(f"encoding must be 'zz' or 'angle', got {encoding!r}")
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(weights, x):
        if encoding == "zz":
            zz_feature_map(x, n_qubits)
        else:
            angle_feature_map(x, n_qubits)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


# --------------------------------------------------------------------------- #
# scikit-learn style estimator
# --------------------------------------------------------------------------- #
class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational quantum classifier with fit / predict / predict_proba.

    Args:
        encoding: ``'zz'`` or ``'angle'``.
        n_layers: number of StronglyEntanglingLayers (trainable depth).
        lr: Adam step size.
        epochs: fixed number of passes over the training data (no early stopping).
        batch_size: patients per mini-batch.
        random_state: seed for weight initialisation and batch shuffling (42).

    Attributes after fit:
        weights_: trained dials, shape (n_layers, n_qubits, 3).
        loss_history_: list with one value per epoch = mean class-weighted binary
            cross-entropy over that epoch's mini-batches.
        classes_, n_qubits_, class_weights_.
    """

    def __init__(
        self,
        encoding: str = "zz",
        n_layers: int = 2,
        lr: float = 0.05,
        epochs: int = 20,
        batch_size: int = 32,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.encoding = encoding
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _prob_class1(z):
        """Map <Z_0> in [-1, 1] to P(class 1) in [0, 1].

        z = +1 means qubit 0 is |0> -> class 0; z = -1 means |1> -> class 1.
        """
        return (1.0 - z) / 2.0

    def _check_X(self, X) -> np.ndarray:
        X = check_array(X, dtype=np.float64)
        if X.min() < -ANGLE_TOL or X.max() > np.pi + ANGLE_TOL:
            raise ValueError(
                "VQC inputs must lie in [0, pi] (use PimaPreprocessor); "
                f"got range [{X.min():.4f}, {X.max():.4f}]."
            )
        return X

    # -- scikit-learn API -------------------------------------------------- #
    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("VQCClassifier is a binary classifier (needs 2 classes).")
        y01 = (y == self.classes_[1]).astype(np.float64)

        n, n_qubits = X.shape
        self.n_features_in_ = n_qubits
        self.n_qubits_ = n_qubits

        # Class weights like sklearn's 'balanced': n / (2 * n_c). Rare class counts more.
        counts = np.bincount(y01.astype(int), minlength=2)
        self.class_weights_ = n / (2.0 * counts)
        sample_w = self.class_weights_[y01.astype(int)]

        circuit = build_qnode(n_qubits, self.encoding)
        rng = np.random.default_rng(self.random_state)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=n_qubits)
        # Small random start (like the PennyLane variational-classifier demo's 0.01*randn).
        weights = pnp.array(rng.normal(0.0, 0.1, size=shape), requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        self.loss_history_ = []

        for _ in range(self.epochs):
            order = rng.permutation(n)
            epoch_losses, epoch_sizes = [], []
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb, wb = X[idx], y01[idx], sample_w[idx]

                def cost(w, xb=xb, yb=yb, wb=wb):
                    p = self._prob_class1(circuit(w, xb))
                    bce = -(yb * pnp.log(p + EPS) + (1 - yb) * pnp.log(1 - p + EPS))
                    return pnp.mean(wb * bce)

                weights, batch_loss = opt.step_and_cost(cost, weights)
                epoch_losses.append(float(batch_loss))
                epoch_sizes.append(len(idx))
            self.loss_history_.append(float(np.average(epoch_losses, weights=epoch_sizes)))

        self.weights_ = np.asarray(weights)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, ["weights_", "classes_"])
        X = self._check_X(X)
        if X.shape[1] != self.n_qubits_:
            raise ValueError(f"Expected {self.n_qubits_} columns, got {X.shape[1]}")
        # Rebuilt (cheap) instead of stored, so fitted models stay picklable for joblib.
        circuit = build_qnode(self.n_qubits_, self.encoding)
        w = pnp.array(self.weights_, requires_grad=False)
        p1 = np.concatenate(
            [
                np.asarray(self._prob_class1(circuit(w, X[s : s + 256])))
                for s in range(0, len(X), 256)
            ]
        )
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]


def make_model(encoding: str = "zz", n_layers: int = 2, **kwargs: Any) -> VQCClassifier:
    """Return a scikit-learn compatible Variational Quantum Classifier."""
    return VQCClassifier(encoding=encoding, n_layers=n_layers, **kwargs)

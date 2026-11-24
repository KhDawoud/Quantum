"""Variational Quantum Classifier (PennyLane default.qubit + StronglyEntanglingLayers).

Contract:
  - ``make_model(encoding='zz', n_layers=2, **kwargs)`` returning an estimator with
    ``fit(X, y)``, ``predict(X)``, ``predict_proba(X)``, and ``loss_history_``.
  - ``PARAM_GRID`` with <= 4 combinations (e.g. 2 learning rates x 2 epoch counts).
  - Day 1 gate: measure 1 fit at 8 qubits and check ``325 * fit_time <= 3 hours``.
"""

from __future__ import annotations

from typing import Any

# Small grid (<= 4 combinations) because VQC training is the runtime bottleneck.
PARAM_GRID: dict[str, list[Any]] = {
    "lr": [0.02, 0.05],
    "epochs": [20, 40],
}


def make_model(encoding: str = "zz", n_layers: int = 2, **kwargs: Any):
    """Return a scikit-learn compatible Variational Quantum Classifier."""
    raise NotImplementedError(
        "Implement the batched backprop VQC in src/model_vqc.py."
    )

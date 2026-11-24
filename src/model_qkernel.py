"""Quantum Kernel SVM classifier (PennyLane default.qubit + precomputed SVC).

Contract:
  - ``make_model(encoding='zz', **kwargs)`` returning an estimator with fit/predict.
  - ``PARAM_GRID`` with <= 8 combinations and ``gamma <= 1.0`` (e.g. {0.5, 1.0}).
  - ``kernel_stats(K)`` returning off-diagonal mean and variance for Experiment 1/3.
"""

from __future__ import annotations

from typing import Any

# At most 8 combinations; gamma must stay <= 1.0 so angles never exceed pi.
PARAM_GRID: dict[str, list[Any]] = {
    "C": [0.1, 1.0, 5.0, 10.0],
    "gamma": [0.5, 1.0],
}


def make_model(encoding: str = "zz", **kwargs: Any):
    """Return a scikit-learn compatible Quantum Kernel SVM estimator."""
    raise NotImplementedError(
        "Implement the ZZ/angle quantum kernel in src/model_qkernel.py."
    )


def kernel_stats(K) -> dict[str, float]:
    """Return mean and variance of off-diagonal entries of kernel matrix ``K``."""
    raise NotImplementedError(
        "Implement kernel_stats(K) for kernel concentration checks."
    )

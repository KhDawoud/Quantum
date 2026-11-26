"""Gold experiments runner (Nov 26 freeze rule).

Experiments:
  - Exp 1: Number of qubits (4, 6, 8) + kernel concentration
  - Exp 2: Training set size (30, 60, 120, 200)
  - Exp 3a: Kernel encoding (angle vs. ZZ)
  - Exp 3b: VQC encoding (angle vs. ZZ) + loss curves
  - Exp 4 (spare): VQC circuit depth (1, 2, 3 layers)
  - Exp 5: Positive control on make_moons (artificial curved data)
"""

from __future__ import annotations

# Member B's standalone contribution. The all-model Exp 2 entry point below
# remains separate until all four models are integrated.
from src.experiment_rbf import run_exp2_trainsize_rbf


def run_exp1_qubits() -> None:
    """Saves results/exp1_qubits.csv and figures/exp1_qubits.png."""
    raise NotImplementedError("Experiment 1 is not yet implemented.")


def run_exp2_trainsize() -> None:
    """Saves results/exp2_trainsize.csv and figures/exp2_trainsize.png."""
    raise NotImplementedError("Experiment 2 is not yet implemented.")


def run_exp3_encoding_kernel() -> None:
    """Saves results/exp3_encoding_kernel.csv and figures/exp3_encoding_kernel.png."""
    raise NotImplementedError("Experiment 3 (kernel) is not yet implemented.")


def run_exp3_encoding_vqc() -> None:
    """Saves results/exp3_encoding_vqc.csv and figures/exp3_encoding_vqc.png."""
    raise NotImplementedError("Experiment 3 (VQC) is not yet implemented.")


def run_exp4_depth() -> None:
    """Saves results/exp4_depth.csv."""
    raise NotImplementedError("Experiment 4 (optional) is not yet implemented.")


def run_exp5_moons() -> None:
    """Saves results/exp5_moons.csv and figures/exp5_moons.png."""
    raise NotImplementedError("Experiment 5 (control) is not yet implemented.")

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

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PimaPreprocessor, load_pima
from src.evaluate import EXPERIMENT_EVAL_SPLITS, compare_models, evaluate
# Member B's standalone contribution. The all-model Exp 2 entry point below
# remains separate until all four models are integrated.
from src.experiment_rbf import run_exp2_trainsize_rbf
from src.model_qkernel import (
    PARAM_GRID as QKERNEL_PARAM_GRID,
    classical_angle_kernel,
    compute_kernel_matrix,
    kernel_stats,
    make_model as make_qkernel_model,
)

ALWAYS_GUESS_FLOOR = 500.0 / 768.0


def _extract_metric_stats(
    summary: Any,
    csv_path: Path,
) -> dict[str, float]:
    """Read mean/std metrics from ``evaluate()`` output dict or saved per-split CSV."""
    df = pd.read_csv(csv_path) if csv_path.exists() else None
    stats: dict[str, float] = {}

    if isinstance(summary, dict):
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "train_time_sec",
            "train_time",
            "training_time",
            "fit_time",
        ):
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            target_key = (
                "train_time"
                if key in ("train_time_sec", "training_time", "fit_time")
                else key
            )
            if mean_key in summary:
                stats[f"{target_key}_mean"] = float(summary[mean_key])
                stats[f"{target_key}_std"] = float(summary.get(std_key, 0.0))
            elif isinstance(summary.get(key), dict) and "mean" in summary[key]:
                stats[f"{target_key}_mean"] = float(summary[key]["mean"])
                stats[f"{target_key}_std"] = float(summary[key].get("std", 0.0))

    if df is not None and not df.empty:
        col_map = {
            "accuracy": ("accuracy",),
            "balanced_accuracy": ("balanced_accuracy",),
            "macro_f1": ("macro_f1",),
            "train_time": ("train_time_sec", "train_time", "training_time", "fit_time"),
        }
        for target_key, candidates in col_map.items():
            if f"{target_key}_mean" in stats:
                continue
            for col in candidates:
                if col in df.columns:
                    stats[f"{target_key}_mean"] = float(df[col].mean())
                    stats[f"{target_key}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
                    break
        stats["n_splits"] = float(len(df))
    elif isinstance(summary, dict) and "n_splits" in summary:
        stats["n_splits"] = float(summary["n_splits"])
    else:
        stats["n_splits"] = float("nan")

    return stats


def run_exp1_qubits() -> None:
    """Saves results/exp1_qubits.csv and figures/exp1_qubits.png."""
    raise NotImplementedError("Experiment 1 is not yet implemented.")


def run_exp2_trainsize() -> None:
    """Saves results/exp2_trainsize.csv and figures/exp2_trainsize.png."""
    raise NotImplementedError("Experiment 2 is not yet implemented.")


def run_exp3_encoding_kernel(
    quick: bool = False,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> pd.DataFrame:
    """Compare simple angle encoding vs. ZZ feature map for the quantum kernel.

    Saves ``results/exp3_encoding_kernel.csv`` and ``figures/exp3_encoding_kernel.png``.
    """
    res_dir = Path(results_dir)
    fig_dir = Path(figures_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run both encodings through the shared zero-leakage evaluate() pipeline
    # using the explicit 5-split experiment count.
    summary_angle = evaluate(
        "qkernel_angle",
        lambda: make_qkernel_model(encoding="angle"),
        QKERNEL_PARAM_GRID,
        n_features=8,
        quick=quick,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )
    summary_zz = evaluate(
        "qkernel_zz",
        lambda: make_qkernel_model(encoding="zz"),
        QKERNEL_PARAM_GRID,
        n_features=8,
        quick=quick,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )

    angle_csv = res_dir / "qkernel_angle.csv"
    zz_csv = res_dir / "qkernel_zz.csv"
    metrics_angle = _extract_metric_stats(summary_angle, angle_csv)
    metrics_zz = _extract_metric_stats(summary_zz, zz_csv)

    # 2. Paired significance test on identical splits when both per-split CSVs exist.
    p_value = float("nan")
    mean_diff = float("nan")
    if angle_csv.exists() and zz_csv.exists():
        try:
            cmp_res = compare_models(str(angle_csv), str(zz_csv))
            if isinstance(cmp_res, dict):
                p_value = float(cmp_res.get("p_value", float("nan")))
                mean_diff = float(cmp_res.get("mean_diff", float("nan")))
        except NotImplementedError:
            pass

    # 3. Kernel concentration (off-diagonal spread) on cleaned [0, pi] Pima features.
    X_raw, y_raw = load_pima()
    X_scaled = PimaPreprocessor(n_features=8).fit_transform(X_raw, y_raw)
    K_angle = compute_kernel_matrix(X_scaled, encoding="angle", gamma=1.0, polish=True)
    K_zz = compute_kernel_matrix(X_scaled, encoding="zz", reps=2, gamma=1.0, polish=True)

    # Verify the unentangled angle kernel matches its exact classical formula.
    K_angle_classical = classical_angle_kernel(X_scaled, gamma=1.0)
    classical_angle_max_diff = float(np.max(np.abs(K_angle - K_angle_classical)))

    stats_by_enc = {
        "angle": kernel_stats(K_angle),
        "zz": kernel_stats(K_zz),
    }

    rows = []
    for enc, m in (("angle", metrics_angle), ("zz", metrics_zz)):
        k_st = stats_by_enc[enc]
        rows.append(
            {
                "encoding": enc,
                "classically_simulable": enc == "angle",
                "n_features": 8,
                "quick": bool(quick),
                "n_splits": m.get("n_splits", float("nan")),
                "accuracy_mean": m.get("accuracy_mean", float("nan")),
                "accuracy_std": m.get("accuracy_std", float("nan")),
                "balanced_accuracy_mean": m.get("balanced_accuracy_mean", float("nan")),
                "balanced_accuracy_std": m.get("balanced_accuracy_std", float("nan")),
                "macro_f1_mean": m.get("macro_f1_mean", float("nan")),
                "macro_f1_std": m.get("macro_f1_std", float("nan")),
                "train_time_mean": m.get("train_time_mean", float("nan")),
                "train_time_std": m.get("train_time_std", float("nan")),
                "kernel_mean": k_st["mean"],
                "kernel_var": k_st["var"],
                "kernel_std": k_st["std"],
                "kernel_min": k_st["min"],
                "kernel_max": k_st["max"],
                "classical_formula_max_diff": classical_angle_max_diff if enc == "angle" else float("nan"),
                "paired_mean_diff_angle_minus_zz": mean_diff,
                "paired_p_value": p_value,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = res_dir / "exp3_encoding_kernel.csv"
    out_df.to_csv(out_csv, index=False)

    # 4. Plot classification performance and off-diagonal kernel distribution.
    fig, (ax_perf, ax_dist) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)

    metric_names = ["Accuracy", "Balanced Acc", "Macro F1"]
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    x_pos = np.arange(len(metric_names))
    width = 0.34

    ang_means = [metrics_angle.get(f"{k}_mean", 0.0) for k in metric_keys]
    ang_stds = [metrics_angle.get(f"{k}_std", 0.0) for k in metric_keys]
    zz_means = [metrics_zz.get(f"{k}_mean", 0.0) for k in metric_keys]
    zz_stds = [metrics_zz.get(f"{k}_std", 0.0) for k in metric_keys]

    ax_perf.bar(x_pos - width / 2, ang_means, width, yerr=ang_stds, capsize=4, label="Angle (no entanglement)", color="#4C78A8")
    ax_perf.bar(x_pos + width / 2, zz_means, width, yerr=zz_stds, capsize=4, label="ZZ feature map (2 reps)", color="#F58518")
    ax_perf.axhline(ALWAYS_GUESS_FLOOR, color="#444444", linestyle="--", linewidth=1.2, label="Always-guess floor (65.1%)")
    ax_perf.set_xticks(x_pos)
    ax_perf.set_xticklabels(metric_names)
    ax_perf.set_ylim(0.0, 1.0)
    ax_perf.set_ylabel("Score")
    ax_perf.set_title("Quantum Kernel: Angle vs. ZZ (8 Qubits)")
    ax_perf.legend(loc="lower right", fontsize=8)
    ax_perf.grid(axis="y", alpha=0.25)

    off_mask = ~np.eye(K_angle.shape[0], dtype=bool)
    ax_dist.hist(
        K_angle[off_mask],
        bins=35,
        density=True,
        alpha=0.55,
        color="#4C78A8",
        label=f"Angle (mean={stats_by_enc['angle']['mean']:.3f}, var={stats_by_enc['angle']['var']:.2e})",
    )
    ax_dist.hist(
        K_zz[off_mask],
        bins=35,
        density=True,
        alpha=0.55,
        color="#F58518",
        label=f"ZZ (mean={stats_by_enc['zz']['mean']:.3f}, var={stats_by_enc['zz']['var']:.2e})",
    )
    ax_dist.set_xlabel("Off-diagonal kernel value K(x_i, x_j)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Off-Diagonal Kernel Spread on Pima (gamma=1.0)")
    ax_dist.legend(loc="upper right", fontsize=8)
    ax_dist.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_dir / "exp3_encoding_kernel.png", dpi=150)
    plt.close(fig)

    return out_df


def run_exp3_encoding_vqc() -> None:
    """Saves results/exp3_encoding_vqc.csv and figures/exp3_encoding_vqc.png."""
    raise NotImplementedError("Experiment 3 (VQC) is not yet implemented.")


def run_exp4_depth() -> None:
    """Saves results/exp4_depth.csv."""
    raise NotImplementedError("Experiment 4 (optional) is not yet implemented.")


def run_exp5_moons() -> None:
    """Saves results/exp5_moons.csv and figures/exp5_moons.png."""
    raise NotImplementedError("Experiment 5 (control) is not yet implemented.")

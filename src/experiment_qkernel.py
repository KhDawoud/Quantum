"""Quantum Kernel experiments runner (Experiments 1, 2, 3, and 5).

Uses only the team's shared splitting, preprocessing, and evaluation harness.
Run: python -m src.experiment_qkernel --exp 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

from src.data import PimaPreprocessor, load_pima
from src.evaluate import (
    EXPERIMENT_EVAL_SPLITS,
    compare_models,
    evaluate,
    get_fixed_test_split,
    get_splits,
    tune_hyperparameters,
)
from src.model_logreg import (
    PARAM_GRID as LOGREG_PARAM_GRID,
    make_model as make_logreg_model,
)
from src.model_qkernel import (
    PARAM_GRID,
    classical_angle_kernel,
    compute_kernel_matrix,
    hand_zz_2qubit_kernel,
    kernel_stats,
    make_model,
)

ALWAYS_GUESS_FLOOR = 500.0 / 768.0


def compute_exp1_kernel_spread(
    n_splits: int = EXPERIMENT_EVAL_SPLITS,
) -> pd.DataFrame:
    """Compute off-diagonal ``kernel_stats()`` across training splits for 4, 6, 8 columns."""
    X, y = load_pima()
    splits = get_splits(X, y)[:n_splits]
    rows: list[dict[str, Any]] = []
    for k in (4, 6, 8):
        for gamma in (0.5, 1.0):
            means, variances = [], []
            for train_idx, _ in splits:
                X_tr = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
                y_tr = y[train_idx]
                X_scaled = PimaPreprocessor(n_features=k).fit_transform(X_tr, y_tr)
                K = compute_kernel_matrix(
                    X_scaled, encoding="zz", reps=2, gamma=gamma, polish=True
                )
                st = kernel_stats(K)
                means.append(st["mean"])
                variances.append(st["var"])
            ddof = 1 if len(means) > 1 else 0
            rows.append(
                {
                    "n_features": k,
                    "gamma": gamma,
                    "kernel_mean_mean": float(np.mean(means)),
                    "kernel_mean_std": float(np.std(means, ddof=ddof)),
                    "kernel_var_mean": float(np.mean(variances)),
                    "kernel_var_std": float(np.std(variances, ddof=ddof)),
                }
            )
    return pd.DataFrame(rows)


def plot_exp1_qkernel(
    metrics_rows: list[dict[str, Any]],
    spread_df: pd.DataFrame,
    figures_dir: str | Path = "figures",
) -> Path:
    """Create figures/exp1_qkernel.png showing score and kernel spread vs. qubits."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "exp1_qkernel.png"

    ks = [int(r["n_features"]) for r in metrics_rows]
    bacc_m = [float(r["balanced_accuracy_mean"]) * 100.0 for r in metrics_rows]
    bacc_s = [float(r["balanced_accuracy_std"]) * 100.0 for r in metrics_rows]
    acc_m = [float(r["accuracy_mean"]) * 100.0 for r in metrics_rows]
    acc_s = [float(r["accuracy_std"]) * 100.0 for r in metrics_rows]

    fig, (ax_score, ax_var) = plt.subplots(
        1, 2, figsize=(10.5, 4.3), dpi=200, layout="constrained"
    )

    ax_score.errorbar(
        ks, bacc_m, yerr=bacc_s, marker="o", capsize=5, linewidth=2.0,
        color="#0E7C7B", label="Balanced accuracy (mean ± std)",
    )
    ax_score.errorbar(
        ks, acc_m, yerr=acc_s, marker="s", linestyle="--", capsize=5, linewidth=1.8,
        color="#2563EB", label="Accuracy (mean ± std)",
    )
    ax_score.axhline(
        ALWAYS_GUESS_FLOOR * 100.0, linestyle=":", linewidth=1.5, color="#DC2626",
        label=f"Always-guess floor ({ALWAYS_GUESS_FLOOR * 100.0:.1f}%)",
    )
    for x_k, y_b in zip(ks, bacc_m):
        ax_score.text(
            x_k + 0.16, y_b + 0.9, f"{y_b:.1f}%", ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color="#0E7C7B",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88),
        )
    ax_score.set(
        xlabel="Number of Columns (= Qubits)",
        ylabel="Score (%)",
        title="Quantum Kernel: Score vs. Number of Qubits\n(First 5 Stratified Splits, Balanced Weighting)",
        xticks=ks,
        xlim=(3.4, 8.6),
        ylim=(55.0, 86.0),
    )
    ax_score.grid(axis="y", alpha=0.25)
    ax_score.legend(loc="lower right", fontsize=8)

    for gamma, col, marker, ls in (
        (0.5, "#0E7C7B", "s", "--"),
        (1.0, "#D96B27", "^", "-"),
    ):
        sub = spread_df[spread_df["gamma"] == gamma].sort_values("n_features")
        ax_var.errorbar(
            sub["n_features"], sub["kernel_var_mean"], yerr=sub["kernel_var_std"],
            marker=marker, linestyle=ls, capsize=4, linewidth=2.0, color=col,
            label=f"ZZ gamma={gamma:g} (mean ± std across 5 train splits)",
        )
        for x_k, v_m in zip(sub["n_features"], sub["kernel_var_mean"]):
            mult = 1.22 if gamma == 0.5 else 0.72
            ax_var.text(x_k, v_m * mult, f"{v_m:.2e}", ha="center", va="center",
                        fontsize=8.0, fontweight="bold", color=col)

    ax_var.set_yscale("log")
    ax_var.set(
        xlabel="Number of Columns (= Qubits)",
        ylabel="Off-Diagonal Kernel Variance (log scale)",
        title="Kernel Concentration: Off-Diagonal Variance\n(Smaller Variance = Patients Look More Alike)",
        xticks=ks,
        xlim=(3.4, 8.6),
        ylim=(3.0e-3, 1.2e-1),
    )
    ax_var.grid(True, which="both", alpha=0.25)
    ax_var.legend(loc="lower left", fontsize=8)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_exp1_qkernel(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Run Experiment 1 (4, 6, 8 columns) for the quantum kernel."""
    res_dir = Path(results_dir)
    main_csv = res_dir / "qkernel.csv"
    if not main_csv.exists():
        raise FileNotFoundError(f"Missing 25-split baseline file: {main_csv}")

    q8_baseline_df = pd.read_csv(main_csv).iloc[:EXPERIMENT_EVAL_SPLITS].copy()

    # Step E: Run 4-column and 6-column variants on the first 5 splits.
    s_q4 = evaluate(
        "exp1_qkernel_q4",
        make_model,
        PARAM_GRID,
        n_features=4,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )
    s_q6 = evaluate(
        "exp1_qkernel_q6",
        make_model,
        PARAM_GRID,
        n_features=6,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )

    q4_df = s_q4["results"]
    q6_df = s_q6["results"]

    # Step F: Compare against 8-column baseline (first 5 rows of results/qkernel.csv).
    cmp_q4 = compare_models(q4_df, q8_baseline_df, metric="balanced_accuracy")
    cmp_q6 = compare_models(q6_df, q8_baseline_df, metric="balanced_accuracy")

    # Compute kernel_stats() at 4, 6, 8 columns across the 5 training splits.
    spread_df = compute_exp1_kernel_spread(n_splits=EXPERIMENT_EVAL_SPLITS)

    def _summarize_df(k: int, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_features": k,
            "n_splits": len(df),
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std(ddof=1)),
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(df["balanced_accuracy"].std(ddof=1)),
            "macro_f1_mean": float(df["macro_f1"].mean()),
            "macro_f1_std": float(df["macro_f1"].std(ddof=1)),
            "train_time_sec_mean": float(df["train_time_sec"].mean()),
            "train_time_sec_std": float(df["train_time_sec"].std(ddof=1)),
        }

    metrics_rows = [
        _summarize_df(4, q4_df),
        _summarize_df(6, q6_df),
        _summarize_df(8, q8_baseline_df),
    ]
    fig_path = plot_exp1_qkernel(metrics_rows, spread_df, figures_dir=figures_dir)

    print("=== Selected Features per Split (Verify Same Columns Across Models) ===")
    for k, df in ((4, q4_df), (6, q6_df), (8, q8_baseline_df)):
        print(f"n_features={k}:")
        for _, row in df.iterrows():
            print(f"  split {int(row['split'])}: {row['selected_features']}")

    print("\n=== Experiment 1 Metrics (First 5 Splits) ===")
    print(pd.DataFrame(metrics_rows).to_string(index=False))

    print("\n=== Experiment 1 Kernel Spread (kernel_stats across 5 train splits) ===")
    print(spread_df.to_string(index=False))

    print("\n=== Paired Comparison vs. 8-Column Baseline (balanced_accuracy) ===")
    for label, cmp in (("q4 vs q8", cmp_q4), ("q6 vs q8", cmp_q6)):
        print(
            f"  {label}: mean_a={cmp['mean_a']:.4f}, mean_b={cmp['mean_b']:.4f}, "
            f"diff={cmp['mean_diff']:+.4f}, p={cmp['p_value']:.4f} -> {cmp['conclusion']}"
        )

    print(f"\nSaved figure: {fig_path}")
    return {
        "metrics": metrics_rows,
        "spread": spread_df,
        "comparisons": {"q4_vs_q8": cmp_q4, "q6_vs_q8": cmp_q6},
        "figure_path": str(fig_path),
        "selected_features": {
            4: q4_df[["split", "selected_features"]].to_dict("records"),
            6: q6_df[["split", "selected_features"]].to_dict("records"),
            8: q8_baseline_df[["split", "selected_features"]].to_dict("records"),
        },
    }


def plot_exp2_qkernel(
    summary_rows: list[dict[str, Any]],
    full_pool_ref: dict[str, Any],
    fixed_test_floor: float,
    figures_dir: str | Path = "figures",
) -> Path:
    """Create figures/exp2_qkernel.png showing score and spread across 10 draws."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "exp2_qkernel.png"

    sizes = [int(r["train_size"]) for r in summary_rows]
    bacc_m = [float(r["balanced_accuracy_mean"]) * 100.0 for r in summary_rows]
    bacc_s = [float(r["balanced_accuracy_std"]) * 100.0 for r in summary_rows]
    acc_m = [float(r["accuracy_mean"]) * 100.0 for r in summary_rows]
    acc_s = [float(r["accuracy_std"]) * 100.0 for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200, layout="constrained")

    ax.errorbar(
        sizes,
        bacc_m,
        yerr=bacc_s,
        marker="o",
        capsize=5,
        linewidth=2.2,
        color="#0E7C7B",
        label="Balanced accuracy (mean ± std, 10 draws)",
    )
    ax.errorbar(
        sizes,
        acc_m,
        yerr=acc_s,
        marker="s",
        linestyle="--",
        capsize=5,
        linewidth=1.8,
        color="#2563EB",
        label="Accuracy (mean ± std, 10 draws)",
    )

    ref_bacc = float(full_pool_ref["balanced_accuracy_mean"]) * 100.0
    ax.axhline(
        ref_bacc,
        linestyle="-.",
        linewidth=1.5,
        color="#D96B27",
        label=f"Full training pool reference (n=614: {ref_bacc:.1f}% balanced acc)",
    )
    ax.axhline(
        fixed_test_floor * 100.0,
        linestyle=":",
        linewidth=1.5,
        color="#DC2626",
        label=f"Fixed-test majority floor ({fixed_test_floor * 100.0:.1f}%)",
    )

    for sz, bm in zip(sizes, bacc_m):
        ax.text(
            sz + 4,
            bm + 1.0,
            f"{bm:.1f}%",
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0E7C7B",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.88),
        )

    ax.set(
        xlabel="Training Patients per Draw (8 Qubits, Fixed Tuned Settings)",
        ylabel="Score on Fixed 154-Patient Test Set (%)",
        title="Quantum Kernel: Effect of Training Set Size\n(10 Stratified Draws per Size · Tuned Once on 614-Patient Pool)",
        xticks=sizes,
        xlim=(15, 225),
        ylim=(52.0, 86.0),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.2)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_exp2_qkernel(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Run Experiment 2 (training sizes 30, 60, 120, 200) for the quantum kernel."""
    res_dir = Path(results_dir)
    X, y = load_pima()
    pool_idx, test_idx = get_fixed_test_split(X, y)
    X_pool = X.iloc[pool_idx] if isinstance(X, pd.DataFrame) else X[pool_idx]
    y_pool = y[pool_idx]

    fixed_params = tune_hyperparameters(
        make_model, PARAM_GRID, X_pool, y_pool, n_features=8
    )
    print(f"=== Tuned Once on {len(pool_idx)}-Patient Pool ===")
    print(f"  fixed_params = {fixed_params}")

    fixed_test_floor = float(max((y[test_idx] == 0).mean(), (y[test_idx] == 1).mean()))
    sizes = (30, 60, 120, 200)
    summary_rows: list[dict[str, Any]] = []

    for size in sizes:
        summary = evaluate(
            f"exp2_qkernel_n{size}",
            make_model,
            PARAM_GRID,
            n_features=8,
            train_size=size,
            n_draws=10,
            fixed_params=fixed_params,
            results_dir=res_dir,
        )
        summary_rows.append(
            {
                "train_size": size,
                "n_draws": int(summary["n_splits"]),
                "accuracy_mean": float(summary["accuracy_mean"]),
                "accuracy_std": float(summary["accuracy_std"]),
                "balanced_accuracy_mean": float(summary["balanced_accuracy_mean"]),
                "balanced_accuracy_std": float(summary["balanced_accuracy_std"]),
                "macro_f1_mean": float(summary["macro_f1_mean"]),
                "macro_f1_std": float(summary["macro_f1_std"]),
                "train_time_sec_mean": float(summary["train_time_sec_mean"]),
                "train_time_sec_std": float(summary["train_time_sec_std"]),
                "results_path": summary["results_path"],
            }
        )

    # Full 614-patient pool reference on the same fixed 154-patient test set.
    with tempfile.TemporaryDirectory(prefix="qkernel-pool-ref-") as tmp_dir:
        full_ref = evaluate(
            "exp2_qkernel_fullpool_ref",
            make_model,
            PARAM_GRID,
            n_features=8,
            train_size=len(pool_idx),
            n_draws=1,
            fixed_params=fixed_params,
            results_dir=tmp_dir,
        )

    fig_path = plot_exp2_qkernel(
        summary_rows,
        full_ref,
        fixed_test_floor=fixed_test_floor,
        figures_dir=figures_dir,
    )

    print("\n=== Experiment 2 Summary (10 Draws per Size, Fixed 154-Patient Test Set) ===")
    print(f"  Fixed-test majority floor: {fixed_test_floor * 100.0:.2f}%")
    print(pd.DataFrame(summary_rows).drop(columns=["results_path"]).to_string(index=False))
    print(
        f"\n  Full-pool reference (n={len(pool_idx)}, 1 draw): "
        f"acc={full_ref['accuracy_mean'] * 100.0:.2f}%, "
        f"bacc={full_ref['balanced_accuracy_mean'] * 100.0:.2f}%, "
        f"macro_f1={full_ref['macro_f1_mean'] * 100.0:.2f}%, "
        f"time={full_ref['train_time_sec_mean']:.4f}s"
    )
    print(f"\nSaved figure: {fig_path}")

    return {
        "fixed_params": fixed_params,
        "fixed_test_floor": fixed_test_floor,
        "summary_rows": summary_rows,
        "full_pool_ref": {
            "n_train": int(len(pool_idx)),
            "accuracy": float(full_ref["accuracy_mean"]),
            "balanced_accuracy": float(full_ref["balanced_accuracy_mean"]),
            "macro_f1": float(full_ref["macro_f1_mean"]),
            "train_time_sec": float(full_ref["train_time_sec_mean"]),
        },
        "figure_path": str(fig_path),
    }


def compute_exp3_kernel_spread(
    n_splits: int = EXPERIMENT_EVAL_SPLITS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], float]:
    """Compute ``kernel_stats()`` across training splits for angle vs. ZZ encoding."""
    X, y = load_pima()
    splits = get_splits(X, y)[:n_splits]
    rows: list[dict[str, Any]] = []
    split0_offdiag: dict[str, np.ndarray] = {}
    classical_max_diff = 0.0

    for enc in ("angle", "zz"):
        for gamma in (0.5, 1.0):
            means, variances = [], []
            for split_idx, (train_idx, _) in enumerate(splits):
                X_tr = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
                y_tr = y[train_idx]
                X_scaled = PimaPreprocessor(n_features=8).fit_transform(X_tr, y_tr)
                K = compute_kernel_matrix(
                    X_scaled, encoding=enc, reps=2, gamma=gamma, polish=True
                )
                if enc == "angle":
                    K_cls = classical_angle_kernel(X_scaled, gamma=gamma)
                    classical_max_diff = max(
                        classical_max_diff, float(np.max(np.abs(K - K_cls)))
                    )
                if split_idx == 0 and gamma == 1.0:
                    mask = ~np.eye(K.shape[0], dtype=bool)
                    split0_offdiag[enc] = K[mask]

                st = kernel_stats(K)
                means.append(st["mean"])
                variances.append(st["var"])

            ddof = 1 if len(means) > 1 else 0
            rows.append(
                {
                    "encoding": enc,
                    "gamma": gamma,
                    "kernel_mean_mean": float(np.mean(means)),
                    "kernel_mean_std": float(np.std(means, ddof=ddof)),
                    "kernel_var_mean": float(np.mean(variances)),
                    "kernel_var_std": float(np.std(variances, ddof=ddof)),
                }
            )
    return pd.DataFrame(rows), split0_offdiag, classical_max_diff


def plot_exp3_qkernel(
    angle_df: pd.DataFrame,
    zz_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    split0_offdiag: dict[str, np.ndarray],
    figures_dir: str | Path = "figures",
) -> Path:
    """Create figures/exp3_qkernel.png comparing angle vs. ZZ scores and kernel spread."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "exp3_qkernel.png"

    fig, (ax_score, ax_dist) = plt.subplots(
        1, 2, figsize=(10.8, 4.4), dpi=200, layout="constrained"
    )

    metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    labels = ("Accuracy", "Balanced Accuracy", "Macro F1")
    x = np.arange(len(metrics))
    width = 0.34

    ang_m = [float(angle_df[m].mean()) * 100.0 for m in metrics]
    ang_s = [float(angle_df[m].std(ddof=1)) * 100.0 for m in metrics]
    zz_m = [float(zz_df[m].mean()) * 100.0 for m in metrics]
    zz_s = [float(zz_df[m].std(ddof=1)) * 100.0 for m in metrics]

    b1 = ax_score.bar(
        x - width / 2, ang_m, width, yerr=ang_s, capsize=4,
        color="#2563EB", label="Angle encoding (RY, no entanglement)",
    )
    b2 = ax_score.bar(
        x + width / 2, zz_m, width, yerr=zz_s, capsize=4,
        color="#D96B27", label="ZZ encoding (2 reps, entangled)",
    )
    ax_score.axhline(
        ALWAYS_GUESS_FLOOR * 100.0, linestyle=":", linewidth=1.5, color="#DC2626",
        label=f"Always-guess floor ({ALWAYS_GUESS_FLOOR * 100.0:.1f}%)",
    )
    for bar, mv, sv in zip(b1, ang_m, ang_s):
        ax_score.text(
            bar.get_x() + bar.get_width() / 2, mv + sv + 0.8, f"{mv:.1f}%",
            ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#2563EB",
        )
    for bar, mv, sv in zip(b2, zz_m, zz_s):
        ax_score.text(
            bar.get_x() + bar.get_width() / 2, mv + sv + 0.8, f"{mv:.1f}%",
            ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#D96B27",
        )

    ax_score.set(
        ylabel="Score (%)",
        title="Quantum Kernel: Angle vs. ZZ Encoding\n(First 5 Stratified Splits, 8 Qubits, Balanced Weighting)",
        xticks=x,
        xticklabels=labels,
        ylim=(52.0, 88.0),
    )
    ax_score.grid(axis="y", alpha=0.25)
    ax_score.legend(loc="lower right", fontsize=8)

    ang_g1 = spread_df[(spread_df["encoding"] == "angle") & (spread_df["gamma"] == 1.0)].iloc[0]
    zz_g1 = spread_df[(spread_df["encoding"] == "zz") & (spread_df["gamma"] == 1.0)].iloc[0]
    ax_dist.hist(
        split0_offdiag["angle"], bins=35, density=True, alpha=0.52, color="#2563EB",
        label=f"Angle (mean={ang_g1['kernel_mean_mean']:.3f}, var={ang_g1['kernel_var_mean']:.2e})",
    )
    ax_dist.hist(
        split0_offdiag["zz"], bins=35, density=True, alpha=0.58, color="#D96B27",
        label=f"ZZ (mean={zz_g1['kernel_mean_mean']:.3f}, var={zz_g1['kernel_var_mean']:.2e})",
    )
    ax_dist.set(
        xlabel="Off-Diagonal Kernel Value K(x_i, x_j)",
        ylabel="Density",
        title="Off-Diagonal Kernel Spread on Split-0 Train Data\n(8 Qubits, gamma = 1.0)",
    )
    ax_dist.grid(alpha=0.25)
    ax_dist.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_exp3_qkernel(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Run Experiment 3 (angle vs. ZZ encoding) for the quantum kernel."""
    res_dir = Path(results_dir)
    main_csv = res_dir / "qkernel.csv"
    if not main_csv.exists():
        raise FileNotFoundError(f"Missing 25-split ZZ baseline file: {main_csv}")

    zz_baseline_df = pd.read_csv(main_csv).iloc[:EXPERIMENT_EVAL_SPLITS].copy()

    # Step E: Run angle encoding on the exact same first 5 splits.
    s_angle = evaluate(
        "exp3_qkernel_angle",
        lambda: make_model(encoding="angle"),
        PARAM_GRID,
        n_features=8,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )
    angle_df = s_angle["results"]

    # Step F: Compare angle vs. ZZ baseline (first 5 rows of results/qkernel.csv).
    cmp_res = compare_models(angle_df, zz_baseline_df, metric="balanced_accuracy")

    # Kernel spread across the 5 training splits + classical angle formula check.
    spread_df, split0_offdiag, classical_max_diff = compute_exp3_kernel_spread(
        n_splits=EXPERIMENT_EVAL_SPLITS
    )

    fig_path = plot_exp3_qkernel(
        angle_df, zz_baseline_df, spread_df, split0_offdiag, figures_dir=figures_dir
    )

    def _summarize(label: str, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "encoding": label,
            "n_splits": len(df),
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std(ddof=1)),
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(df["balanced_accuracy"].std(ddof=1)),
            "macro_f1_mean": float(df["macro_f1"].mean()),
            "macro_f1_std": float(df["macro_f1"].std(ddof=1)),
            "train_time_sec_mean": float(df["train_time_sec"].mean()),
            "train_time_sec_std": float(df["train_time_sec"].std(ddof=1)),
        }

    summary_table = [_summarize("angle", angle_df), _summarize("zz", zz_baseline_df)]
    print("=== Experiment 3 Metrics (First 5 Splits, 8 Qubits) ===")
    print(pd.DataFrame(summary_table).to_string(index=False))

    print("\n=== Experiment 3 Kernel Spread (kernel_stats across 5 train splits) ===")
    print(spread_df.to_string(index=False))
    print(f"  Angle vs. classical closed-form max diff: {classical_max_diff:.3e}")

    print("\n=== Paired Comparison: Angle vs. ZZ Baseline (balanced_accuracy) ===")
    print(
        f"  angle vs zz: mean_a={cmp_res['mean_a']:.4f}, mean_b={cmp_res['mean_b']:.4f}, "
        f"diff={cmp_res['mean_diff']:+.4f}, p={cmp_res['p_value']:.4f} -> {cmp_res['conclusion']}"
    )
    print(f"\nSaved figure: {fig_path}")

    return {
        "metrics": summary_table,
        "spread": spread_df,
        "classical_angle_max_diff": classical_max_diff,
        "comparison": cmp_res,
        "figure_path": str(fig_path),
    }


def make_moons_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the artificial 2D curved positive-control dataset (768 points, noise=0.2)."""
    X_moons, y_moons = make_moons(n_samples=768, noise=0.2, random_state=42)
    return np.asarray(X_moons, dtype=np.float64), np.asarray(y_moons, dtype=np.int64)


def compute_exp5_kernel_spread(
    X_moons: np.ndarray,
    y_moons: np.ndarray,
    n_splits: int = EXPERIMENT_EVAL_SPLITS,
) -> tuple[pd.DataFrame, float]:
    """Compute off-diagonal ``kernel_stats()`` across training splits for 2-qubit moons."""
    splits = get_splits(X_moons, y_moons)[:n_splits]
    rows: list[dict[str, Any]] = []
    hand_zz_max_diff = 0.0

    for enc in ("zz", "angle"):
        for gamma in (0.5, 1.0):
            means, variances = [], []
            for split_idx, (train_idx, _) in enumerate(splits):
                X_tr = X_moons[train_idx]
                y_tr = y_moons[train_idx]
                X_scaled = PimaPreprocessor(
                    n_features=2, zero_as_missing_cols=()
                ).fit_transform(X_tr, y_tr)
                K = compute_kernel_matrix(
                    X_scaled, encoding=enc, reps=2, gamma=gamma, polish=True
                )
                if split_idx == 0 and enc == "zz":
                    K_hand = hand_zz_2qubit_kernel(X_scaled, reps=2, gamma=gamma)
                    hand_zz_max_diff = max(
                        hand_zz_max_diff, float(np.max(np.abs(K - K_hand)))
                    )
                st = kernel_stats(K)
                means.append(st["mean"])
                variances.append(st["var"])
            ddof = 1 if len(means) > 1 else 0
            rows.append(
                {
                    "encoding": enc,
                    "gamma": gamma,
                    "kernel_mean_mean": float(np.mean(means)),
                    "kernel_mean_std": float(np.std(means, ddof=ddof)),
                    "kernel_var_mean": float(np.mean(variances)),
                    "kernel_var_std": float(np.std(variances, ddof=ddof)),
                }
            )
    return pd.DataFrame(rows), hand_zz_max_diff


def plot_exp5_qkernel_moons(
    X_moons: np.ndarray,
    y_moons: np.ndarray,
    zz_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    logreg_df: pd.DataFrame,
    figures_dir: str | Path = "figures",
) -> Path:
    """Create figures/exp5_qkernel.png for the artificial make_moons positive control."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "exp5_qkernel.png"

    fig, (ax_boundary, ax_bar) = plt.subplots(
        1, 2, figsize=(11.2, 4.6), dpi=200, layout="constrained"
    )

    # Left panel: split-0 2D decision boundaries (train-only preprocessing).
    train_idx, test_idx = get_splits(X_moons, y_moons)[0]
    prep = PimaPreprocessor(n_features=2, zero_as_missing_cols=()).fit(
        X_moons[train_idx], y_moons[train_idx]
    )
    X_tr_scaled = prep.transform(X_moons[train_idx])

    zz_params = json.loads(str(zz_df.iloc[0]["best_params"]))
    lr_params = json.loads(str(logreg_df.iloc[0]["best_params"]))
    zz_clf = make_model(encoding="zz", **zz_params).fit(X_tr_scaled, y_moons[train_idx])
    lr_clf = make_logreg_model(**lr_params).fit(X_tr_scaled, y_moons[train_idx])

    x0_min, x0_max = float(X_moons[:, 0].min()) - 0.35, float(X_moons[:, 0].max()) + 0.35
    x1_min, x1_max = float(X_moons[:, 1].min()) - 0.35, float(X_moons[:, 1].max()) + 0.35
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 100),
        np.linspace(x1_min, x1_max, 100),
    )
    grid = np.column_stack([xx0.ravel(), xx1.ravel()])
    grid_scaled = prep.transform(grid)
    zz_scores = zz_clf.decision_function(grid_scaled).reshape(xx0.shape)
    lr_scores = lr_clf.decision_function(grid_scaled).reshape(xx0.shape)

    ax_boundary.contourf(
        xx0,
        xx1,
        (zz_scores > 0).astype(float),
        levels=[-0.5, 0.5, 1.5],
        colors=["#2563EB", "#D96B27"],
        alpha=0.20,
    )
    zz_contour = ax_boundary.contour(
        xx0, xx1, zz_scores, levels=[0.0], colors=["#111827"], linewidths=2.2
    )
    lr_contour = ax_boundary.contour(
        xx0,
        xx1,
        lr_scores,
        levels=[0.0],
        colors=["#475569"],
        linestyles="--",
        linewidths=1.8,
    )

    for cls_val, color, label in (
        (0, "#2563EB", "Class 0 (test)"),
        (1, "#D96B27", "Class 1 (test)"),
    ):
        tr_mask = y_moons[train_idx] == cls_val
        te_mask = y_moons[test_idx] == cls_val
        ax_boundary.scatter(
            X_moons[train_idx][tr_mask, 0],
            X_moons[train_idx][tr_mask, 1],
            c=color,
            s=14,
            alpha=0.22,
            edgecolors="none",
        )
        ax_boundary.scatter(
            X_moons[test_idx][te_mask, 0],
            X_moons[test_idx][te_mask, 1],
            c=color,
            s=36,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            label=label,
        )

    handles, labels = ax_boundary.get_legend_handles_labels()
    handles.extend([zz_contour.legend_elements()[0][0], lr_contour.legend_elements()[0][0]])
    labels.extend(["ZZ Quantum Kernel boundary", "Logistic Regression boundary"])
    ax_boundary.legend(handles, labels, loc="upper right", fontsize=7.6, framealpha=0.92)
    ax_boundary.set(
        xlabel="Moon Feature x0 (raw coordinates)",
        ylabel="Moon Feature x1 (raw coordinates)",
        title="Artificial Data: make_moons (n=768, noise=0.2)\nSplit-0 Decision Boundaries (2 Qubits)",
        xlim=(x0_min, x0_max),
        ylim=(x1_min, x1_max),
    )
    ax_boundary.grid(alpha=0.22)

    # Right panel: 5-split balanced accuracy comparison with spread + per-split dots.
    entries = [
        ("Logistic Regression\n(Linear Ref.)", logreg_df, "#64748B"),
        ("Angle Kernel\n(2q RY Ref.)", angle_df, "#2563EB"),
        ("ZZ Quantum Kernel\n(2q Entangled)", zz_df, "#D96B27"),
    ]
    x_pos = np.arange(len(entries))
    means = [float(df["balanced_accuracy"].mean()) * 100.0 for _, df, _ in entries]
    stds = [float(df["balanced_accuracy"].std(ddof=1)) * 100.0 for _, df, _ in entries]
    colors = [col for _, _, col in entries]

    bars = ax_bar.bar(
        x_pos,
        means,
        width=0.52,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.90,
        error_kw={"ecolor": "#111827", "elinewidth": 1.5},
    )
    rng = np.random.default_rng(42)
    for idx, (_, df, _) in enumerate(entries):
        pts = df["balanced_accuracy"].to_numpy(dtype=float) * 100.0
        jitter = rng.uniform(-0.08, 0.08, size=len(pts))
        ax_bar.scatter(
            np.full(len(pts), x_pos[idx]) + jitter,
            pts,
            color="#111827",
            s=18,
            zorder=4,
            alpha=0.75,
        )

    ax_bar.axhline(
        50.0,
        linestyle=":",
        linewidth=1.6,
        color="#DC2626",
        label="Always-guess floor (50.0%)",
    )
    for bar, mv, sv in zip(bars, means, stds):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            mv + sv + 1.6,
            f"{mv:.1f}%\n± {sv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#111827",
        )

    ax_bar.set(
        ylabel="Balanced Accuracy (%)",
        title="5-Split Balanced Accuracy on Artificial make_moons\n(2 Features = 2 Qubits, Mean ± Std Across 5 Splits)",
        xticks=x_pos,
        xticklabels=[lbl for lbl, _, _ in entries],
        ylim=(40.0, 113.0),
    )
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def run_exp5_qkernel_moons(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Run Experiment 5 (artificial make_moons positive control) standalone for Member C."""
    res_dir = Path(results_dir)
    X_moons, y_moons = make_moons_dataset()

    # Step E: Real Member C run on artificial make_moons (2 qubits, first 5 splits).
    s_zz = evaluate(
        "exp5_qkernel_moons",
        make_model,
        PARAM_GRID,
        n_features=2,
        X=X_moons,
        y=y_moons,
        zero_as_missing_cols=(),
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=res_dir,
    )
    zz_df = s_zz["results"]

    # Positive-control references on identical splits inside a temporary directory
    # so no extra teammate CSV files are written to results/.
    with tempfile.TemporaryDirectory(prefix="qkernel-moons-ref-") as tmp_dir:
        s_logreg = evaluate(
            "exp5_logreg_moons_ref",
            make_logreg_model,
            LOGREG_PARAM_GRID,
            n_features=2,
            X=X_moons,
            y=y_moons,
            zero_as_missing_cols=(),
            n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
            results_dir=tmp_dir,
        )
        s_angle = evaluate(
            "exp5_angle_moons_ref",
            lambda: make_model(encoding="angle"),
            PARAM_GRID,
            n_features=2,
            X=X_moons,
            y=y_moons,
            zero_as_missing_cols=(),
            n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
            results_dir=tmp_dir,
        )
        logreg_df = s_logreg["results"]
        angle_df = s_angle["results"]

    # Step F: Paired comparisons on balanced_accuracy across the identical 5 splits.
    cmp_vs_logreg = compare_models(zz_df, logreg_df, metric="balanced_accuracy")
    cmp_vs_angle = compare_models(zz_df, angle_df, metric="balanced_accuracy")

    # Kernel spread + exact 2-qubit linear-algebra cross-check on split 0.
    spread_df, hand_zz_max_diff = compute_exp5_kernel_spread(
        X_moons, y_moons, n_splits=EXPERIMENT_EVAL_SPLITS
    )

    # Step G: Positive-control figure.
    fig_path = plot_exp5_qkernel_moons(
        X_moons, y_moons, zz_df, angle_df, logreg_df, figures_dir=figures_dir
    )

    def _summarize(label: str, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "model": label,
            "n_splits": len(df),
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std(ddof=1)),
            "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(df["balanced_accuracy"].std(ddof=1)),
            "macro_f1_mean": float(df["macro_f1"].mean()),
            "macro_f1_std": float(df["macro_f1"].std(ddof=1)),
            "train_time_sec_mean": float(df["train_time_sec"].mean()),
            "train_time_sec_std": float(df["train_time_sec"].std(ddof=1)),
        }

    summary_table = [
        _summarize("zz_qkernel", zz_df),
        _summarize("angle_qkernel_ref", angle_df),
        _summarize("logreg_ref", logreg_df),
    ]

    print("=== Experiment 5 Metrics (Artificial make_moons, n=768, noise=0.2, 5 Splits) ===")
    print(f"  Majority floor: {s_zz['majority_floor'] * 100.0:.2f}%")
    print(pd.DataFrame(summary_table).to_string(index=False))

    print("\n=== Per-Split Best Params & Scores (ZZ Quantum Kernel) ===")
    print(
        zz_df[
            ["split", "accuracy", "balanced_accuracy", "macro_f1", "train_time_sec", "best_params"]
        ].to_string(index=False)
    )

    print("\n=== Per-Split Best Params & Scores (Angle Reference & LogReg Reference) ===")
    print("Angle ref:")
    print(angle_df[["split", "balanced_accuracy", "best_params"]].to_string(index=False))
    print("LogReg ref:")
    print(logreg_df[["split", "balanced_accuracy", "best_params"]].to_string(index=False))

    print("\n=== Experiment 5 Kernel Spread (kernel_stats across 5 train splits) ===")
    print(spread_df.to_string(index=False))
    print(f"  ZZ vs. 2-qubit hand algebra max diff (split 0): {hand_zz_max_diff:.3e}")

    print("\n=== Paired Comparisons on balanced_accuracy (5 Splits) ===")
    for label, cmp in (
        ("zz_qkernel vs logreg_ref", cmp_vs_logreg),
        ("zz_qkernel vs angle_qkernel_ref", cmp_vs_angle),
    ):
        print(
            f"  {label}: mean_a={cmp['mean_a']:.4f}, mean_b={cmp['mean_b']:.4f}, "
            f"diff={cmp['mean_diff']:+.4f}, p={cmp['p_value']:.4f} -> {cmp['conclusion']}"
        )

    print(f"\nSaved CSV: {s_zz['results_path']}")
    print(f"Saved figure: {fig_path}")

    return {
        "metrics": summary_table,
        "zz_df": zz_df,
        "angle_df": angle_df,
        "logreg_df": logreg_df,
        "spread": spread_df,
        "hand_zz_max_diff": hand_zz_max_diff,
        "comparisons": {
            "zz_vs_logreg": cmp_vs_logreg,
            "zz_vs_angle": cmp_vs_angle,
        },
        "results_path": str(s_zz["results_path"]),
        "figure_path": str(fig_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantum Kernel experiments.")
    parser.add_argument("--exp", type=int, choices=(1, 2, 3, 5), default=1)
    args = parser.parse_args()
    if args.exp == 1:
        run_exp1_qkernel()
    elif args.exp == 2:
        run_exp2_qkernel()
    elif args.exp == 3:
        run_exp3_qkernel()
    elif args.exp == 5:
        run_exp5_qkernel_moons()

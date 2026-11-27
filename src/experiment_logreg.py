"""Person A's standalone Experiment 1 and Experiment 2 runners (Logistic Regression).

Evaluates Logistic Regression through the shared ``evaluate()`` harness:
  - Exp 1: 4 and 6 training-ranked Pima columns vs. 8-column baseline (first 5 splits).
  - Exp 2: 30, 60, 120, and 200 training patients (10 draws each, fixed 154-patient test set).

Run: ``python -m src.experiment_logreg [exp1|exp2]``
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_pima
from src.evaluate import (
    EXPERIMENT_EVAL_SPLITS,
    compare_models,
    evaluate,
    get_fixed_test_split,
    tune_hyperparameters,
)
from src.model_logreg import PARAM_GRID, make_model

TRAIN_SIZES: tuple[int, ...] = (30, 60, 120, 200)
EXP2_N_DRAWS = 10


def run_exp1_qubits_logreg(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    baseline_csv: str | Path = "results/logreg.csv",
) -> dict[str, Any]:
    """Run Logistic Regression on 4 and 6 features; compare with 8-feature baseline."""
    baseline_path = Path(baseline_csv)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Main baseline results file not found: {baseline_path}"
        )
    df_full = pd.read_csv(baseline_path)
    if len(df_full) < EXPERIMENT_EVAL_SPLITS:
        raise ValueError(
            f"Baseline CSV needs >= {EXPERIMENT_EVAL_SPLITS} rows, got {len(df_full)}."
        )
    df_q8 = df_full.iloc[:EXPERIMENT_EVAL_SPLITS].copy().reset_index(drop=True)
    if list(df_q8["split"]) != list(range(EXPERIMENT_EVAL_SPLITS)):
        raise ValueError("Baseline first 5 rows must correspond to splits 0..4.")

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_q4 = evaluate(
        "exp1_logreg_q4",
        make_model,
        PARAM_GRID,
        n_features=4,
        quick=False,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=out_dir,
    )
    summary_q6 = evaluate(
        "exp1_logreg_q6",
        make_model,
        PARAM_GRID,
        n_features=6,
        quick=False,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        results_dir=out_dir,
    )
    df_q4 = summary_q4["results"].copy().reset_index(drop=True)
    df_q6 = summary_q6["results"].copy().reset_index(drop=True)

    comparisons = {
        "q4_vs_q8": compare_models(df_q4, df_q8, metric="balanced_accuracy"),
        "q6_vs_q8": compare_models(df_q6, df_q8, metric="balanced_accuracy"),
    }
    figure_path = plot_exp1_logreg(
        df_q4,
        df_q6,
        df_q8,
        figures_dir=figures_dir,
        majority_floor=summary_q4["majority_floor"],
    )
    return {
        "q4": df_q4,
        "q6": df_q6,
        "q8": df_q8,
        "comparisons": comparisons,
        "figure_path": figure_path,
    }


def _plot_metric_curve(
    x_values: tuple[int, ...] | list[int],
    frames: list[pd.DataFrame],
    *,
    x_offset: float,
    xlabel: str,
    title: str,
    floor: float,
    floor_label: str,
    ylim: tuple[float, float],
    out_path: Path,
) -> Path:
    """Plot balanced accuracy and accuracy (mean +/- sample SD) across an axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    xs = np.array(x_values, dtype=float)
    bal_means = [float(df["balanced_accuracy"].mean()) for df in frames]
    bal_stds = [float(df["balanced_accuracy"].std(ddof=1)) for df in frames]
    acc_means = [float(df["accuracy"].mean()) for df in frames]
    acc_stds = [float(df["accuracy"].std(ddof=1)) for df in frames]

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.errorbar(
        xs - x_offset,
        bal_means,
        yerr=bal_stds,
        marker="o",
        capsize=5,
        color="#2459a6",
        linewidth=2.0,
        label="Balanced accuracy (mean +/- sample SD)",
    )
    ax.errorbar(
        xs + x_offset,
        acc_means,
        yerr=acc_stds,
        marker="s",
        linestyle="--",
        capsize=5,
        color="#d95f02",
        linewidth=1.8,
        label="Accuracy (mean +/- sample SD)",
    )
    ax.axhline(
        floor,
        linestyle=":",
        color="#666666",
        linewidth=1.4,
        label=floor_label,
    )
    ax.set(
        xlabel=xlabel,
        ylabel="Score",
        ylim=ylim,
        title=title,
        xticks=list(x_values),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_exp1_logreg(
    df_q4: pd.DataFrame,
    df_q6: pd.DataFrame,
    df_q8: pd.DataFrame,
    figures_dir: str | Path,
    majority_floor: float,
) -> Path:
    """Plot balanced accuracy and accuracy (mean +/- sample SD) vs. column count."""
    return _plot_metric_curve(
        [4, 6, 8],
        [df_q4, df_q6, df_q8],
        x_offset=0.06,
        xlabel="Number of columns / measurements (4, 6, 8)",
        title=(
            "Experiment 1 — Logistic Regression: score vs. number of columns\n"
            "First 5 stratified splits · top-k columns ranked by f_classif on training rows"
        ),
        floor=majority_floor,
        floor_label=f"Always-guess floor ({majority_floor:.1%})",
        ylim=(0.50, 0.90),
        out_path=Path(figures_dir) / "exp1_logreg.png",
    )


def run_exp2_trainsize_logreg(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    """Tune once on 614-patient pool; run 10 draws at 30, 60, 120, and 200 patients."""
    X, y = load_pima()
    pool_idx, test_idx = get_fixed_test_split(X, y)
    tuned_params = tune_hyperparameters(
        make_model, PARAM_GRID, X.iloc[pool_idx], y[pool_idx]
    )
    test_floor = float(max((y[test_idx] == 0).mean(), (y[test_idx] == 1).mean()))

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_size: dict[int, pd.DataFrame] = {}
    for size in TRAIN_SIZES:
        summary = evaluate(
            f"exp2_logreg_n{size}",
            make_model,
            PARAM_GRID,
            n_features=8,
            train_size=size,
            quick=False,
            fixed_params=tuned_params,
            n_draws=EXP2_N_DRAWS,
            results_dir=out_dir,
        )
        by_size[size] = summary["results"].copy().reset_index(drop=True)

    figure_path = plot_exp2_logreg(
        by_size, figures_dir=figures_dir, test_floor=test_floor
    )
    return {
        "tuned_params": tuned_params,
        "pool_size": len(pool_idx),
        "test_size": len(test_idx),
        "test_floor": test_floor,
        "by_size": by_size,
        "figure_path": figure_path,
    }


def plot_exp2_logreg(
    by_size: dict[int, pd.DataFrame],
    figures_dir: str | Path,
    test_floor: float,
) -> Path:
    """Plot balanced accuracy and accuracy (mean +/- sample SD) vs. training size."""
    return _plot_metric_curve(
        TRAIN_SIZES,
        [by_size[size] for size in TRAIN_SIZES],
        x_offset=1.8,
        xlabel="Number of training patients (10 stratified draws per size)",
        title=(
            "Experiment 2 — Logistic Regression: score vs. training size\n"
            "Fixed 154-patient test set · C tuned once on 614-patient pool"
        ),
        floor=test_floor,
        floor_label=f"Fixed-test majority floor ({test_floor:.1%})",
        ylim=(0.50, 0.85),
        out_path=Path(figures_dir) / "exp2_logreg.png",
    )


def _print_exp1_report(exp1_out: dict[str, Any]) -> None:
    print("Experiment 1 (Pima column count, Logistic Regression, first 5 splits):")
    for k, key in ((4, "q4"), (6, "q6"), (8, "q8")):
        df = exp1_out[key]
        print(
            f"  k={k}: bal_acc={df.balanced_accuracy.mean():.2%} +/- {df.balanced_accuracy.std():.2%} | "
            f"acc={df.accuracy.mean():.2%} +/- {df.accuracy.std():.2%} | "
            f"f1={df.macro_f1.mean():.2%} +/- {df.macro_f1.std():.2%} | "
            f"time={df.train_time_sec.mean():.4f}s +/- {df.train_time_sec.std():.4f}s"
        )
    print("\nSelected features in each exam (first 5 splits):")
    for k, key in ((4, "q4"), (6, "q6"), (8, "q8")):
        df = exp1_out[key]
        print(f"  k={k}:")
        for _, row in df.iterrows():
            print(f"    split {int(row['split'])}: {row['selected_features']}")
    print("\nPaired Nadeau-Bengio comparisons on balanced_accuracy vs. 8-column baseline:")
    for label, cmp in exp1_out["comparisons"].items():
        print(
            f"  {label}: mean_a={cmp['mean_a']:.4f}, mean_b={cmp['mean_b']:.4f}, "
            f"diff={cmp['mean_diff']:+.4f}, se={cmp['corrected_se']:.4f}, "
            f"t={cmp['t_stat']:+.4f}, df={cmp['df']}, p={cmp['p_value']:.4f} -> {cmp['conclusion']}"
        )
    print(f"Saved figure: {exp1_out['figure_path']}")


def _print_exp2_report(exp2_out: dict[str, Any]) -> None:
    print(
        "\nExperiment 2 (Training size, Logistic Regression, "
        f"fixed {exp2_out['test_size']}-patient test set, 10 draws/size):"
    )
    print(
        f"  Tuned once on {exp2_out['pool_size']}-patient pool -> "
        f"fixed_params={exp2_out['tuned_params']}"
    )
    print(f"  Fixed-test majority floor : {exp2_out['test_floor']:.2%}")
    for size in TRAIN_SIZES:
        df = exp2_out["by_size"][size]
        bal = df["balanced_accuracy"]
        acc = df["accuracy"]
        f1 = df["macro_f1"]
        t = df["train_time_sec"]
        print(
            f"  n={size:3d}: "
            f"bal_acc={bal.mean():.2%} +/- {bal.std():.2%} (range {bal.min():.2%}..{bal.max():.2%}) | "
            f"acc={acc.mean():.2%} +/- {acc.std():.2%} | "
            f"f1={f1.mean():.2%} +/- {f1.std():.2%} | "
            f"time={t.mean():.4f}s +/- {t.std():.4f}s"
        )
    print(f"Saved figure: {exp2_out['figure_path']}")


if __name__ == "__main__":
    requested = set(sys.argv[1:]) or {"exp1", "exp2"}
    if "exp1" in requested:
        _print_exp1_report(run_exp1_qubits_logreg())
    if "exp2" in requested:
        _print_exp2_report(run_exp2_trainsize_logreg())

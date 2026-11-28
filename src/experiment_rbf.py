"""Model B's contributions to Experiments 1 and 2.

Uses only the team's shared splitting, tuning and evaluation functions.
Run: python -m src.experiment_rbf exp1
     python -m src.experiment_rbf exp2 (also the default for compatibility)
"""

from pathlib import Path
import argparse
import tempfile

import pandas as pd

from src.data import load_pima
from src.evaluate import (
    EXPERIMENT_EVAL_SPLITS, compare_models, get_fixed_test_split,
    tune_hyperparameters, evaluate,
)
from src.model_rbf_svm import PARAM_GRID, make_model


def run_exp1_qubits_rbf(
    results_dir="results", figures_dir="figures", baseline_csv="results/rbf_svm.csv",
):
    """Compare 4/6 training-ranked columns with the first five 8-column results.

    Follows Member A's src/experiment_logreg.py protocol and the notebook's
    existing exp1_rbf_svm_q4/q6 filenames. No new 8-column CSV is produced.
    """
    baseline = pd.read_csv(baseline_csv).iloc[:EXPERIMENT_EVAL_SPLITS].copy()
    if baseline["split"].tolist() != list(range(EXPERIMENT_EVAL_SPLITS)):
        raise ValueError("Baseline must contain the first five splits in order (0..4).")
    if not baseline["selected_features"].str.split(",").map(len).eq(8).all():
        raise ValueError("Baseline must use all eight features.")
    frames = {8: baseline}
    for k in (4, 6):
        summary = evaluate(
            f"exp1_rbf_svm_q{k}", make_model, PARAM_GRID,
            n_features=k, quick=False, n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
            results_dir=results_dir,
        )
        frames[k] = summary["results"].copy()
    comparisons = {
        f"q{k}_vs_q8": compare_models(frames[k], baseline, metric="balanced_accuracy")
        for k in (4, 6)
    }
    figure_path = plot_exp1_rbf(frames, figures_dir)
    return {"q4": frames[4], "q6": frames[6], "q8": baseline,
            "comparisons": comparisons, "figure_path": figure_path}


def plot_exp1_rbf(frames, figures_dir):
    """Show mean and sample SD, with metric-specific majority/chance baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ks = (4, 6, 8)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for metric, offset, color, label in (
        ("balanced_accuracy", -0.06, "#7C3AED", "Balanced accuracy"),
        ("accuracy", 0.06, "#d95f02", "Accuracy"),
    ):
        ax.errorbar([k + offset for k in ks],
                    [frames[k][metric].mean() for k in ks],
                    yerr=[frames[k][metric].std(ddof=1) for k in ks],
                    marker="o", capsize=5, color=color, label=f"{label}: mean ± SD")
    ax.axhline(500 / 768, linestyle=":", color="#777777", label="Majority accuracy: 65.1%")
    ax.axhline(0.5, linestyle="--", color="#aaaaaa", label="Balanced accuracy floor: 50%")
    ax.set(xlabel="Number of training-ranked clinical columns", ylabel="Score",
           title="Experiment 1 — RBF SVM\nFirst 5 stratified splits · full 8-setting grid",
           xticks=ks, ylim=(0.47, 0.90))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    path = Path(figures_dir) / "exp1_rbf_svm.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_exp2_trainsize_rbf(results_dir="results", figures_dir="figures"):
    """Tune once on 614 patients; evaluate 10 draws at each of four sizes."""
    X, y = load_pima()
    pool, test = get_fixed_test_split(X, y)
    fixed = tune_hyperparameters(make_model, PARAM_GRID, X.iloc[pool], y[pool])
    rows = []
    # Per-size intermediate files are temporary; the deliverable preserves all draws.
    with tempfile.TemporaryDirectory(prefix="rbf-trainsize-") as intermediate:
        for size in (30, 60, 120, 200):
            summary = evaluate(
                f"rbf_size_{size}", make_model, PARAM_GRID,
                X=X, y=y, n_features=8, train_size=size, n_draws=10,
                fixed_params=fixed, results_dir=intermediate,
            )
            frame = summary["results"].copy()
            frame.insert(0, "model", "rbf_svm")
            frame["tuning_pool_size"] = len(pool)
            frame["test_majority_floor"] = max((y[test] == 0).mean(), (y[test] == 1).mean())
            rows.append(frame)
    results = pd.concat(rows, ignore_index=True)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "exp2_trainsize_rbf.csv", index=False)
    plot_trainsize(results, figures_dir)
    return results


def plot_trainsize(results, figures_dir):
    """Accuracy mean and sample SD; repeated draws share the same test patients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    stats = results.groupby("n_train")["accuracy"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                marker="o", capsize=5, color="#2459a6", label="RBF SVM: mean ± sample SD")
    floor = results["test_majority_floor"].iloc[0]
    ax.axhline(floor, linestyle="--", color="#777777",
               label=f"Fixed-test majority baseline ({floor:.1%})")
    ax.set(xlabel="Training patients", ylabel="Test accuracy", ylim=(0.4, 1),
           title="RBF SVM: effect of training size\n10 draws per size · fixed 154-patient test set",
           xticks=stats.index)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    dest = Path(figures_dir)
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "exp2_trainsize_rbf.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("exp1", "exp2"), default="exp2")
    args = parser.parse_args()
    if args.experiment == "exp1":
        result = run_exp1_qubits_rbf()
        for k in (4, 6, 8):
            frame = result[f"q{k}"]
            print(f"{k} columns:\n{frame[['accuracy', 'balanced_accuracy', 'macro_f1', 'train_time_sec']].agg(['mean', 'std'])}")
        print(result["comparisons"])
        print(f"Saved figure: {result['figure_path']}")
    else:
        result = run_exp2_trainsize_rbf()
        print(result.groupby("n_train")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"]))

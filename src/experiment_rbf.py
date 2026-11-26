"""Model B's contribution to Experiment 2; not the all-model comparison.

Uses only the team's shared splitting, tuning and evaluation functions.
Run: python -m src.experiment_rbf
"""

from pathlib import Path
import tempfile

import pandas as pd

from src.data import load_pima
from src.evaluate import get_fixed_test_split, tune_hyperparameters, evaluate
from src.model_rbf_svm import PARAM_GRID, make_model


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
    result = run_exp2_trainsize_rbf()
    print(result.groupby("n_train")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"]))

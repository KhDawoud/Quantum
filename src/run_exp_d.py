"""Member D's VQC experiment runner (experiments 3, 1, 2, 5, 6).

Usage (from the repo root, one run at a time):
    .venv/Scripts/python -m src.run_exp_d exp3 --quick   # quick check -> scratch_quick/
    .venv/Scripts/python -m src.run_exp_d exp3           # real run -> results/experiments/
    .venv/Scripts/python -m src.run_exp_d exp3-report    # comparison table + figure

All training goes through the shared ``src.evaluate.evaluate()``; this file only
chooses the settings, saves logs, writes manifest rows and draws figures.
Baselines are never saved as new files: the ZZ baseline is the first 5 rows of
``results/vqc_zz_25.csv`` read into memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    EXPERIMENT_EVAL_SPLITS,
    compare_models,
    evaluate,
    format_summary,
)
from src.model_vqc import PARAM_GRID, make_model

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "results" / "experiments"
LOG_DIR = EXP_DIR / "logs"
FIG_DIR = ROOT / "figures" / "experiments"
SCRATCH_DIR = ROOT / "scratch_quick"
MANIFEST = EXP_DIR / "MANIFEST_D.md"
MAIN_ZZ = ROOT / "results" / "vqc_zz_25.csv"

MANIFEST_HEADER = (
    "| file | experiment | what changed | what stayed the same | exams or draws used "
    "| columns | encoding | layers | settings list | tuning | git commit "
    "| date and time | exact command | baseline used |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - git missing
        return "unverified"


def _now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")


def _baseline_first5() -> pd.DataFrame:
    """First 5 exams of the 25-exam ZZ main run (in memory only, never saved)."""
    return pd.read_csv(MAIN_ZZ).head(EXPERIMENT_EVAL_SPLITS)


def _refuse_if_exists(path: Path) -> None:
    if path.exists():
        sys.exit(f"STOP: {path} already exists. Not overwriting; ask before re-running.")


def _add_manifest_row(cells: list[str]) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        MANIFEST.write_text(
            "# MANIFEST_D: Member D's experiment result files\n\n"
            "One row per result file, added right after each run.\n\n" + MANIFEST_HEADER,
            encoding="utf-8",
        )
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(c.replace("|", "/") for c in cells) + " |\n")


def _run(name: str, model_factory, *, quick: bool, command: str, manifest: dict | None,
         param_grid: dict | None = None, **eval_kwargs) -> dict:
    """Run one evaluate() call; real runs save a log (and a manifest row if given)."""
    grid = PARAM_GRID if param_grid is None else param_grid
    if quick:
        summary = evaluate(name, model_factory, grid, quick=True,
                           results_dir=SCRATCH_DIR, **{**eval_kwargs, "n_splits_to_run": 1})
        print(format_summary(summary))
        print("QUICK CHECK ONLY: numbers are not used.")
        return summary

    target = EXP_DIR / f"{name}.csv"
    _refuse_if_exists(target)
    started = _now()
    summary = evaluate(name, model_factory, grid, results_dir=EXP_DIR, **eval_kwargs)
    text = format_summary(summary)
    print(text)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    winners = summary["results"]["best_params"].tolist()
    (LOG_DIR / f"{name}.txt").write_text(
        text + f"\n  Started            : {started}\n  Finished           : {_now()}\n"
        f"  Winning settings   : {winners}\n  Settings grid      : {json.dumps(grid)}\n",
        encoding="utf-8",
    )
    if manifest is None:
        return summary
    _add_manifest_row([
        f"`results/experiments/{name}.csv`",
        manifest["experiment"],
        manifest["changed"],
        manifest["same"],
        manifest["exams"],
        manifest["columns"],
        manifest["encoding"],
        manifest["layers"],
        json.dumps(grid),
        manifest["tuning"],
        _git_short_hash(),
        started,
        f"`{command}`",
        manifest["baseline"],
    ])
    return summary


# --------------------------------------------------------------------------- #
# experiment 3: angle vs ZZ encoding
# --------------------------------------------------------------------------- #
EXP3_NAME = "exp3_vqc_angle_5ex"


def run_exp3(quick: bool) -> None:
    _run(
        EXP3_NAME,
        lambda: make_model(encoding="angle"),
        quick=quick,
        command="python -m src.run_exp_d exp3",
        n_features=8,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        manifest={
            "experiment": "3 (encoding)",
            "changed": "encoding ZZ -> angle (one RX per qubit)",
            "same": "2 layers, PARAM_GRID, 8 columns, first 5 exams, seed 42",
            "exams": "first 5 exams (splits 0-4)",
            "columns": "8",
            "encoding": "angle",
            "layers": "2",
            "tuning": "4 combinations per exam (3-fold inner GridSearchCV)",
            "baseline": "first 5 rows of results/vqc_zz_25.csv (ZZ), in memory",
        },
    )


def report_exp3() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angle = pd.read_csv(EXP_DIR / f"{EXP3_NAME}.csv")
    zz = _baseline_first5()

    rows = []
    for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
        res = compare_models(angle, zz, metric=metric)
        rows.append({
            "metric": metric,
            "angle_mean": res["mean_a"],
            "angle_std": float(angle[metric].std(ddof=1)),
            "zz_mean": res["mean_b"],
            "zz_std": float(zz[metric].std(ddof=1)),
            "mean_diff_angle_minus_zz": res["mean_diff"],
            "t_stat": res["t_stat"],
            "df": res["df"],
            "p_value": res["p_value"],
            "conclusion": res["conclusion"],
        })
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp3_vqc_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))

    print("\nPer exam (balanced accuracy):")
    per = pd.DataFrame({
        "exam": angle["split"],
        "zz": zz["balanced_accuracy"].round(4),
        "angle": angle["balanced_accuracy"].round(4),
        "angle_best_params": angle["best_params"],
        "angle_time_s": angle["train_time_sec"].round(0),
        "zz_time_s": zz["train_time_sec"].round(0),
    })
    print(per.to_string(index=False))

    # Figure: per-exam points joined by lines, plus mean +/- std.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "exp3_vqc.png"
    _refuse_if_exists(fig_path)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    xs = np.array([0.0, 1.0])
    for z, a in zip(zz["balanced_accuracy"], angle["balanced_accuracy"]):
        ax.plot(xs, [z, a], color="0.75", lw=1, zorder=1)
    for x, d, label, color in ((0, zz, "ZZ", "#4C72B0"), (1, angle, "angle", "#DD8452")):
        vals = d["balanced_accuracy"].to_numpy()
        ax.scatter(np.full(len(vals), x), vals, color=color, s=28, zorder=2)
        ax.errorbar(x + 0.12, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color="black",
                    capsize=5, zorder=3)
        ax.annotate(f"{100 * vals.mean():.1f}%", (x + 0.17, vals.mean()), va="center",
                    fontsize=9)
    ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.text(1.32, 0.5, "chance (50%)", va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xticks(xs, ["ZZ encoding\n(main run, exams 0-4)", "angle encoding\n(exp 3)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Balanced accuracy (test patients)")
    p = table.loc[table.metric == "balanced_accuracy", "p_value"].item()
    ax.set_title(f"Exp 3: VQC encoding, same 5 exams (p = {p:.3f})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")


# --------------------------------------------------------------------------- #
# experiment 1: number of columns (= qubits), ZZ encoding
# --------------------------------------------------------------------------- #
def exp1_name(q: int) -> str:
    return f"exp1_vqc_q{q}_5ex"


def run_exp1(q: int, quick: bool) -> None:
    if q not in (4, 6):
        sys.exit("experiment 1 runs n_features=4 or 6 (8 is the main-run baseline).")
    _run(
        exp1_name(q),
        lambda: make_model(encoding="zz"),
        quick=quick,
        command=f"python -m src.run_exp_d exp1 --q {q}",
        n_features=q,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        manifest={
            "experiment": "1 (columns/qubits)",
            "changed": f"columns 8 -> {q} (top-{q} by f_classif on study patients, per exam)",
            "same": "ZZ encoding, 2 layers, PARAM_GRID, first 5 exams, seed 42",
            "exams": "first 5 exams (splits 0-4)",
            "columns": str(q),
            "encoding": "zz",
            "layers": "2",
            "tuning": "4 combinations per exam (3-fold inner GridSearchCV)",
            "baseline": "first 5 rows of results/vqc_zz_25.csv (ZZ, 8 columns), in memory",
        },
    )


def exp1_live_dials(n_draws: int = 5, n_inputs: int = 20, tol: float = 1e-8) -> None:
    """Count dials whose gradient is non-zero (live) at 4, 6, 8 qubits, 2 layers.

    Random dials uniform in [0, 2 pi), random inputs uniform in [0, pi]; a dial is
    live if |d<Z0>/d dial| > tol for at least one input in at least one draw.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    from src.model_vqc import build_qnode

    rng = np.random.default_rng(42)
    for q in (4, 6, 8):
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=q)
        total = int(np.prod(shape))
        live = np.zeros(shape, dtype=bool)
        for enc in ("zz", "angle"):
            circuit = build_qnode(q, enc)
            live_enc = np.zeros(shape, dtype=bool)
            for _ in range(n_draws):
                w = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
                x = pnp.array(rng.uniform(0, np.pi, size=(n_inputs, q)), requires_grad=False)
                jac = qml.jacobian(lambda ww: circuit(ww, x))(w)  # (n_inputs, *shape)
                live_enc |= np.abs(np.asarray(jac)).max(axis=0) > tol
            live |= live_enc
            per_layer = [int(live_enc[layer].sum()) for layer in range(shape[0])]
            print(f"{q} qubits, {enc:5s}: live dials {int(live_enc.sum())} of {total} "
                  f"(layer 1: {per_layer[0]}, layer 2: {per_layer[1]})")


def report_exp1() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {4: pd.read_csv(EXP_DIR / f"{exp1_name(4)}.csv"),
            6: pd.read_csv(EXP_DIR / f"{exp1_name(6)}.csv"),
            8: _baseline_first5()}
    rows = []
    for q in (4, 6):
        for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
            res = compare_models(runs[q], runs[8], metric=metric)
            rows.append({
                "comparison": f"{q} vs 8 columns",
                "metric": metric,
                "mean_q": res["mean_a"],
                "std_q": float(runs[q][metric].std(ddof=1)),
                "mean_8": res["mean_b"],
                "std_8": float(runs[8][metric].std(ddof=1)),
                "mean_diff_q_minus_8": res["mean_diff"],
                "t_stat": res["t_stat"],
                "df": res["df"],
                "p_value": res["p_value"],
                "conclusion": res["conclusion"],
            })
    res46 = compare_models(runs[4], runs[6], metric="balanced_accuracy")
    rows.append({"comparison": "4 vs 6 columns", "metric": "balanced_accuracy",
                 "mean_q": res46["mean_a"], "std_q": float(runs[4]["balanced_accuracy"].std(ddof=1)),
                 "mean_8": res46["mean_b"], "std_8": float(runs[6]["balanced_accuracy"].std(ddof=1)),
                 "mean_diff_q_minus_8": res46["mean_diff"], "t_stat": res46["t_stat"],
                 "df": res46["df"], "p_value": res46["p_value"], "conclusion": res46["conclusion"]})
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp1_vqc_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))

    for q in (4, 6, 8):
        d = runs[q]
        print(f"\n{q} columns: time/exam {d.train_time_sec.mean():.0f} s +/- "
              f"{d.train_time_sec.std(ddof=1):.0f}")
        print(d[["split", "balanced_accuracy", "best_params", "selected_features"]]
              .round(4).to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "exp1_vqc.png"
    _refuse_if_exists(fig_path)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    qs = [4, 6, 8]
    for exam in range(EXPERIMENT_EVAL_SPLITS):
        ax.plot(qs, [runs[q]["balanced_accuracy"].iloc[exam] for q in qs],
                color="0.8", lw=1, zorder=1)
    for q in qs:
        vals = runs[q]["balanced_accuracy"].to_numpy()
        ax.scatter(np.full(len(vals), q), vals, color="#4C72B0", s=24, zorder=2)
        ax.errorbar(q + 0.25, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color="black",
                    capsize=5, zorder=3)
        ax.annotate(f"{100 * vals.mean():.1f}%", (q + 0.4, vals.mean()), va="center",
                    fontsize=9)
    ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.set_xticks(qs, ["4 columns\n(4 qubits)", "6 columns\n(6 qubits)",
                       "8 columns\n(main run)"])
    ax.set_xlim(3.3, 9.2)
    ax.set_ylabel("Balanced accuracy (test patients)")
    ax.set_title("Exp 1: ZZ VQC, number of columns, same 5 exams", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")


# --------------------------------------------------------------------------- #
# experiment 2: number of training patients (fixed 20% test set, 10 draws)
# --------------------------------------------------------------------------- #
EXP2_SIZES = (30, 60, 120, 200)
EXP2_TUNING_LOG = LOG_DIR / "exp2_vqc_tuning.txt"


def exp2_name(n: int) -> str:
    return f"exp2_vqc_n{n}_10draws"


def run_exp2(quick: bool) -> None:
    """Tune once on the 614-patient pool, then train every size with those settings."""
    from src.data import load_pima
    from src.evaluate import get_fixed_test_split, tune_hyperparameters

    if quick:
        # Quick check: no tuning, first settings of the grid, smallest size, 3 draws.
        fixed = {k: v[0] for k, v in PARAM_GRID.items()}
        _run(exp2_name(30), make_model, quick=True, command="", manifest={},
             train_size=30, fixed_params=fixed, n_draws=10)
        return

    for n in EXP2_SIZES:
        _refuse_if_exists(EXP_DIR / f"{exp2_name(n)}.csv")
    _refuse_if_exists(EXP2_TUNING_LOG)

    X, y = load_pima()
    y = np.asarray(y)
    pool_idx, test_idx = get_fixed_test_split(X, y)
    started = _now()
    t0 = dt.datetime.now()
    best = tune_hyperparameters(make_model, PARAM_GRID, X.iloc[pool_idx], y[pool_idx])
    tune_sec = (dt.datetime.now() - t0).total_seconds()
    msg = (f"Exp 2 tuning (once, on the training pool)\n"
           f"  Pool patients      : {len(pool_idx)} ; fixed test patients: {len(test_idx)}\n"
           f"  Grid               : {json.dumps(PARAM_GRID)}\n"
           f"  Winning settings   : {json.dumps(best, sort_keys=True)}\n"
           f"  Tuning time        : {tune_sec:.1f} s\n"
           f"  Started            : {started}\n  Finished           : {_now()}\n")
    print(msg, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXP2_TUNING_LOG.write_text(msg, encoding="utf-8")

    for n in EXP2_SIZES:
        _run(
            exp2_name(n),
            make_model,
            quick=False,
            command="python -m src.run_exp_d exp2",
            train_size=n,
            fixed_params=best,
            n_draws=10,
            manifest={
                "experiment": "2 (training patients)",
                "changed": f"training patients 614 -> {n} (10 random stratified draws)",
                "same": "ZZ encoding, 2 layers, 8 columns, settings fixed, seed 42",
                "exams": f"10 draws of {n}; fixed 20% test set ({len(test_idx)} patients)",
                "columns": "8",
                "encoding": "zz",
                "layers": "2",
                "tuning": f"tuned once on the {len(pool_idx)}-patient pool -> "
                          f"{json.dumps(best, sort_keys=True)}",
                "baseline": "none (sizes compared with each other; no compare_models)",
            },
        )


def report_exp2() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {n: pd.read_csv(EXP_DIR / f"{exp2_name(n)}.csv") for n in EXP2_SIZES}
    rows = []
    for n, d in runs.items():
        row = {"train_size": n, "n_draws": len(d), "n_test": int(d["n_test"].iloc[0])}
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", "train_time_sec"):
            row[f"{metric}_mean"] = float(d[metric].mean())
            row[f"{metric}_std"] = float(d[metric].std(ddof=1))
        row["balanced_accuracy_min"] = float(d["balanced_accuracy"].min())
        row["balanced_accuracy_max"] = float(d["balanced_accuracy"].max())
        rows.append(row)
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp2_vqc_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    for n, d in runs.items():
        print(f"\nn={n} balanced accuracy per draw:",
              d["balanced_accuracy"].round(3).tolist())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "exp2_vqc.png"
    _refuse_if_exists(fig_path)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    sizes = list(EXP2_SIZES)
    rng = np.random.default_rng(0)  # horizontal jitter for the dots only
    for n in sizes:
        vals = runs[n]["balanced_accuracy"].to_numpy()
        ax.scatter(n + rng.uniform(-4, 4, len(vals)), vals, color="#4C72B0", alpha=0.5,
                   s=18, zorder=2)
    means = [runs[n]["balanced_accuracy"].mean() for n in sizes]
    stds = [runs[n]["balanced_accuracy"].std(ddof=1) for n in sizes]
    ax.errorbar(sizes, means, yerr=stds, fmt="-o", color="black", capsize=5, zorder=3,
                label="mean +/- spread (10 draws)")
    for n, m in zip(sizes, means):
        ax.annotate(f"{100 * m:.1f}%", (n + 6, m), va="center", fontsize=9)
    ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.set_xticks(sizes)
    ax.set_xlabel("Training patients")
    ax.set_ylabel("Balanced accuracy (fixed 154-patient test set)")
    ax.set_title("Exp 2: ZZ VQC, number of training patients", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")


# --------------------------------------------------------------------------- #
# experiment 5 references: saved moons results for logreg, RBF SVM, angle kernel
# --------------------------------------------------------------------------- #
EXP5_REF_NAMES = ("exp5_logreg_moons", "exp5_rbf_svm_moons", "exp5_qkernel_angle_moons")


def run_exp5_refs() -> None:
    """Same data as C's runner (make_moons_dataset), first 5 exams, 2 columns.

    Each model's make_model and PARAM_GRID are imported unchanged. Files go to
    results/ next to C's results/exp5_qkernel_moons.csv; nothing is overwritten.
    """
    from src import model_logreg, model_qkernel, model_rbf_svm
    from src.experiment_qkernel import make_moons_dataset

    results_dir = ROOT / "results"
    for name in EXP5_REF_NAMES:
        _refuse_if_exists(results_dir / f"{name}.csv")

    X_moons, y_moons = make_moons_dataset()
    print(f"moons data: {X_moons.shape[0]} points, class counts {np.bincount(y_moons).tolist()}")
    runs = (
        ("exp5_logreg_moons", model_logreg.make_model, model_logreg.PARAM_GRID),
        ("exp5_rbf_svm_moons", model_rbf_svm.make_model, model_rbf_svm.PARAM_GRID),
        ("exp5_qkernel_angle_moons", lambda: model_qkernel.make_model(encoding="angle"),
         model_qkernel.PARAM_GRID),
    )
    for name, factory, grid in runs:  # one at a time
        summary = evaluate(
            name, factory, grid,
            n_features=2,
            X=X_moons, y=y_moons,
            zero_as_missing_cols=(),
            n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
            results_dir=results_dir,
        )
        print(format_summary(summary), flush=True)


# --------------------------------------------------------------------------- #
# experiment 5 for the VQC: same moons data and arguments as exp5-refs
# --------------------------------------------------------------------------- #
EXP5_VQC = {"zz": "exp5_vqc_zz_moons768_5ex", "angle": "exp5_vqc_angle_moons768_5ex"}
EXP5_SAVED = {  # label -> saved file (same data, same 5 exams)
    "Logistic regression": ROOT / "results" / "exp5_logreg_moons.csv",
    "RBF SVM": ROOT / "results" / "exp5_rbf_svm_moons.csv",
    "Kernel, angle": ROOT / "results" / "exp5_qkernel_angle_moons.csv",
    "Kernel, ZZ": ROOT / "results" / "exp5_qkernel_moons.csv",
}


def run_exp5_vqc(encoding: str, quick: bool) -> None:
    from src.experiment_qkernel import make_moons_dataset

    if encoding not in EXP5_VQC:
        sys.exit("exp5-vqc needs --enc zz or --enc angle")
    X_moons, y_moons = make_moons_dataset()
    _run(
        EXP5_VQC[encoding],
        lambda: make_model(encoding=encoding),
        quick=quick,
        command=f"python -m src.run_exp_d exp5-vqc --enc {encoding}",
        n_features=2,
        X=X_moons,
        y=y_moons,
        zero_as_missing_cols=(),
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        manifest={
            "experiment": "5 (moons control, ARTIFICIAL data)",
            "changed": "data: Pima -> make_moons(n_samples=768, noise=0.2, random_state=42) "
                       "from make_moons_dataset(); 2 columns = 2 qubits",
            "same": f"{encoding} encoding, 2 layers, PARAM_GRID, first 5 exams, seed 42, "
                    "same evaluate() arguments as exp5-refs",
            "exams": "first 5 exams (splits 0-4) of the moons data",
            "columns": "2 (moons x0, x1)",
            "encoding": encoding,
            "layers": "2",
            "tuning": "4 combinations per exam (3-fold inner GridSearchCV)",
            "baseline": "saved results/exp5_*_moons.csv (logreg, RBF SVM, kernel angle, kernel ZZ)",
        },
    )


def report_exp5_vqc() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {label: pd.read_csv(path) for label, path in EXP5_SAVED.items()}
    runs["VQC, angle"] = pd.read_csv(EXP_DIR / f"{EXP5_VQC['angle']}.csv")
    runs["VQC, ZZ"] = pd.read_csv(EXP_DIR / f"{EXP5_VQC['zz']}.csv")
    logreg = runs["Logistic regression"]

    summary = []
    for label, d in runs.items():
        row = {
            "model": label,
            "balanced_accuracy_mean": float(d["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(d["balanced_accuracy"].std(ddof=1)),
            "accuracy_mean": float(d["accuracy"].mean()),
            "train_time_sec_mean": float(d["train_time_sec"].mean()),
            "gap_to_logreg": float(d["balanced_accuracy"].mean() - logreg["balanced_accuracy"].mean()),
            "above_50_floor": float(d["balanced_accuracy"].mean() - 0.5),
            "best_params": "; ".join(f"{k} x{v}" for k, v in d["best_params"].value_counts().items()),
        }
        summary.append(row)
    print(pd.DataFrame(summary).round(4).to_string(index=False))

    pairs = [
        ("VQC, ZZ", "Logistic regression"),
        ("VQC, angle", "Logistic regression"),
        ("VQC, ZZ", "Kernel, ZZ"),
        ("VQC, angle", "Kernel, angle"),
        ("VQC, ZZ", "VQC, angle"),
    ]
    rows = []
    for a, b in pairs:
        res = compare_models(runs[a], runs[b], metric="balanced_accuracy")
        rows.append({"model_a": a, "model_b": b, "mean_a": res["mean_a"], "mean_b": res["mean_b"],
                     "mean_diff_a_minus_b": res["mean_diff"], "t_stat": res["t_stat"],
                     "df": res["df"], "p_value": res["p_value"], "conclusion": res["conclusion"]})
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp5_vqc_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    print()
    with pd.option_context("display.width", 220):
        print(table.round(4).to_string(index=False))
    for label in ("VQC, angle", "VQC, ZZ"):
        d = runs[label]
        print(f"\n{label} per exam:")
        print(d[["split", "balanced_accuracy", "train_time_sec", "best_params"]].round(4)
              .to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "exp5_vqc.png"
    _refuse_if_exists(fig_path)
    order = ["Logistic regression", "RBF SVM", "Kernel, angle", "Kernel, ZZ",
             "VQC, angle", "VQC, ZZ"]
    colors = ["#8C8C8C", "#8C8C8C", "#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for x, (label, color) in enumerate(zip(order, colors)):
        vals = runs[label]["balanced_accuracy"].to_numpy()
        ax.bar(x, vals.mean(), color=color, alpha=0.8, width=0.6)
        ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), color="black", capsize=5)
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.12, 0.12, len(vals)), vals,
                   color="black", s=10, zorder=3)
        ax.text(x, vals.mean() + vals.std(ddof=1) + 0.012, f"{100 * vals.mean():.1f}%",
                ha="center", fontsize=8.5)
    ax.axhline(0.5, color="red", ls=":", lw=1.2, label="always-guess floor (50%)")
    ax.set_xticks(range(len(order)), [o.replace(", ", "\n") for o in order], fontsize=8.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Balanced accuracy (test points)")
    ax.set_title("Exp 5: moons control (ARTIFICIAL data, 768 points, 2 qubits), same 5 exams",
                 fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")


# --------------------------------------------------------------------------- #
# experiment 6 (added after seeing results): circuit structure vs training, moons
# --------------------------------------------------------------------------- #
EXP6_LONG_GRID = {"lr": [0.02, 0.05], "epochs": [100, 200]}
EXP6 = {
    # run -> (file name, model factory, settings grid, what changed vs exp5 ZZ VQC)
    "A": ("exp6_vqc_zz_rot_moons768_5ex",
          lambda: make_model(encoding="zz", final_rotation=True), None,
          "final_rotation=True (one extra trainable Rot on wire 0); usual PARAM_GRID"),
    "B": ("exp6_vqc_zz_long_moons768_5ex",
          lambda: make_model(encoding="zz"), EXP6_LONG_GRID,
          "longer training: epochs {100, 200} x lr {0.02, 0.05}; circuit unchanged"),
}


def run_exp6(run: str, quick: bool) -> None:
    """Same evaluate() arguments as exp5-vqc; one change per run (A or B)."""
    from src.experiment_qkernel import make_moons_dataset

    if run not in EXP6:
        sys.exit("exp6 needs --run A or --run B")
    name, factory, grid, changed = EXP6[run]
    X_moons, y_moons = make_moons_dataset()
    print(f"Exp 6 run {run}: {changed}", flush=True)
    _run(
        name,
        factory,
        quick=quick,
        command=f"python -m src.run_exp_d exp6 --run {run}",
        manifest=None,  # MANIFEST_D.md not edited for exp 6 (not in the agreed file list)
        param_grid=grid,
        n_features=2,
        X=X_moons,
        y=y_moons,
        zero_as_missing_cols=(),
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
    )


def _exam0_train_score(factory, params: dict) -> tuple[float, float]:
    """Refit exam 0 with its winning settings; return (train, test) balanced accuracy.

    Uses the same pipeline builder and splits as evaluate(); seed 42 makes the refit
    identical to the model evaluate() scored, which the test score lets us check.
    """
    from sklearn.metrics import balanced_accuracy_score

    from src.evaluate import _build_pipeline, get_splits
    from src.experiment_qkernel import make_moons_dataset

    X, y = make_moons_dataset()
    y = np.asarray(y)
    train_idx, test_idx = get_splits(X, y)[0]
    pipe = _build_pipeline(factory, 2, ())
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    pipe.fit(X[train_idx], y[train_idx])
    return (float(balanced_accuracy_score(y[train_idx], pipe.predict(X[train_idx]))),
            float(balanced_accuracy_score(y[test_idx], pipe.predict(X[test_idx]))))


def report_exp6() -> None:
    zz5 = pd.read_csv(EXP_DIR / f"{EXP5_VQC['zz']}.csv")
    logreg = pd.read_csv(EXP5_SAVED["Logistic regression"])
    runs = {r: pd.read_csv(EXP_DIR / f"{EXP6[r][0]}.csv") for r in EXP6}

    print("Mean +/- spread (balanced accuracy, 5 exams):")
    for label, d in (("exp5 ZZ VQC", zz5), ("logistic regression", logreg),
                     ("exp6 A (final Rot)", runs["A"]), ("exp6 B (long training)", runs["B"])):
        ba = d["balanced_accuracy"]
        print(f"  {label:24s} {ba.mean():.4f} +/- {ba.std(ddof=1):.4f}   "
              f"time/exam {d['train_time_sec'].mean():.1f} s")

    rows = []
    for r, d in runs.items():
        for other_label, other in (("exp5 ZZ VQC", zz5), ("logistic regression", logreg)):
            res = compare_models(d, other, metric="balanced_accuracy")
            rows.append({"run": r, "model_a": EXP6[r][0], "model_b": other_label,
                         "mean_a": res["mean_a"], "std_a": float(d["balanced_accuracy"].std(ddof=1)),
                         "mean_b": res["mean_b"],
                         "std_b": float(other["balanced_accuracy"].std(ddof=1)),
                         "mean_diff_a_minus_b": res["mean_diff"], "t_stat": res["t_stat"],
                         "df": res["df"], "p_value": res["p_value"],
                         "conclusion": res["conclusion"]})
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp6_vqc_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    print()
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(table.drop(columns=["model_a"]).round(4).to_string(index=False))

    print("\nPer exam (balanced accuracy, winning settings):")
    for label, d in (("exp5 ZZ VQC", zz5), ("A", runs["A"]), ("B", runs["B"])):
        print(f"  {label}: {d['balanced_accuracy'].round(4).tolist()}  {d['best_params'].tolist()}")

    print("\nExam 0, balanced accuracy on its own 614 training points (refit, winning settings):")
    factories = {"exp5 ZZ VQC": (lambda: make_model(encoding="zz"), zz5),
                 "A": (EXP6["A"][1], runs["A"]), "B": (EXP6["B"][1], runs["B"])}
    for label, (factory, d) in factories.items():
        params = json.loads(d["best_params"].iloc[0])
        train_ba, test_ba = _exam0_train_score(factory, params)
        saved = float(d["balanced_accuracy"].iloc[0])
        print(f"  {label:12s} settings {params}: train {train_ba:.4f} | test {test_ba:.4f} "
              f"(saved test {saved:.4f}, match: {abs(test_ba - saved) < 1e-12})")


# --------------------------------------------------------------------------- #
# experiment 6, run C (added after seeing results): chain wiring on Pima
# --------------------------------------------------------------------------- #
EXP6C_NAME = "exp6_pima_ranges11"
MAIN_LOGREG = ROOT / "results" / "logreg.csv"


def _exp6c_factory():
    return make_model(encoding="zz", cable_ranges=[1, 1])


def run_exp6c(quick: bool) -> None:
    """Pima, ZZ, 8 columns, 2 layers, PARAM_GRID, first 5 exams; only cable_ranges changes."""
    _run(
        EXP6C_NAME,
        _exp6c_factory,
        quick=quick,
        command="python -m src.run_exp_d exp6c",
        n_features=8,
        n_splits_to_run=EXPERIMENT_EVAL_SPLITS,
        manifest={
            "experiment": "6 run C (chain wiring, Pima; added after seeing results)",
            "changed": "cable_ranges=[1, 1]: StronglyEntanglingLayers ranges (1, 2) -> (1, 1) "
                       "(layer 2 CNOTs i -> i+1 instead of i -> i+2); live dials 28 -> 38 of 48",
            "same": "ZZ encoding, 2 layers, PARAM_GRID, 8 columns, read qubit 0, first 5 exams, "
                    "seed 42, evaluate() defaults as the main run",
            "exams": "first 5 exams (splits 0-4)",
            "columns": "8",
            "encoding": "zz",
            "layers": "2 (ranges 1, 1)",
            "tuning": "4 combinations per exam (3-fold inner GridSearchCV)",
            "baseline": "first 5 rows of results/vqc_zz_25.csv (ZZ) and of results/logreg.csv, "
                        "in memory",
        },
    )


def report_exp6c() -> None:
    from sklearn.metrics import balanced_accuracy_score

    from src.data import load_pima
    from src.evaluate import ZERO_AS_MISSING_COLS, _build_pipeline, get_splits

    c = pd.read_csv(EXP_DIR / f"{EXP6C_NAME}.csv")
    zz = _baseline_first5()
    lr = pd.read_csv(MAIN_LOGREG).head(EXPERIMENT_EVAL_SPLITS)
    for label, d in (("main ZZ VQC", zz), ("logistic regression", lr)):
        same = (d["split"].tolist() == c["split"].tolist()
                and d["n_train"].tolist() == c["n_train"].tolist()
                and d["n_test"].tolist() == c["n_test"].tolist())
        print(f"split ids / sizes match {label}: {same}  {d['split'].tolist()}")

    print("\nMean +/- spread (balanced accuracy, 5 exams):")
    for label, d in (("main ZZ VQC (first 5)", zz), ("logistic regression (first 5)", lr),
                     ("run C (ranges 1,1)", c)):
        ba = d["balanced_accuracy"]
        print(f"  {label:30s} {ba.mean():.4f} +/- {ba.std(ddof=1):.4f}   "
              f"time/exam {d['train_time_sec'].mean():.1f} s +/- {d['train_time_sec'].std(ddof=1):.1f}")

    rows = []
    for label, other in (("main ZZ VQC (first 5)", zz), ("logistic regression (first 5)", lr)):
        res = compare_models(c, other, metric="balanced_accuracy")
        rows.append({"model_a": EXP6C_NAME, "model_b": label, "mean_a": res["mean_a"],
                     "std_a": float(c["balanced_accuracy"].std(ddof=1)), "mean_b": res["mean_b"],
                     "std_b": float(other["balanced_accuracy"].std(ddof=1)),
                     "mean_diff_a_minus_b": res["mean_diff"], "t_stat": res["t_stat"],
                     "df": res["df"], "p_value": res["p_value"], "conclusion": res["conclusion"]})
    table = pd.DataFrame(rows)
    out = EXP_DIR / "exp6_pima_ranges11_comparison.csv"
    _refuse_if_exists(out)
    table.to_csv(out, index=False)
    print()
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table.drop(columns=["model_a"]).round(4).to_string(index=False))

    print("\nPer exam (balanced accuracy, winning settings, time):")
    print(pd.DataFrame({"exam": c["split"], "main_zz": zz["balanced_accuracy"].round(4),
                        "run_C": c["balanced_accuracy"].round(4), "C_settings": c["best_params"],
                        "C_time_s": c["train_time_sec"].round(0)}).to_string(index=False))

    # Exam 0: rebuild the winning model, check the test score, score its training patients.
    X, y = load_pima()
    y = np.asarray(y)
    train_idx, test_idx = get_splits(X, y)[0]
    params = json.loads(c["best_params"].iloc[0])
    pipe = _build_pipeline(_exp6c_factory, 8, ZERO_AS_MISSING_COLS)
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    pipe.fit(X.iloc[train_idx], y[train_idx])
    train_ba = balanced_accuracy_score(y[train_idx], pipe.predict(X.iloc[train_idx]))
    test_ba = balanced_accuracy_score(y[test_idx], pipe.predict(X.iloc[test_idx]))
    saved = float(c["balanced_accuracy"].iloc[0])
    print(f"\nExam 0 run C, settings {params}: training patients {train_ba:.4f} | "
          f"test {test_ba:.4f} (saved {saved:.4f}, match: {abs(test_ba - saved) < 1e-12})")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["exp3", "exp3-report", "exp1", "exp1-dials",
                                         "exp1-report", "exp2", "exp2-report", "exp5-refs",
                                         "exp5-vqc", "exp5-vqc-report", "exp6",
                                         "exp6-report", "exp6c", "exp6c-report"])
    parser.add_argument("--enc", choices=["zz", "angle"], help="encoding for exp5-vqc")
    parser.add_argument("--run", choices=sorted(EXP6), help="exp6 run: A or B")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--q", type=int, help="number of columns for exp1 (4 or 6)")
    args = parser.parse_args()
    if args.task == "exp3":
        run_exp3(args.quick)
    elif args.task == "exp3-report":
        report_exp3()
    elif args.task == "exp1":
        run_exp1(args.q, args.quick)
    elif args.task == "exp1-dials":
        exp1_live_dials()
    elif args.task == "exp1-report":
        report_exp1()
    elif args.task == "exp2":
        run_exp2(args.quick)
    elif args.task == "exp2-report":
        report_exp2()
    elif args.task == "exp5-refs":
        run_exp5_refs()
    elif args.task == "exp5-vqc":
        run_exp5_vqc(args.enc, args.quick)
    elif args.task == "exp5-vqc-report":
        report_exp5_vqc()
    elif args.task == "exp6":
        run_exp6(args.run, args.quick)
    elif args.task == "exp6-report":
        report_exp6()
    elif args.task == "exp6c":
        run_exp6c(args.quick)
    elif args.task == "exp6c-report":
        report_exp6c()


if __name__ == "__main__":
    main()

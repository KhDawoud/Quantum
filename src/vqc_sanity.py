"""Development sanity checks + timing for the VQC (NOT competition results).

Run one section at a time:
    python -m src.vqc_sanity overfit   # can it memorise 20 patients?
    python -m src.vqc_sanity easy      # 2 easy columns: VQC vs logistic regression
    python -m src.vqc_sanity loss      # does the loss go down on the real 8-qubit task?
    python -m src.vqc_sanity timing    # time of one fit + full-run arithmetic

The few splits used here are quick dev checks only. Final numbers come from
src/evaluate.py (the single shared harness), never from this file.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.data import PimaPreprocessor, load_pima
from src.model_vqc import PARAM_GRID, make_model

SEED = 42


def check_overfit() -> None:
    """Can the model memorise 20 training patients? (No test data involved.)

    Finding (see notes/DESIGN_D.md): with 2 layers it plateaus at ~75-90% in a local
    minimum, and different seeds land in different minima. 4 layers gets ~95%.
    Unregularised logistic regression reaches 100%, so we report this honestly.
    """
    from sklearn.linear_model import LogisticRegression

    X, y = load_pima()
    rng = np.random.default_rng(SEED)
    idx = np.r_[rng.choice(np.where(y == 0)[0], 10, replace=False),
                rng.choice(np.where(y == 1)[0], 10, replace=False)]
    Xs = PimaPreprocessor().fit_transform(X.iloc[idx], y[idx])
    ref = LogisticRegression(C=1e6, max_iter=10000).fit(Xs, y[idx])
    print(f"[overfit] reference: logistic regression, no regularisation: "
          f"{(ref.predict(Xs) == y[idx]).mean():.0%}")
    for enc in ("angle", "zz"):
        for layers in (2, 4):
            m = make_model(encoding=enc, n_layers=layers, epochs=250, lr=0.1, batch_size=20)
            t = time.perf_counter()
            m.fit(Xs, y[idx])
            acc = (m.predict(Xs) == y[idx]).mean()
            print(f"[overfit] {enc:5s} {layers} layers: train accuracy on the 20 patients = {acc:.0%}, "
                  f"loss {m.loss_history_[0]:.3f} -> {m.loss_history_[-1]:.3f} "
                  f"({time.perf_counter() - t:.0f}s)")


def check_easy_columns() -> None:
    """Top-2 columns (ranked on training rows only) -> VQC should be near logistic regression."""
    X, y = load_pima()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = {"logreg": [], "vqc-angle": [], "vqc-zz": []}
    for tr, te in cv.split(X, y):
        prep = PimaPreprocessor(n_features=2).fit(X.iloc[tr], y[tr])
        Xtr, Xte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
        lr = LogisticRegression(class_weight="balanced", random_state=SEED).fit(Xtr, y[tr])
        scores["logreg"].append(balanced_accuracy_score(y[te], lr.predict(Xte)))
        for enc in ("angle", "zz"):
            m = make_model(encoding=enc, epochs=40, lr=0.05).fit(Xtr, y[tr])
            scores[f"vqc-{enc}"].append(balanced_accuracy_score(y[te], m.predict(Xte)))
    print(f"[easy] columns used (last fold): {prep.selected_feature_names_}")
    for k, v in scores.items():
        print(f"[easy] {k:10s} balanced accuracy over 5 folds: {np.mean(v):.3f} +- {np.std(v):.3f}")


def check_loss_8_qubits() -> None:
    """Loss should fall during training on the real 8-column task."""
    X, y = load_pima()
    Xs = PimaPreprocessor().fit_transform(X, y)  # dev check on training loss only
    for enc in ("angle", "zz"):
        m = make_model(encoding=enc, epochs=20, lr=0.05).fit(Xs, y)
        h = m.loss_history_
        print(f"[loss] {enc:5s} 8 qubits, 20 epochs: first={h[0]:.4f}  last={h[-1]:.4f}  "
              f"min={min(h):.4f}  (always-0.5 loss would be 0.6931)")


def _time_fit(n_rows: int, epochs: int, encoding: str, n_features: int = 8) -> float:
    X, y = load_pima()
    Xs = PimaPreprocessor().fit_transform(X, y)[:, :n_features]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), n_rows, replace=False)
    t = time.perf_counter()
    make_model(encoding=encoding, epochs=epochs).fit(Xs[idx], y[idx])
    return time.perf_counter() - t


def timing(batch_note: bool = True) -> None:
    """Measure fits at the sizes the real pipeline will use, then do the arithmetic.

    Outer 5-fold train = 614 rows; inner 3-fold train = 2/3 of 614 ~ 409 rows.
    """
    print(f"[timing] cpu cores visible: {os.cpu_count()}")
    n_outer, n_inner = 614, 409
    epochs_grid = sorted(set(PARAM_GRID["epochs"]))
    for enc in ("zz", "angle"):
        t = {(n, e): _time_fit(n, e, enc) for n in (n_inner, n_outer) for e in epochs_grid}
        for (n, e), v in t.items():
            print(f"[timing] {enc:5s} 8 qubits, {n} rows, {e} epochs: {v:6.1f} s")
        n_lr = len(PARAM_GRID["lr"])
        inner_total = 3 * n_lr * sum(t[(n_inner, e)] for e in epochs_grid)  # lr does not change time
        for label, refit in (("best=fewest epochs", t[(n_outer, epochs_grid[0])]),
                             ("best=most epochs", t[(n_outer, epochs_grid[-1])])):
            per_split = inner_total + refit
            print(f"[timing] {enc:5s} per outer split: inner {inner_total:6.0f}s + refit {refit:5.0f}s "
                  f"({label}) = {per_split / 60:5.1f} min;  x25 splits = {25 * per_split / 3600:5.2f} h;  "
                  f"x5 splits = {5 * per_split / 3600:5.2f} h")
        simple = 325 * t[(n_inner, epochs_grid[-1])]
        print(f"[timing] {enc:5s} simple rule 325 x (one 409-row, {epochs_grid[-1]}-epoch fit) = "
              f"{simple / 3600:5.2f} h")


if __name__ == "__main__":
    sections = {"overfit": check_overfit, "easy": check_easy_columns,
                "loss": check_loss_8_qubits, "timing": timing}
    wanted = sys.argv[1:] or list(sections)
    for name in wanted:
        sections[name]()

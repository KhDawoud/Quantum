"""Run Model B through the team's shared evaluation and output formatter."""

from __future__ import annotations

import argparse

from src.evaluate import evaluate, format_summary
from src.model_rbf_svm import PARAM_GRID, make_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use the smaller dev grid")
    args = parser.parse_args()

    # Keep quick results separate so they cannot replace the main comparison CSV.
    name = "rbf_svm_quick" if args.quick else "rbf_svm"
    summary = evaluate(
        name=name,
        make_model=make_model,
        param_grid=PARAM_GRID,
        n_features=8,
        quick=args.quick,
        # Explicit Nov 26 team policy, including on older shared-harness checkouts.
        n_splits_to_run=5 if args.quick else 25,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()

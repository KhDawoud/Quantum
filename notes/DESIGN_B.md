# DESIGN_B: design choices for Person B

Scope: Model B, its runner and tests, and `src/experiment_rbf.py`.
This delivery implements Model B, its shared evaluation, and its RBF-only
training-size experiment. The combined all-model experiment and slides are separate.

## 1. What the model does

An RBF SVM compares two patients using `exp(-gamma * squared_distance)`.
Similar patients have a kernel value near one; distant patients have a value
near zero. The fitted support vectors and their weights define a nonlinear
decision boundary. Scikit-learn's SVC handles optimization and prediction.

| Choice | Why |
|---|---|
| `SVC(kernel="rbf")` | Required classical nonlinear baseline; native scikit-learn cloning and pipeline support. |
| `C=[1, 10]` | Two penalties for margin violations. Larger C tolerates fewer violations, with a risk of fitting noise. |
| `gamma=[0.01, 0.1, 0.5, 1.0]` | Four distance scales for the common [0, pi] inputs. Larger gamma makes similarity more local. These are the requested settings, fixed before outer-test results were inspected. |
| Eight combinations | Uses the agreed tuning budget without adding a class-weight dimension. `class_weight="balanced"` is fixed; SVC computes weights from the labels received in each fit as n_samples / (2 x class count). |
| Default probability calibration (off) | The evaluator uses labels. Leave the deprecated `probability` argument unspecified in scikit-learn 1.9.1 to avoid warnings and extra calibration fits. |
| `random_state=RANDOM_SEED` | Matches the project seed. With probability calibration off, SVC's seed is unused; the data splitters still explicitly use seed 42. |
| Existing shared preprocessor | Keeps clinical cleaning, scaling, and feature selection identical to the other models, learned within every inner and outer training fold. |

RBF gamma multiplies squared distances; it is not a quantum gate-angle scale.
The quantum gamma constraint is therefore not a mathematical requirement for
this model. The requested numerical grid nevertheless ends at one. We do not
claim this small grid is globally optimal or revise it using outer-test scores.

## 2. GitHub contribution alignment (2026-11-26)

Fetched origin and inspected main `5bd1d52` and Model C's `feat/qkernel`
`cb90428`. C's two commits add its model, tests, and its encoding experiment
in `src/experiments.py`; that fetched branch has no committed CSVs, figures,
or Member C notes. Its shared evaluator is still a placeholder, so it is not
used as a replacement for main's evaluator.

B follows the same model + tests + owned-experiment pattern, with generated
results and provenance notes included. The experiment implementation lives in
`src/experiment_rbf.py`, exposed through `src/experiments.py`, to minimize edits
to the shared file. Only that import should be added if C's experiment is merged
before B. Do not replace teammates' shared files with this older checkout.

The contribution was developed from `668b608` and prepared for integration
on current main. The runner explicitly requests
25 main splits / 5 quick splits, matching current main. Main's changes to the
shared evaluator only change default split policy; preprocessing and scoring
are unchanged. Compatibility is tested in a temporary export of current main.
The shared data/evaluator files and other models are not changed by B.

## 3. Current balanced-weight benchmark

The current CSVs replace the historical unweighted five-split outputs.
The full run uses all eight settings, all eight features, 25 outer splits,
and three-fold inner balanced-accuracy tuning. Quick mode is development only.
Timing includes tuning and refit; machine-dependent timings are not speed rankings.

| Metric | Mean ± sample standard deviation |
|---|---|
| Accuracy | 74.76% ± 3.82% |
| Balanced accuracy | 74.99% ± 3.70% |
| Macro F1 | 73.47% ± 3.80% |
| Training time per split | 0.2182 s ± 0.0037 s |

The majority-guess accuracy floor on the full dataset is 65.10%. No superiority
or significance claim against the quantum models is made from these results.

## 4. Training-size experiment (RBF contribution only)

Tune once on the 614-patient pool outside the fixed stratified 154-patient test
set (seed 42). Reuse those settings for 10 draws each at sizes 30, 60, 120, 200.
Each draw fits its own preprocessor on its training subset, through the shared
evaluator. Seeds are derived from 42 by the existing shared splitting function.
Tuning uses more patients than each small training set; this is the explicitly
agreed tuning shortcut, not a claim of learning exclusively from 30 patients.

Outputs: `results/exp2_trainsize_rbf.csv` (40 per-draw rows) and
`figures/exp2_trainsize_rbf.png`. This does not replace the planned all-model
`exp2_trainsize.csv`; it cannot establish whether quantum needs fewer patients.

| Training patients | Accuracy mean ± sample SD |
|---|---|
| 30 | 70.71% ± 2.11% |
| 60 | 71.17% ± 2.32% |
| 120 | 71.30% ± 1.05% |
| 200 | 72.60% ± 1.29% |

Mean accuracy rises from 70.71% at 30 patients to 72.60% at 200 patients.
The smaller-size error bars overlap; these descriptive standard deviations are
not confidence intervals or a significance test because the draws share test
patients. More training data may stabilize the boundary, but that explanation
is a hypothesis, and no comparison with another model is available here.

The chart's 64.94% baseline is the actual majority class fraction in the fixed
154-patient test set; the full-dataset baseline remains 65.10%.

## 5. Validation and limits

Tests cover the RBF calculation, nonlinear learning, balanced weights from
fit-time labels, estimator cloning, reduced feature counts, shared CSV format,
and tuning once without accessing the experiment's test patients. Chart checked
visually. See AI_USE_B.md for the final test counts.

Python 3.14.8, scikit-learn 1.9.1, SciPy 1.18.1, pytest 9.2.0,
Matplotlib 3.11.2, NumPy 2.5.4, pandas 3.0.6. The exact installed package
snapshot is `notes/requirements-tested-B.txt`. NumPy and pandas differ from the
team pins; a clean installation of the complete requirements is not verified.
Quantum training/tests were not run for this contribution.

B and C use fixed balanced weights, and D uses weighted loss. Logistic
regression on current main still tunes `class_weight` between None and balanced;
it is therefore inaccurate to say all four models use fixed balanced weights.
The source of SVC usage is cited in the model module. The all-model experiment,
clean-install script/check, slides, and main comparison statistics remain wider
team/Member B deliverables outside this model-specific package.

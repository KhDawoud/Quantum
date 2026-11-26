# DESIGN_D: why the VQC is built this way (Member D)

File: `src/model_vqc.py`. Checks: `tests/test_vqc.py`, `src/vqc_sanity.py`.

## What the model is, in plain language
1. **Encode**: each patient is 8 numbers in [0, pi]. They become the angles of gates on 8 qubits.
2. **Trainable dials**: `StronglyEntanglingLayers` = per layer, one 3-angle rotation (`Rot`) on every qubit, then CNOT gates linking neighbouring qubits. 2 layers x 8 qubits x 3 angles = **48 dials**. These are the model's only learned parameters (like weights in logistic regression).
3. **Measure**: read `<Z>` on qubit 0, a number z in [-1, 1].
4. **Probability**: P(diabetes) = (1 - z) / 2. z = +1 (qubit 0 is `|0>`) means class 0; z = -1 means class 1.
5. **Training**: Adam repeatedly nudges the 48 dials to reduce class-weighted binary cross-entropy. Gradients come from backprop through the simulator.

## Choices and why
| Choice | Why |
|---|---|
| `default.qubit`, qubits = columns | Hard rule (<= 10 qubits, simulator). |
| Angle encoding = `RX(x_i)` per qubit | Simplest possible; x in [0, pi] never wraps. **Check it matches teammate's qkernel "angle" encoding** (exp 3 compares them). |
| ZZ map, 2 reps, H / RZ(x_i) / neighbour `ZZ(x_i*x_(i+1))` | As specified in PROJECT_CONTEXT.md. Implemented with `IsingZZ`, exactly equal to CNOT-RZ-CNOT (unit test), but ~3x fewer simulated gates. |
| Readout `<Z_0>` -> (1-z)/2 | The literal "expectation to probability"; no extra classical dials, so the model stays purely quantum. |
| Class weights n/(2 n_c) | ~2/3 of patients are negative; without weights the model drifts to "always negative" (65.1% floor). Same as sklearn `'balanced'`. |
| Adam, mini-batches of 32, fixed epochs, seed 42 | Spec. Seed drives init (N(0, 0.1)) and shuffling; same seed gives identical weights (tested). |
| `diff_method='backprop'` + whole batch per call | PennyLane parameter broadcasting runs all 32 patients in one circuit execution. Parameter-shift would need ~2 x 48 extra circuits per step. |
| `loss_history_` = mean batch loss per epoch | No extra forward pass; slightly lags the true end-of-epoch loss. |
| Rejects inputs outside [0, pi] | Enforces the angle-bounds rule inside the model. |
| Circuit rebuilt inside `predict_proba` | Keeps fitted models picklable for parallel runs. |

All models use balanced class weighting, fixed and not tuned, because the classes are uneven and we evaluate on balanced accuracy.
(VQC: weights are `n / (2 x class count)`, computed inside `fit()` from the labels it receives, about 0.77 healthy and 1.43 diabetic on the full data; applied per patient as `mean(weight x cross-entropy)` over each mini-batch; not used in `predict`; not in `PARAM_GRID`.)

## Docs vs. the prompt (PennyLane 0.45.1)
Everything in the prompt matches the installed version. Notes: the 0.45 docs write the alias `qp` (old `qml` still works, no warnings); no torch/jax installed, so backprop uses autograd (`pennylane.numpy`), which is fine; PennyLane's Adam default `beta2` is 0.99 (not 0.999), left as is.

## Sanity-check results (dev checks only, not final numbers)
- **Gradients**: autograd matches finite differences (atol 1e-6), both encodings.
- **Loss goes down**: 8 qubits, 20 epochs, whole dataset: angle 0.662 -> 0.575, zz 0.668 -> 0.624 (always-0.5 would be 0.693). Passes, but ZZ learns more slowly.
- **Two easy columns (plas, mass; 5 dev folds, balanced acc.)**: logreg 0.733 +- 0.028, VQC-angle 0.731 +- 0.021, VQC-zz 0.721 +- 0.013. Passes: VQC is close to logreg.
- **Overfit 20 patients: only partly passes.** Logreg (no regularisation) 100%. VQC 2 layers: angle 75%, zz 80% (loss plateaus; different seeds give 75-90%, i.e. different local minima). 4 layers: 95% for both. Longer training and larger learning rate do not help at 2 layers. Interpretation: with 48 dials and one readout qubit the model is not flexible enough to memorise 20 noisy patients, and the loss landscape has several minima. Not a bug (gradients verified, loss falls, depth helps). Worth a sentence in the report; exp 4 (depth) tests it.

## Day-1 timing decision (8 qubits, 8 CPU cores visible, one process)
Fit times (zz): 409 rows (inner fold): 45 s @20 epochs, 87 s @40; 614 rows (refit): 69 s @20, 179 s @40. Angle is ~10-35% faster.
Per outer split: 3 inner folds x 4 combos = 792 s, plus refit 69-179 s = **14-16 min**.
- **25 splits: ~6.0-6.7 h (zz), ~5.1-5.5 h (angle). Simple rule 325 x 87 s = 7.8 h.** Both exceed 3 h.
- **5 splits: ~1.2-1.35 h (zz).**

~~Decision: apply the PROJECT_CONTEXT fallback (first 5 splits for everything).~~ Superseded, see below.

**Updated decision (Nov 26, team):** the estimate above was too pessimistic. The real 5-split VQC-ZZ run (`results/vqc_zz.csv`) took about **7 minutes per split** (370-486 s, 4 combinations), so 25 splits take about **3 hours**. Main table = **25 splits for all models** (the `evaluate()` default). Experiments = the **first 5 splits for all models**, passed explicitly with `n_splits_to_run=EXPERIMENT_EVAL_SPLITS`. Never mix split counts across models within one table or experiment. Use `fixed_params` (tune once) in experiments where possible.

## Known caveats
- ZZ pair angle x_i*x_(i+1) can reach pi^2 (> pi) as specified; the pair term can wrap. Flag to the team.
- Timings are one noisy measurement on a laptop; re-measure before the final run.

Prediction for experiment 6, written before running: the extra rotation will improve fitting on the 20-patient check, but will change the 5-split test scores by less than the spread between splits, because on this data the overlap between the groups limits the score, not the number of dials.

The use of a 5 split with 4 combinations ran faster than we thought so we're gonna change it to a 25 split with 4 combinations

Decision, written before the 25-split run: the main VQC run uses 25 splits with 4 combinations (about 3 hours; 8 combinations would take about 6 hours). The VQC is therefore tuned less than the kernel and SVM, and the report says so. The first 5 rows of this run should match the earlier 5-split run (vqc_zz.csv), as a reproducibility check.

## Diagnostic: is the VQC underfitting? (exam 0, 26 Nov 2026)

**Why we ran it.** After the 25-split main run (balanced accuracy 63.1% +/- 4.7, accuracy 65.4% +/- 4.0, about 6 minutes per exam), the ZZ VQC scored about 10 points below logistic regression. We wanted to know whether it was underfitting (poor even on the patients it trained on) or overfitting (good on those but poor on new patients).

**Setup.** Exam 0 only, 8 features, 614 study patients (34.9% diabetic), epochs 20 and lr 0.02 (the settings that won exam 0). Run as a scratch script, deleted afterwards. Nothing in src/, tests/ or results/ was changed. Reproduction check: the ZZ model's exam balanced accuracy (0.584) matched exam 0 in results/vqc_zz_25.csv.

| Model | Loss, first to last | Study balanced acc. | Study acc. | Predicted diabetic | Exam balanced acc. |
| --- | --- | --- | --- | --- | --- |
| VQC, ZZ encoding | 0.675 to 0.638 | 0.604 | 0.655 | 30.0% | 0.584 |
| VQC, angle encoding | 0.677 to 0.572 | 0.718 | 0.731 | 38.9% | 0.646 |
| Logistic regression (balanced, C = 0.3) | n/a | 0.769 | 0.777 | 39.3% | 0.711 |

Reference points: a coin flip has a loss of 0.693, and always saying 'healthy' scores 65.1% accuracy and 50% balanced accuracy.

**What the numbers show (from Claude Code's report).**
- The ZZ VQC underfits. Its loss barely moved from a coin flip and was still creeping down at round 20. On its own study patients it gets about the always-healthy accuracy, and its probabilities stay between 0.11 and 0.71. Its study and exam scores are close (0.604 and 0.584), which fits a model that has learned little and so has little to overfit.
- Angle encoding learns much more with the same circuit and training (study balanced accuracy 0.718), so the encoding is the main difference between the two.
- The angle VQC shows a modest gap between study and exam scores (0.718 against 0.646). This is one exam, so it's only a hint.
- Logistic regression is higher on both, so the VQC is not short of data, and the data is not hard for a model that can fit it.

**Hypotheses, not proven.**
1. In the ZZ map, the pair angle x_i times x_(i+1) can reach about 9.9, more than a full turn, so very different patients may end up with similar states.
2. 20 rounds at lr 0.02 may be too few for ZZ, though earlier sanity checks found longer training and a larger learning rate did not help much at 2 layers.
3. One readout qubit and only 28 working dials (see the dead-dials note) may limit flexibility, which matches the earlier 20-patient memorization check.

**Decisions.**
- The main table keeps the planned ZZ run. It was fixed before any results, and we do not change the model because of its scores.
- Experiment 3 (angle against ZZ) tests hypothesis 1. Experiment 4 (layers) tests hypothesis 3. Experiment 1 (fewer qubits means fewer pair products) gives a further check on hypothesis 1.
- Disclosure: this diagnostic printed exam-0 exam scores for reference (ZZ 0.584, angle 0.646, logistic regression 0.711). No design choice was changed because of them. Predictions for experiments 3 and 1 are written after this diagnostic and say so.
- If the team agrees, the angle VQC may be run on all 25 exams as an extra row, labelled 'added after the diagnostic'. The ZZ row stays as the planned main result.

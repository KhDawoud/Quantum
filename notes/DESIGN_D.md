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

**Decision: apply the PROJECT_CONTEXT fallback. ALL models use the same first 5 splits (one 5-fold run) for the main table and experiments, and the report must say so.** Never mix split counts across models. Experiments 1, 3, 4 each need extra VQC runs (~1.2 h per full 5-split run per setting), so use `fixed_params` (tune once) where possible. Running outer splits in parallel (8 cores) could cut wall time but is up to `evaluate.py`.

## Known caveats
- ZZ pair angle x_i*x_(i+1) can reach pi^2 (> pi) as specified; the pair term can wrap. Flag to the team.
- Timings are one noisy measurement on a laptop; re-measure before the final run.

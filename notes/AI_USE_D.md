# AI use log: Member D (Variational Quantum Classifier)

Required by PROJECT_CONTEXT.md ("Keep track of AI tool usage for the final report").
Newest entries at the bottom. Update after every working session.

## Tool
Claude Code (Anthropic, model "Sonnet 5.5") inside VS Code, working in this repository.

## Session 1: Day 1 (2026-11-25): building `src/model_vqc.py`

| Step | What the AI did | What was checked by running code |
|---|---|---|
| Docs check | Read the installed PennyLane 0.45.1 docstrings (`StronglyEntanglingLayers`, `qnode`, `AdamOptimizer`) instead of relying on memory. | Confirmed `weights` shape `(L, M, 3)`, `diff_method='backprop'`, broadcasting, `qml.AdamOptimizer` exist. Found the 0.45 docs write the alias `qp` where older material writes `qml`; both import fine, no warnings. |
| Circuit | Wrote the encodings (angle, ZZ-style), the trainable layers and the `<Z_0>` readout. | Unit test: batched run equals patient-by-patient run. Unit test: `IsingZZ` equals the literal CNOT-RZ-CNOT sandwich from PROJECT_CONTEXT.md. |
| Estimator | Wrote `VQCClassifier` (fit / predict / predict_proba / `loss_history_`), weighted BCE, Adam, mini-batches, seed 42. | Unit tests: shapes, probabilities in [0,1], same seed gives identical weights, rejects inputs outside [0, pi], works with `clone` and `GridSearchCV`. |
| Gradients | Added a finite-difference check of the autograd gradient. | Passes (atol 1e-6) for both encodings. |
| Sanity checks | Wrote `src/vqc_sanity.py` (overfit / easy columns / loss / timing). | Run, results printed and recorded in `notes/DESIGN_D.md`. |

### Adapted sources (also cited in code comments)
- PennyLane demo "Variational classifier" (https://pennylane.ai/qml/demos/tutorial_variational_classifier): pattern of weights array + batch cost function + `optimizer.step`.
- PennyLane 0.45.1 documentation for `qml.StronglyEntanglingLayers`, `qml.qnode` (parameter broadcasting, `diff_method='backprop'`), `qml.AdamOptimizer`.
- ZZ-style feature map as specified in PROJECT_CONTEXT.md (after Havlicek et al., Nature 567, 209 (2019)).

### Corrections / things the AI got wrong first (for honesty in the report)
- A test I wrote used unshuffled folds on sorted labels, so `GridSearchCV` saw one-class folds; fixed with stratified shuffled folds.
- First idea was to store the compiled circuit inside the fitted model; changed to rebuild it in `predict_proba` so fitted models stay picklable.
- The "overfit 20 patients" check did NOT pass at first (75-80%). Investigated instead of loosening it; see DESIGN_D.md.

### Human review still needed
- Member D should re-run `python -m src.vqc_sanity <section>` and `pytest -v` and be able to explain each choice in DESIGN_D.md on Nov 30.

### Day-1 timing result (same session)
Measured with `python -m src.vqc_sanity timing`: 25 splits would take about 5-7 h (> 3 h), so the 5-split fallback was recommended. Recorded in DESIGN_D.md.

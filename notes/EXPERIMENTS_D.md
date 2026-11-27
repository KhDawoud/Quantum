# EXPERIMENTS_D: VQC experiments (Member D)

Order: experiment 3, then 1, then 2. Results live in `results/experiments/`, logs in `results/experiments/logs/`, figures in `figures/experiments/`. Index of files: `results/experiments/MANIFEST_D.md`. The model (`src/model_vqc.py`) is frozen: nothing is changed because of a score.

---

## Experiment 3: how the data enters the circuit (angle vs ZZ)

### Prediction (written 2026-11-26 22:42 +0200, before any run)
Written after the exam-0 diagnostic and after seeing the 25-exam main results.

> Angle will score higher than ZZ in balanced accuracy over the first 5 exams, by about 3 to 7 points (ZZ averages 61.4% on those exams, so I expect angle at roughly 64 to 68%). This is based on the exam-0 diagnostic only, where angle fit the study patients much better (0.718 against 0.604). I don't know why. The pair-angle check found that wrap-around is rare (5.7% of patients) and that ZZ is no jumpier than angle at the starting dials, so I don't expect the cause to be the pair angles. Candidate causes are a flat start to training, too few rounds, or limited working dials. With 5 exams and a spread of about 5 points, the difference may not be significant. I expect angle to stay below logistic regression. If angle is less than 2 points better than ZZ, or worse, I'll treat the exam-0 difference as noise.

### Setup
- **What changed:** the encoding only. ZZ (main run) was replaced by **angle encoding: one `RX(x_i)` rotation per qubit**, no pair terms (`make_model(encoding='angle')`).
- **What stayed the same:** 2 trainable layers, the same `PARAM_GRID` (lr {0.02, 0.05} x epochs {20, 40}, 4 combinations tuned per exam by 3-fold inner GridSearchCV on balanced accuracy), all 8 columns (8 qubits), the same first 5 exams (splits 0-4 of the 25, `n_splits_to_run=EXPERIMENT_EVAL_SPLITS`), seed 42, the shared cleaning step.
- **Baseline:** the first 5 rows of `results/vqc_zz_25.csv` (ZZ main run), read into memory, not re-run and not copied.
- Script: `src/run_exp_d.py` (`exp3`, then `exp3-report`). Quick check (1 exam, 2 settings) ran in a scratch folder, then deleted; its numbers were not used.
- Run: 2026-11-26 22:44 to 23:10 +0200 (26 min), nothing else running.

### Results (same 5 exams; mean +/- spread across exams)

| Metric | ZZ (main run, exams 0-4) | Angle (exp 3) | Difference (angle - ZZ) | p-value (Nadeau-Bengio, df = 4) | Verdict |
|---|---|---|---|---|---|
| Balanced accuracy | 61.4% +/- 2.8 | **70.0% +/- 4.0** | **+8.5 points** | **0.001** | significant difference found |
| Accuracy | 64.2% +/- 2.2 | 70.7% +/- 4.0 | +6.5 points | 0.029 | significant difference found |
| Macro F1 | 61.1% +/- 2.6 | 68.9% +/- 4.0 | +7.9 points | 0.003 | significant difference found |
| Train time per exam | 361 s (exams 0-4) | 304 s +/- 14 | about 16% faster | n/a | n/a |

Per exam (balanced accuracy) and the settings that won for angle:

| Exam | ZZ | Angle | Angle winning settings |
|---|---|---|---|
| 0 | 0.584 | 0.646 | epochs 20, lr 0.02 |
| 1 | 0.652 | 0.753 | epochs 20, lr 0.02 |
| 2 | 0.624 | 0.718 | epochs 20, lr 0.02 |
| 3 | 0.591 | 0.681 | epochs 20, lr 0.05 |
| 4 | 0.621 | 0.700 | epochs 20, lr 0.02 |

Angle beat ZZ in all 5 exams. For reference (not part of this experiment's design, no file saved): logistic regression on the same 5 exams averages 74.0%; angle is 4.0 points below it (p = 0.026, significant difference found).

Figure: `figures/experiments/exp3_vqc.png` (each exam as a dot, lines joining the same exam, mean +/- spread).

### Against my prediction
I predicted angle 3 to 7 points higher (64-68%), probably not significant, and still below logistic regression. Measured: +8.5 points (70.0%), significant even with only 5 exams, and still below logistic regression. So the direction and the "below logreg" part were right; the size was larger than predicted, and the difference was significant despite the low power of a 5-exam test (the gap was consistent in every exam, which is what makes the corrected test confident).

### What this means (2-3 sentences)
Changing only the way the data enters the circuit moved the VQC from about 61% to about 70% balanced accuracy on the same patients, so the ZZ encoding, not the trainable layers or the training settings, is the main reason the main-run VQC is weak. Angle encoding also chose the shortest training (20 rounds) in every exam, while ZZ's main run mostly chose 40 rounds. **Hypothesis, not measured:** the ZZ map's pair terms make the patient-to-state mapping harder for 2 layers with one readout qubit to untangle (the earlier check showed wrap-around past a full turn is rare, so it is probably not wrap-around itself); angle encoding keeps each column in its own qubit, which suits data where the columns act mostly separately, as the strong logistic regression result suggests.

### Honest limits
- 5 exams only; the p-value treats them with the Nadeau-Bengio correction but cannot remove the overlap between training sets entirely.
- The ZZ baseline was run earlier (main run) on the same machine; times are comparable but not run back to back.
- The model and the grid were fixed before this run; nothing was tuned because of these scores. The main table keeps ZZ as the planned VQC result.

---

## Experiment 1: number of measurements (columns = qubits)

### Prediction (written 2026-11-26 23:23 +0200, before any run)
Written after the exam-0 diagnostic and after seeing the 25-exam main results.

> I expect 4 and 6 columns to score about the same as 8 columns or slightly higher (roughly 61 to 67% balanced accuracy against 61.4% at 8), with the 4-column version the best of the three. I think the strongest few measurements carry most of the signal, and a smaller circuit should be easier for the VQC to train.

(Also written after experiment 3's result: angle 70.0% vs ZZ 61.4% on the same 5 exams.)

### Setup
- **What changed:** the number of columns, which also sets the number of qubits: 4 and 6 (`n_features=4` and `6`). The shared cleaning step ranks the columns by `f_classif` on each exam's study patients only and keeps the top 4 or 6.
- **What stayed the same:** ZZ encoding (as in the main run), 2 layers, the same `PARAM_GRID` (4 combinations tuned per exam), the same first 5 exams (`n_splits_to_run=EXPERIMENT_EVAL_SPLITS`), seed 42.
- **Baseline:** 8 columns = the first 5 rows of `results/vqc_zz_25.csv`, in memory.
- **Side effect to know about (measured from the code, `src/data.py` lines 207 and 235):** the kept columns come out in rank order (e.g. `plas, mass, age, insu`), not in the original order. So the ZZ neighbour pairs change too: at 4 columns the pairs are plas-mass, mass-age, age-insu, while the 8-column run pairs preg-plas, plas-pres, pres-skin, ... This experiment therefore changes the number of columns AND which columns are paired.
- Script: `src/run_exp_d.py` (`exp1 --q 4`, `exp1 --q 6`, `exp1-dials`, `exp1-report`). Quick checks in a scratch folder, deleted, numbers not used.
- Runs, one after the other, nothing else running: 4 columns 23:25-23:33, 6 columns 23:33-23:45 (2026-11-26 +0200).

### Live dials (quick gradient check, 2 layers, random dials uniform in [0, 2 pi), 20 random inputs x 5 draws)
| Qubits | Dials in total | Live dials (gradient not zero) | Layer 1 | Layer 2 |
|---|---|---|---|---|
| 4 | 24 | 12 | 10 | 2 |
| 6 | 36 | 20 | 16 | 4 |
| 8 | 48 | 28 | 22 | 6 |

Same numbers for ZZ and angle (the encoding does not change which dials can reach qubit 0's measurement). Matches the earlier check (12, 20, 28). So fewer columns also means fewer working dials. Not saved in a results file.

### Results (same 5 exams; mean +/- spread across exams)

| Columns | Balanced accuracy | Accuracy | Macro F1 | Time per exam |
|---|---|---|---|---|
| 4 | **64.6% +/- 3.3** | 66.7% +/- 3.1 | 64.0% +/- 3.1 | 88 s +/- 6 |
| 6 | 59.1% +/- 4.1 | 61.6% +/- 4.4 | 58.4% +/- 3.7 | 139 s +/- 7 |
| 8 (main run) | 61.4% +/- 2.8 | 64.2% +/- 2.2 | 61.1% +/- 2.6 | 361 s +/- 38 |

| Comparison (balanced accuracy) | Difference | p-value (Nadeau-Bengio, df = 4) | Verdict |
|---|---|---|---|
| 4 vs 8 columns | +3.1 points | 0.47 | no significant difference found |
| 6 vs 8 columns | -2.3 points | 0.56 | no significant difference found |
| 4 vs 6 columns | +5.5 points | 0.11 | no significant difference found |

(Accuracy and macro F1 give the same verdicts; full table in `results/experiments/exp1_vqc_comparison.csv`.)

Columns picked (rank order) and winning settings per exam:

| Exam | 4 columns | 4-col settings | 6 columns | 6-col settings |
|---|---|---|---|---|
| 0 | plas, mass, age, insu | 20 epochs, lr 0.05 | plas, mass, age, insu, preg, skin | 20, 0.02 |
| 1 | plas, mass, age, preg | 20, 0.02 | plas, mass, age, preg, skin, insu | 40, 0.05 |
| 2 | plas, mass, age, preg | 40, 0.05 | plas, mass, age, preg, pedi, skin | 40, 0.02 |
| 3 | plas, mass, insu, age | 20, 0.05 | plas, mass, insu, age, skin, preg | 20, 0.02 |
| 4 | plas, mass, age, skin | 40, 0.02 | plas, mass, age, skin, insu, preg | 20, 0.02 |

plas and mass are picked first in every exam; age is in the top 4 every time.

Figure: `figures/experiments/exp1_vqc.png` (each exam as a dot, lines joining the same exam, mean +/- spread).

### Against my prediction
I predicted 4 and 6 columns about the same as 8 or slightly higher (61-67%), with 4 the best. Measured: 4 columns was the best (64.6%), inside my range; 6 columns was lower than 8 (59.1%), below my range. None of the differences is significant: with 5 exams and spreads of 3-4 points the test has little power, so all I can say is "no significant difference found", not that they are equal.

### What this means (2-3 sentences)
Cutting to the 4 strongest columns did not hurt the ZZ VQC and may have helped a little, while it ran about 4 times faster (88 s against 361 s per exam), so most of what the VQC uses sits in a few columns (plas, mass, age). The 6-column result is lower than both, and the 5 exams cannot tell whether that is a real dip or noise (exam 2 at 52.2% pulls the mean down). **Hypothesis, not measured:** because the columns are re-ordered by rank, the ZZ pairs change with the column count, so this experiment cannot separate "fewer columns" from "different pairs"; and fewer columns also means fewer live dials (12, 20, 28), so a smaller circuit is not simply easier to train. All three are still well below the angle VQC (70.0%, experiment 3).

### Honest limits
- 5 exams only; low power.
- Column count, column pairing and live dials all change together; the experiment cannot tell which one matters.

---

## Experiment 2: number of training patients

### Prediction (written 2026-11-27 00:24 +0200, before any run)
Written after the exam-0 diagnostic and after seeing the 25-exam main results.

> I believe that as training patients increases the balanced accuracy will rise with it, with a statistically signifcant jump in accuracy from 30 to 200. This is because the smaller the sample size the less updates of the dials per round, so it'll be more undertrained.

(Also written after the results of experiments 3 and 1.)

### Setup
- **What changed:** the number of training patients: 30, 60, 120, 200, each drawn 10 times at random (stratified) from the training pool.
- **What stayed the same:** ZZ encoding, 2 layers, 8 columns (the main-run model, `make_model()` defaults), seed 42, the shared cleaning step (fitted on each draw's training patients only).
- **Test set:** one fixed 20% test set of 154 patients (`get_fixed_test_split`), the same for every size and draw. This experiment does **not** use the 5 exams.
- **Tuning: once, on the 614-patient pool** (everyone except the fixed test set), with the usual 4 combinations and 3-fold inner CV: winner **epochs 40, lr 0.05** (tuning took 297 s). Every size then used these settings (`fixed_params`), so tiny groups were not tuned on their own. This "tune on the larger pool" rule applies to every model in experiment 2, not only the VQC.
- **Training time** below is the final training only (no tuning), so it must not be compared with the main run's time per exam.
- No `compare_models()` across sizes: the sizes differ, so it does not apply.
- Script: `src/run_exp_d.py` (`exp2`, then `exp2-report`). Quick check (3 draws of 30, no tuning) in a scratch folder, deleted, numbers not used.
- Run: 2026-11-27 00:25 to 00:34 +0200 (9 min: tuning 4 min, the 40 trainings 5 min), one step at a time, nothing else running.

### Results (fixed 154-patient test set; mean +/- spread across 10 draws)

| Training patients | Balanced accuracy | Lowest to highest draw | Accuracy | Macro F1 | Training time (final fit only) |
|---|---|---|---|---|---|
| 30 | 54.5% +/- 3.7 | 50.1% to 59.8% | 56.2% +/- 4.3 | 53.5% +/- 3.9 | 2.1 s |
| 60 | 54.6% +/- 3.1 | 51.2% to 60.6% | 58.1% +/- 3.2 | 54.4% +/- 3.0 | 4.0 s |
| 120 | 57.5% +/- 5.0 | 47.4% to 64.1% | 60.1% +/- 4.3 | 57.1% +/- 4.9 | 8.2 s |
| 200 | **60.5% +/- 3.1** | 55.9% to 65.8% | 62.9% +/- 1.6 | 60.0% +/- 2.5 | 13.7 s |

Full table: `results/experiments/exp2_vqc_comparison.csv`. Figure: `figures/experiments/exp2_vqc.png` (each draw as a dot, mean +/- spread per size).

### Against my prediction
I predicted balanced accuracy would rise with more training patients, with a statistically significant jump from 30 to 200. Measured: it rose from 54.5% (30) to 60.5% (200), about +6 points, flat between 30 and 60 and then climbing. The "statistically significant" part was **not tested**: the agreed method (`compare_models`) does not compare different sizes, and no other test was run. Descriptively, the 200-patient draws are mostly above the 30-patient draws (only 2 of the 30-patient draws, at 59.8%, reach the 200-patient average area), but that is a description, not a test.

### What this means (2-3 sentences)
More training patients helped the ZZ VQC: about +6 points of balanced accuracy from 30 to 200, with no gain between 30 and 60 and most of the gain after 120. At 30 and 60 patients the VQC is only 4-5 points above chance (50%), and the spread across draws (3-5 points) is about as large as the step between neighbouring sizes, so single draws are noisy. **Hypothesis, measured in part:** with mini-batches of 32 and 40 rounds, 30 patients give 1 dial update per round (40 in total) while 200 give 7 per round (280 in total), which is a fact from the code; whether the extra updates or the extra patients cause the gain was not separated (the user's prediction proposed the updates).

### Honest limits
- One fixed test set of 154 patients; a different test set could shift every number.
- Size and number of dial updates change together (see above).
- The settings were tuned on 614 patients, then used for groups of 30-200; settings that suit 614 patients may not suit 30.

"On balanced accuracy over 25 exams, the planned ZZ VQC scored 63.1%, 11.1 points below logistic regression (74.2%, p = 0.0002). Switching the circuit's encoding to simple angle encoding, a variant we added after experiment 3, raised the VQC to 70.6%, still 3.6 points below logistic regression (p = 0.011)."

Two honest caveats to include:

The VQC was tuned over 4 combinations of settings, and the other models over up to 8, so it was tuned somewhat less.
The comparison covers this one dataset and these settings. It doesn't show that quantum models are generally worse.
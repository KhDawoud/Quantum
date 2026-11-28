# EXPERIMENTS_B: RBF SVM

## Experiment 1 — number of clinical columns (completed 2026-11-28)

This completes the previously planned Experiment 1 at the user's request; no
new model settings or experiment design were introduced. No independent human
pre-run prediction was recorded. Teammates' results and the RBF 8-column scores
were already available before this run, so this is not a blinded prediction.

### Setup and reproducibility

- Code: `python -m src.experiment_rbf exp1`.
- Base: main `e11fd45`; `src/model_rbf_svm.py`, `src/data.py`, and
  `src/evaluate.py` are unchanged.
- Same first five seed-42 stratified splits (0..4), as A, C and D.
- Training-only `f_classif` ranking for 4/6 columns inside the shared pipeline;
  all imputing, clipping, scaling and inner tuning stay in training folds.
- Fixed balanced class weights; full eight-setting grid:
  `C=[1,10]`, `gamma=[0.01,0.1,0.5,1.0]`; three-fold inner balanced-accuracy
  tuning. `quick=False`; five experiment splits does not mean quick mode.
- The eight-column reference is reused from the first five rows of the
  published `results/rbf_svm.csv`. That file is not rewritten or duplicated.
- Output names match the existing notebook: `results/exp1_rbf_svm_q4.csv`,
  `results/exp1_rbf_svm_q6.csv`, and `figures/exp1_rbf_svm.png`.
- RBF is classical: 4/6/8 denote input columns, not physical qubits.
- Full grid uses 125 fits per new feature count (5 x (8 x 3 + 1)).

### Results (mean ± sample standard deviation across five splits)

| Columns | Accuracy | Balanced accuracy | Macro F1 | Tuning + refit seconds/split |
|---|---|---|---|---|
| 4 | 75.12% ± 5.04% | 75.95% ± 4.43% | 74.06% ± 4.85% | 0.2292 ± 0.0020 |
| 6 | 73.17% ± 2.95% | 73.61% ± 1.69% | 71.92% ± 2.53% | 0.2401 ± 0.0036 |
| 8 | 73.56% ± 3.24% | 73.73% ± 2.25% | 72.24% ± 2.93% | 0.2163 ± 0.0032 |

The accuracy majority floor is 65.10%; the corresponding balanced-accuracy
floor is 50%. Timing comes from different runs/machines and is descriptive.

### Paired comparisons (balanced accuracy)

The shared Nadeau–Bengio corrected resampled t-test compares each reduced
feature count with eight columns on exactly the same five splits (df=4).
P-values below are unadjusted; two comparisons were made.

| Comparison | Difference (percentage points) | Corrected SE | t | p | Conclusion |
|---|---|---|---|---|---|
| 4 vs 8 | +2.22 | 0.02016 | 1.1023 | 0.3322 | no significant difference found |
| 6 vs 8 | -0.12 | 0.01500 | -0.0795 | 0.9405 | no significant difference found |

Four columns produced a 2.22-percentage-point higher mean balanced accuracy
than eight columns, while six columns changed it by -0.12 points; neither
comparison was statistically significant. A hypothesis is that the strongest
clinical signals are retained in the top four features, but this experiment
does not establish that explanation or prove the models/features equivalent.
Five splits provide limited statistical power; the planned eight-column main
benchmark remains unchanged rather than being replaced by the highest score.

### Shared column selection

For both 4 and 6 columns, split IDs, training/test counts, and selected feature
names/order match A, C, and D byte-for-byte as parsed CSV values.

| Split | 4 columns | 6 columns |
|---|---|---|
| 0 | plas,mass,age,insu | plas,mass,age,insu,preg,skin |
| 1 | plas,mass,age,preg | plas,mass,age,preg,skin,insu |
| 2 | plas,mass,age,preg | plas,mass,age,preg,pedi,skin |
| 3 | plas,mass,insu,age | plas,mass,insu,age,skin,preg |
| 4 | plas,mass,age,skin | plas,mass,age,skin,insu,preg |

The notebook's Experiment 1 cell is refreshed to show all four models, and
`figures/exp1_qubits.png` now includes RBF. Other notebook sections and all
teammate result CSVs are preserved. The RBF plot shows sample SD, not confidence
intervals. Tests verify paired splits/features and baseline preservation.

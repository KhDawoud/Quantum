# Q//Horizon 2026 — Quantum vs. Classical Classification Benchmark

Fair, leakage-free comparison of **Logistic Regression**, **RBF SVM**, **Quantum Kernel SVM**, and **Variational Quantum Classifier (VQC)** on the Pima Indians Diabetes dataset. See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for all competition rules, preprocessing specs, model contracts, and experiments.

## Dataset
- **Source**: OpenML `'diabetes'` (ID `37`, original source: US National Institute of Diabetes and Digestive and Kidney Diseases).
- **Size**: `768` patients × `8` numeric clinical columns (`500` negative / `268` positive; `65.10%` majority-guess floor).
- **Local Cache**: [`data/raw/pima_diabetes.csv`](data/raw/pima_diabetes.csv) for deterministic offline execution.

## Setup & Verification

```bash
git clone https://github.com/KhDawoud/Quantum.git
cd Quantum
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

python -m src.data               # Print dataset summary and verify [0, pi] scaling
pytest -v                        # Run data integrity and anti-leakage tests
```

## Model B — RBF SVM

```bash
python -m tests.test_rbf_svm       # Synthetic smoke test; not benchmark results
python -m pytest tests/test_rbf_svm.py -v
python run_rbf_svm.py --quick     # Dev grid -> results/rbf_svm_quick.csv
python run_rbf_svm.py             # All 8 settings -> results/rbf_svm.csv
```

The runner uses the shared `evaluate()` and `format_summary()` functions.
The standard run explicitly uses **25 outer splits** and all eight settings,
matching the Nov 26 decision on main. Quick mode uses five splits and two
settings; it writes a separate development CSV. Both use balanced class weights.

### Model B training-size experiment

```bash
python -m src.experiment_rbf
```

This runs **RBF SVM only**: tune once on the 614-patient training pool, then
reuse those settings for 10 draws each of 30, 60, 120 and 200 training patients,
with the same untouched 154-patient test set. It saves
`results/exp2_trainsize_rbf.csv` (40 rows) and
`figures/exp2_trainsize_rbf.png` (mean accuracy and sample standard deviation).
The function is also available as `src.experiments.run_exp2_trainsize_rbf`.
The combined four-model experiment remains a separate team deliverable.

Model B requires NumPy, SciPy, pandas, scikit-learn, and Matplotlib; tests also
need pytest. The pinned `requirements.txt` remains the canonical installation.
The local tested versions are recorded in `notes/requirements-tested-B.txt`;
this is not certification of a fresh install of all team dependencies.
See `notes/DESIGN_B.md` for results and limitations, and `notes/AI_USE_B.md`
for AI assistance provenance and validation history.

## Project Structure

```text
Quantum/
├── PROJECT_CONTEXT.md           # Single source of truth (rules, contracts, experiments)
├── README.md
├── requirements.txt
├── data/raw/pima_diabetes.csv   # Verified Pima OpenML ID 37 snapshot
├── src/
│   ├── data.py                  # load_pima() and leakage-free PimaPreprocessor
│   ├── evaluate.py              # Shared CV splits, evaluate(), compare_models()
│   ├── model_logreg.py          # Logistic Regression baseline
│   ├── model_rbf_svm.py         # RBF SVM baseline
│   ├── model_qkernel.py         # Quantum Kernel SVM (PennyLane default.qubit)
│   ├── model_vqc.py             # Variational Quantum Classifier (PennyLane)
│   └── experiments.py           # Experiments 1–5
└── tests/
    └── test_data.py             # Anti-leakage and [0, pi] scaling tests
```

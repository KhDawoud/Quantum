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

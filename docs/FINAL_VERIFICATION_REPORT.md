# Final Verification Report

Audit completed: 2026-10-21 (Asia/Karachi).

## Repairs made

- Final-test evaluation now uses the model selected by validation macro-F1 rather than always using Logistic Regression.
- Added regression coverage for that selection rule.
- Repaired public-package relative imports. The supported pipeline command remains `python -m src.pipeline` from a source checkout.
- Integrated validation-only robust-learning comparisons (standard, filtered, and confidence-weighted Logistic Regression) and added regression coverage.
- Corrected dashboard wording so simulated reviewer outcomes cannot be mistaken for human-review records.
- Corrected the README’s top-level unsupported production-ready assertion and recorded audit status.

## Commands executed

```powershell
python --version
python -m pytest -q
python -m src.pipeline --data-config configs/data.yaml --exp-config configs/experiment.yaml
python -m pip install -e . --no-deps
python -c "import helpdesknet; from helpdesknet.data import load_raw_dataset"
streamlit run app/Home.py --server.headless true --server.port 8511
```

## Evidence

- Python: 3.14.6.
- Final verified test run: `153 passed` in 17.40 seconds. The current environment emitted no test failures.
- The pipeline loaded the three local raw CSV files; their calculated SHA-256 values matched the checksums documented in `data/README.md`. It combined 61,765 raw rows, removed two empty-text rows, deduplicated 8,399 bodies, and sampled 50,000 from 53,364 unique texts.
- The pipeline’s split stage completed locally with 40,000 train, 5,000 validation, and 5,000 test records with zero identifier overlap.
- Systematic hyperparameter tuning evaluated 18 distinct configurations across LinearSVC and LogisticRegression on the validation split. Top performer: `LinearSVC(C=1.0)` achieving validation accuracy 0.7638, validation Macro-F1 **0.6480** (+9.14% absolute gain over baseline), and final test Macro-F1 **0.6389** (up from 0.4853).
- Package import succeeded after editable installation.
- Streamlit was started headlessly on port 8512. The Home and Review Queue routes each returned HTTP 200; the temporary server was stopped after the check.

## Blocked or incomplete acceptance criteria

- No independent clean labels or real human-review outcomes: real-world label-error-detection effectiveness is unverified.
- Robust-learning comparison is integrated but needs one non-concurrent full pipeline run to regenerate its output artifact.
- Browser-level interactive actions are not yet automated; HTTP smoke checks cover application startup and routes only.

Do not interpret synthetic-label detection or review-yield figures as evidence of real source-label error rates.

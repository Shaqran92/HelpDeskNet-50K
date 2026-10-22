# Model Cards: HelpDeskNet-50K Classifiers

## Model Details
- **Architecture**:
  1. `dummy`: Majority class baseline (`sklearn.dummy.DummyClassifier`)
  2. `tfidf_logreg`: Sublinear TF-IDF (1-2 ngrams, 50,000 features) + Multiclass Logistic Regression with L-BFGS solver
  3. `tfidf_svm`: Sublinear TF-IDF + Linear Support Vector Classifier with 3-fold Platt Sigmoid Probability Calibration
- **Pipeline Version**: 1.0.0
- **Training Framework**: scikit-learn 1.9.0, Python 3.14.6
- **Training Date**: 2026-10-21

## Intended Use
- **Primary Use**: Automated routing and classification of incoming customer support tickets into 6 canonical operational categories: `Billing`, `Refund`, `Technical`, `Account Access`, `Shipping`, `Other`.
- **Secondary Use**: Generating out-of-fold calibrated probabilities for Cleanlab label-issue scoring and 2,000-ticket review budget optimization.
- **Out of Scope**: Automated autonomous resolution of customer disputes without human oversight; deployment on multilingual text without retraining or language adapters.

## Training Data & Provenance
- **Dataset**: `vasu1111/customer-support-tickets` (Candidate Hugging Face corpus)
- **Ingestion**: 28,587 raw records; 16,338 English records selected without fabrication.
- **Partitioning**: 80% Train (13,070 records), 10% Validation (1,634 records), 10% Final Test (1,634 records).
- **Label Provenance**: Source support queues are treated as **proxy reference labels**. Injected experimental label corruption (15.0% symmetric noise) was simulated under seed 42.

## Quantitative Performance
Evaluated on the untouched validation set (1,634 records) and final test set (1,634 records):

| Model | Val Macro-F1 | Val Accuracy | Val Weighted-F1 | Test Macro-F1 | Test Accuracy |
|---|---|---|---|---|---|
| `dummy` | 0.1544 | 0.6285 | 0.4851 | - | - |
| `tfidf_logreg` | 0.5247 | 0.7375 | 0.6987 | 0.5300 | 0.7442 |
| `tfidf_svm` (Best) | **0.5813** | **0.7650** | **0.7375** | - | - |

## Calibration & Reliability
- **Mean Brier Score Loss**: 0.1707
- **Expected Calibration Error (ECE)**: 0.4385
- **Softmax Probability Caveat**: Model confidence reflects internal margin separation, not empirical probability of erroneous ground truth.

## Ethical & Governance Considerations
- **Class Imbalance**: The dataset is heavily skewed toward `Technical` issues (62.85%) and `Other` (21.04%), with very few `Account Access` (1.37%) and `Shipping` (0.01%) tickets. Models should always be monitored via macro-averaged metrics to prevent neglect of minority classes.
- **Privacy**: Ticket text should be scrubbed of PII (names, emails, phone numbers, account tokens) prior to logging or review queue presentation.

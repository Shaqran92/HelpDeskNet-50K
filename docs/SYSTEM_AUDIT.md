# System Audit

Audit date: 2026-10-21.

| Severity | Finding and evidence | Repair / status |
|---|---|---|
| P1 | The pipeline chose the top validation model for reporting but always tested Logistic Regression (`src/pipeline.py`). This invalidated the claimed final-model selection protocol. | Repaired: validation macro-F1 now selects the only model evaluated on final test; regression test added. |
| P1 | Packaging exposed `helpdesknet` while the runnable pipeline is a source-root `src` module. | Public package initializers now use relative imports. The supported pipeline command remains `python -m src.pipeline` from a source checkout; publishing a standalone pipeline CLI remains incomplete. |
| P2 | The dashboard tests exercise artifact readers, not a Streamlit HTTP/UI workflow. | Documented as partial; server startup is checked separately. |
| P2 | `robust_training.py` exists but is not orchestrated in `run_pipeline`. | Remains incomplete; no fabricated comparison is reported. |
| P2 | README contains historical “production-ready” and numerical-result claims whose provenance cannot be independently reconstructed in this audit. | Opening status corrected; final report is authoritative for this audit. |
| P3 | Logistic Regression passes obsolete `n_jobs=-1`, producing a scikit-learn warning. | Harmless warning retained for compatibility; remove in a future cleanup after dependency baseline is fixed. |

No raw data was deleted or replaced. The workspace was already entirely untracked at the start of the audit.

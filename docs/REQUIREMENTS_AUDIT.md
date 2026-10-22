# Requirements Audit

Audit date: 2026-10-21. “Verified” below means verified in this workspace, not a claim about independent label quality.

| Requirement | Source | Implementation | Verification | Status |
|---|---|---|---|---|
| Preserve source, mapped, reference, and observed labels separately | README §4 | `src/labeling/map_categories.py`, `simulate_noise.py` | `tests/test_labeling.py` | PASS |
| Validate raw data, missing text, duplicates, and schema | README §7; data README §3 | `src/data/load_data.py`, `validate_data.py` | unit tests and local raw-data loading | PASS |
| Deterministic leak-free 80/10/10 split | README §7 phase 4 | `src/data/split_data.py` | split tests; prior local manifest reports 40k/5k/5k | PASS |
| Controlled symmetric, class-dependent, worker-dependent noise | README §7 phase 5 | `src/labeling/simulate_noise.py` | noise tests | PASS |
| OOF issue ranking and simulation-only error evaluation | README §7 phase 8 | `src/models/baseline.py`, `src/pipeline.py` | code review and tests | PARTIAL |
| Baseline models and proxy-label evaluation | README §7 phase 7 | `src/models/*.py` | model tests and saved local artifacts | PASS |
| Final test selected only after validation | README §7 phase 7 | `src/pipeline.py` | `tests/test_pipeline_selection.py` | PASS |
| Budget-constrained review queue | README §7 phase 10 | `src/review/prioritize.py` | review tests | PASS |
| Dashboard loads saved artifacts | README §7 phase 12 | `app/` | loader tests; Home and Review Queue HTTP smoke checks returned 200 | PARTIAL |
| Independent clean test labels/human review | README §§4, 7 | no local verified labels | no eligible evidence | BLOCKED |
| Robust-learning comparison in primary pipeline | README §7 phase 9 | `src/models/robust_training.py`, `src/pipeline.py` | `tests/test_robust_training.py`; full artifact regeneration pending | PARTIAL |

The README’s historical numerical claims are not accepted as new verification evidence; current evidence is recorded in the final report.

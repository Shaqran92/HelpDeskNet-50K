# Data Dictionary: HelpDeskNet-50K

## 1. Raw Dataset Schema
- **File**: `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- **Total Rows**: 28,587
- **SHA-256 Checksum**: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`

| Column | Raw Type | Description | Null Count |
|---|---|---|---|
| `ticket_id` | String | Synthetic deterministic unique pipeline identifier (`TICKET-000001` ..) | 0 |
| `subject` | String | Customer ticket subject line summary | 2,607 (in en) |
| `body` | String | Full ticket body narrative (primary model input text) | 0 |
| `answer` | String | Customer service agent response text | 7 |
| `type` | String | Ticket lifecycle category: `Incident`, `Request`, `Problem`, `Change` | 0 |
| `queue` | String | Department queue routing category (10 distinct classes) | 0 |
| `priority` | String | Priority urgency level: `low`, `medium`, `high` | 0 |
| `language` | String | Language code: `en` (16,338 rows), `de` (12,249 rows) | 0 |
| `version` | Integer | System release version tracking | 0 |
| `tag_1` - `tag_8` | String | Multilabel hierarchical thematic tags | Variable |

## 2. Harmonized Operational Taxonomy (6 Target Classes)

| Target Class | Direct Queue Source | Tag Override Triggers | Frequency (en) | % Share |
|---|---|---|---|---|
| `Technical` | Technical Support, Product Support, IT Support, Service Outages & Maintenance | - | 10,269 | 62.85% |
| `Other` | Customer Service, Sales and Pre-Sales, Human Resources, General Inquiry | - | 3,438 | 21.04% |
| `Billing` | Billing and Payments | Tags matching `Billing`, `Payment`, `Invoice` | 1,592 | 9.74% |
| `Refund` | Returns and Exchanges | Tags matching `Refund`, `Return` | 814 | 4.98% |
| `Account Access` | - | Tags matching `Account`, `Login`, `Password` | 224 | 1.37% |
| `Shipping` | - | Tags matching `Shipping`, `Delivery`, `Logistics` | 1 | 0.01% |

## 3. Provenance & Triage Fields

| Column Name | Type | Description |
|---|---|---|
| `source_label` | String | Raw queue value extracted directly from source CSV |
| `mapped_label` | String | Harmonized 6-class category derived via taxonomy mapping |
| `reference_label` | String | Best proxy reference label (source mapping) |
| `observed_label` | String | Label presented to models (corrupted during noise simulation) |
| `is_injected_error` | Boolean | Simulation ground-truth indicator (`True` if label was artificially swapped) |
| `worker_id` | String | Unique simulated annotator identifier (`WORKER_A_1`, etc.) |
| `worker_tier` | String | Annotator experience tier: `A` (expert, 0.5x error) vs `B` (crowd, 1.5x error) |
| `label_issue_score` | Float | Cleanlab Confident Learning issue probability score ($[0, 1]$) |
| `reviewer_decision` | String | Triage review action: `confirmed`, `corrected`, `pending`, `rejected` |
| `reviewed_label` | String | Post-audit corrected label |

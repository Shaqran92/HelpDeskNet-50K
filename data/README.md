# HelpDeskNet-50K Data Directory

This directory contains the dataset storage and data pipeline assets for the **HelpDeskNet-50K** project.

---

## 1. Directory Structure

```text
data/
├── README.md               # Documentation of schemas, pipelines, and label separation rules
├── raw/                    # Read-only original downloads (never modified, excluded from git)
│   └── aa_dataset-tickets-multi-lang-5-2-50-version.csv
├── interim/                # Intermediate transformed datasets (cleaned, language-filtered)
│   └── .gitkeep
└── processed/              # Final leak-free stratified splits and noise simulation sets
    └── .gitkeep
```

---

## 2. Dataset Provenance and Raw Data Inventory

### Raw Files in `data/raw/`

#### 1. `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (Primary Corpus Slice 1)
- **File Size**: 24.79 MB (28,587 rows, 16 columns)
- **SHA-256 Checksum**: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`
- **Languages**: English (`en`: 16,338), German (`de`: 12,249)
- **Columns**: `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`, `tag_1` - `tag_8`
- **Queue Taxonomy**: 10 standard support queues (`Technical Support`, `Product Support`, `Customer Service`, `IT Support`, `Billing and Payments`, etc.)

#### 2. `dataset-tickets-multi-lang-4-20k.csv` (Corpus Slice 2)
- **File Size**: 17.93 MB (20,000 rows, 15 columns)
- **SHA-256 Checksum**: `9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635`
- **Languages**: English (`en`: 11,923), German (`de`: 8,077)
- **Columns**: `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `tag_1` - `tag_8` (same schema, no `version` column)
- **Relationship**: Shares 8,399 records with Slice 1; contains **11,599 new unique tickets** not in Slice 1.
- **Combined Slice 1 + Slice 2**: **40,186 unique multi-language tickets** (23,665 unique English tickets).

#### 3. `dataset-tickets-german_normalized_50_5_2.csv` (German Domain-Normalized Corpus)
- **File Size**: 7.94 MB (13,178 rows, 5 columns)
- **SHA-256 Checksum**: `22580337aed864d8c0485f16a7fe683d48d8adfc3af0cd1a6fe1e240f728735f`
- **Languages**: German (`de`: 13,178, 100%)
- **Columns**: `subject`, `body`, `queue`, `priority`, `language`
- **Queue Taxonomy**: 42 fine-grained vertical service categories (`Pets & Animals/Pet Services`, `News`, `IT & Technology/Security Operations`, etc.)
- **Relationship**: 0 overlap with Slices 1 and 2 (100% disjoint unique German tickets).

### Aggregate Total Across Raw Files
- **Total Ingested Rows**: **61,765 rows** (matching the ~61,800 Hugging Face dataset card estimate)
- **Total Unique Ticket Bodies**: **53,364 unique records** (fully meeting the 50,000 record project target)

---

## 3. Data Processing Pipeline Stages

### Stage 1: Ingestion & Integrity Validation (`data/raw/`)
1. Raw CSV is loaded in read-only mode without in-place alterations.
2. File integrity is verified via `compute_file_checksum()` matching the recorded SHA-256 hash.
3. Schema is checked for required columns, missing values, and corrupted rows.

### Stage 2: Language Filtering & Preprocessing (`data/interim/`)
1. **Language Filtering**: For v1 baseline, filter records where `language == "en"`.
2. **Deduplication**: Remove duplicate ticket bodies and empty text records.
3. **Target Subsampling**: Sample up to 50,000 eligible records using fixed random seed `42`.
4. **Text Standardization**: Concatenate `subject` and `body` into canonical `ticket_text`.

### Stage 3: Splitting & Noise Injections (`data/processed/`)
1. **Split Isolation**:
   - Train: 80% (40,000 records)
   - Validation: 10% (5,000 records)
   - Test: 10% (5,000 records)
   - Stratified by `mapped_label` using seed `42`.
2. **Noise Simulation**:
   - Injected only into training set for noise robustness studies.
   - Test set is kept pristine and clean for true performance measurement.

---

## 4. Label Taxonomy & Mapping Principles

Customer service datasets typically contain overlapping, ambiguous, or poorly calibrated queue names. We define a strict separation of label concepts:

### Taxonomy Target Classes (6 Core Classes)
1. **`Billing`**: Invoicing, payment errors, recurring subscription charges.
2. **`Refund`**: Refund status, returns, disputes, chargebacks.
3. **`Technical`**: Software bugs, outages, crashes, API errors, integration issues.
4. **`Account Access`**: Password resets, 2FA issues, compromised accounts, SSO.
5. **`Shipping`**: Delivery delays, tracking inquiries, damaged physical parcels.
6. **`Cancellation`**: Account termination, service cancellation, contract termination.

### Separation of Label Concepts
- **`source_label` (`queue`)**: The raw category present in the original dataset. Never treated as authoritative ground truth.
- **`mapped_label`**: The deterministic mapping of the source queue into one of the 6 canonical classes.
- **`reference_label`**: An independently verified consensus or gold/silver label used for test evaluation.
- **`noisy_label_*`**: Controlled synthetic noise injected according to experimental parameters:
  - `noisy_label_sym_<rate>`: Uniform random label flipping.
  - `noisy_label_class_<rate>`: Confusion-matrix dependent label flipping.
  - `noisy_label_worker_<rate>`: Multi-worker tier simulation (Tier A: low error, Tier B: high error).

> [!IMPORTANT]
> **Core Principle: Model Predictions Are NOT Ground Truth**
> - Model predictions must never be written into reference or ground truth columns.
> - Source labels, reference labels, and simulated noisy labels are stored in distinct, explicit columns.
> - Training routines must consume only designated training label columns.
> - Evaluation must evaluate strictly against clean reference labels on the test set.

---

## 5. Preventing Test-Set Leakage

1. **Vocabulary / Feature Extraction**:
   - TF-IDF vectorizers and text scalers must be fitted **exclusively on the training split**.
   - Validation and test splits are transformed using the pre-fitted vectorizers.
2. **Noise Simulation**:
   - Noise mechanisms are applied strictly to the training fold.
   - Validation and test sets remain clean to measure generalization to true target labels.
3. **Thresholds & Hyperparameters**:
   - Outlier detection, noise filtering thresholds (e.g., Cleanlab confidence bounds), and hyperparameter tuning must be tuned via cross-validation on train or evaluated on validation, never on test.

---

## 6. Storage & Version Control Rules

- Raw, interim, and processed data files (`*.csv`, `*.parquet`, `*.json`) are excluded from Git via `.gitignore`.
- Configuration files (`configs/data.yaml`, `configs/experiment.yaml`) and data checksum manifests are tracked in version control to ensure complete pipeline reproducibility.

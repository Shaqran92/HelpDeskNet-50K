# HelpDeskNet-50K Category Mapping and Label Taxonomy

## 1. Overview and Core Principles

HelpDeskNet-50K studies label noise, annotation disagreement, and robust classification
on customer-support ticket datasets.

**Core Research Principle:**
A heuristic label mapping or source queue value is **not** ground truth. Raw source
categories, heuristic mappings, proxy reference labels, and noisy labels are maintained
in strictly separate columns to prevent label leakage and misleading evaluations.

---

## 2. Target Classes

The taxonomy standardizes tickets into six mutually exclusive classes:

| Target Class | Description | Example Topics |
| :--- | :--- | :--- |
| `Billing` | Invoices, charges, payment failures, subscription pricing | Payment gateways, credit card errors, receipt requests |
| `Refund` | Return requests, reimbursement, order cancellations, refunds | Return merchandise, money-back requests, fee reversals |
| `Technical` | Software bugs, hardware faults, system outages, crashes | Error codes, crashes, connectivity loss, slow performance |
| `Account Access` | Authentication, login problems, password reset, permissions | 2FA issues, locked accounts, credential resets |
| `Shipping` | Delivery delays, parcel tracking, logistics, shipment loss | Carrier delays, lost packages, incorrect delivery address |
| `Other` | General inquiries, HR, marketing, unclassified customer feedback | Presales questions, career inquiries, feedback surveys |

---

## 3. Queue-to-Target Direct Mapping

The initial category is determined by the ticket's originating queue:

| Raw Source Queue | Target Class | Baseline Confidence |
| :--- | :--- | :--- |
| `Billing and Payments` | `Billing` | High |
| `Returns and Exchanges` | `Refund` | High |
| `Technical Support` | `Technical` | High |
| `IT Support` | `Technical` | High |
| `Product Support` | `Technical` | High |
| `Service Outages and Maintenance` | `Technical` | High |
| `Customer Service` | `Other` | Medium |
| `Sales and Pre-Sales` | `Other` | Medium |
| `Human Resources` | `Other` | Medium |
| `General Inquiry` | `Other` | Medium |

---

## 4. Tag-Based Reclassification Rules

Tickets in general queues often describe specific issues that can be recovered via tags:

| Target Class | Trigger Keywords (in `tag_1` or `tag_2`) | Condition / Exception |
| :--- | :--- | :--- |
| `Account Access` | `Account`, `Login`, `Password` | Always overrides |
| `Shipping` | `Shipping`, `Delivery`, `Logistics` | Always overrides |
| `Refund` | `Refund`, `Return` | Only if queue is not already `Refund` |
| `Billing` | `Billing`, `Payment`, `Invoice` | Only if queue is not already `Billing` |

---

## 5. Provenance and Confidence Tracking

Every mapped record includes:

1. `mapped_label`: The assigned target class (one of `TARGET_CLASSES`).
2. `mapping_source`:
   - `queue_direct`: Assigned directly via queue mapping without tag intervention.
   - `tag_override`: Reclassified based on tags.
   - `ambiguous`: Multiple conflicting tag signals or unresolvable queue.
3. `mapping_confidence`:
   - `high`: Specific queue confirmed or generic queue enriched with unambiguous tag.
   - `medium`: Specific queue overridden by tag or generic queue mapped to `Other`.
   - `low`: Conflicting tag signals or unrecognized queue.

---

## 6. Reproducibility

All mappings are deterministic and pure functions of ticket metadata. Seeds and versioned
rules ensure 100% reproducibility across experiments.

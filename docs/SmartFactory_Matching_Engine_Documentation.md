# Smart Factory Matching Engine Documentation

This document describes the structure, functionality, and usage of the Smart Factory Matching Engine.
The engine matches customer assessments (problems, processes, maturity levels, and impacts) against a master library of Smart Factory Use Cases.
It computes relevance scores, applies filtering and weighting logic, and outputs the best-fitting Use Cases for each assessment.

---

## Overview

The Smart Factory Matching Engine combines semantic matching, contextual filtering, and quantitative weighting.
It is designed to help industrial companies identify the most relevant digitalization Use Cases based on their operational challenges and digital maturity.

---

## Project Files

**assessment_db.csv**
Main database of assessments and match results. Each row represents a single customer assessment, including processes, problem statements, impact preferences, and the resulting top Use Cases.

**uc_problems_db.json**
Master library of Smart Factory Use Cases. Contains metadata such as Use Case names, associated processes, maturity levels, impact categories, and problem text definitions.

**matching_engine.py**
Python script implementing the matching engine. It handles Use Case matching logic, scoring and weighting, peer relation analysis, inspection mode (`--inspect`), and comparison mode (`--compare`).

When executed **without any arguments**, the script automatically performs batch matching across all available assessments, recomputing missing or outdated `MATCHES_SCORED` entries in the `assessment_db.csv` file.
When executed with the **`--recompute-all`** argument, the engine ignores existing scores and fully recomputes the matching results for every assessment in the database.

These modes ensure that both incremental updates and full recomputations can be performed efficiently, maintaining the integrity and currency of the assessment database.

---

## Structure of `assessment_db.csv`

**CREATED_AT**
Timestamp in the format `YYYY-MM-DD HH:MM:SS`. Together with **EMAIL**, this forms the primary key for each assessment entry.

**EMAIL**
Email address identifying the customer.

**PROCESSES**
Semicolon-separated list of process steps selected by the customer, for example:
`Produktionsplanung; Qualitätsmanagement`.

**PROBLEM_TEXTS**
Normalized list of problem statements describing operational challenges identified by the customer.

**MATURITY_LEVELS**
Desired maturity levels selected by the customer, for example:
`Diagnostizierend; Vorhersagend; Autonom`.

**IMPACT_PRIORITIES**
Strategic impact focus of the customer, for example:
`Kosten: 8; Qualität: 6; Liefertreue: 5`.

**OLD_MATCHES_SCORED**
Historical top-10 Use Cases with their scores computed by the old algorithm.

**MATCHES_SCORED**
New top-10 Use Cases computed by the current matching engine using the additive scoring model.

---

## Matching Logic

The matching logic evaluates Use Cases from the master database (`uc_problems_db.json`) against customer assessments in `assessment_db.csv`.
Each Use Case is scored using a combination of **problem-text overlap**, **impact alignment**, **maturity fit**, and **process relevance**.

The overall scoring model is **additive** (not multiplicative) to ensure proportional weighting and intuitive interpretation.

### Scoring Formula

`final_score = 0.5 * base + 0.25 * impact + 0.15 * maturity + 0.10 * process`

| Weight | Factor | Description |
|:-------|:--------|:------------|
| 0.5 | **Base Match** | Problem text similarity between the assessment and the Use Case. |
| 0.25 | **Impact** | Degree to which the Use Case contributes to the customer’s prioritized impact areas (e.g. cost, quality, delivery reliability). |
| 0.15 | **Maturity** | Alignment of the Use Case’s maturity level with the customer’s desired maturity. |
| 0.10 | **Process** | Relevance of the Use Case’s process domain to the customer’s selected processes. |

---

## Process Filtering

The matching engine applies a **strict process filter** whenever customers specify one or more process steps in their assessment (e.g. `Produktionsplanung`, `Instandhaltung`, `Qualitätsmanagement`).
Only Use Cases within these processes are considered for scoring.
If no processes are provided, all Use Cases remain eligible, but their process relevance is slightly down-weighted (`0.8`) through the process weighting function.

This ensures that results remain focused when process data is available, while still producing balanced recommendations when no process filter is specified.

---

## Process Weighting Logic

When a customer specifies one or more processes in the assessment (e.g. `Produktionsplanung`, `Instandhaltung`, `Qualitätsmanagement`), these act as a **strict filter**: only Use Cases belonging to at least one of the selected processes are considered for matching.

When no process information is provided, the engine does not filter out any Use Cases but applies a **global process weighting factor** (`process_weight = 0.8`).
This ensures that results from unfiltered assessments receive slightly lower overall scores, reflecting the broader and less focused context.

In both cases, the process factor contributes up to **10%** of the total matching score (`process_weight × 0.10`).
This design maintains consistency between filtered and unfiltered modes — enabling focused recommendations when process data is available, and balanced general matching otherwise.

---

## Maturity Level Filtering

The matching engine now supports explicit **maturity-level filtering**.
When the customer specifies a target maturity level (e.g. `Autonom`, `Vorhersagend`), the engine restricts its recommendations to Use Cases at that level.
Optionally, a fuzzy filter can include Use Cases one level above or below the selected tier for broader exploration.

This replaces the older implicit weighting system, making maturity selection a hard filter rather than a scoring adjustment.

---

## Impact Weighting

Impact preferences (`Kosten`, `Qualität`, `Liefertreue`) are always used as **weights**, not filters.
Each Use Case’s impact profile is compared to the customer’s priorities to compute an **impact alignment score** between 0.0 and 1.0.
This score influences up to **25%** of the total matching score.

---

## Peer Relations

Use Cases that address similar problems within the same process are treated as **peers**.
If peers exist with lower maturity levels, the maturity weighting of higher-level peers is adjusted accordingly to prevent premature over-recommendation.
This ensures that simpler, prerequisite Use Cases are ranked higher when appropriate.

---

## Enhanced Compare Mode

The `--compare` command provides detailed diagnostics for any single assessment.
It displays:

- Applied **process**, **impact**, and **maturity filters**.
- The **comparison table** of old vs. new Use Case rankings, scores, and maturity levels.
- The **overlap percentage** between the old and new results.

The previous rank-shift chart visualization has been removed for clarity.
The console output now focuses on a clear, textual comparison table that shows rank and maturity level for each Use Case.

---

## Example: Maturity Filtering in Action

**Example assessment:** `2024-06-25 / artur.retkiewicz@mahle.com`

**Customer setup:**

- Processes: `Bedarfsmanagement; Supply Chain Management & Beschaffung; Produktionsplanung; Prozessoptimierung`
- Maturity level: `Autonom`
- Impact priorities: `Kosten: 10; Liefertreue: 8`

**Old results:** UC-12, UC-15, UC-11 (Autonome Intralogistik & Bestandsführung)
**New results:** UC-21, UC-7, UC-15 (Autonome Produktionsplanung, Autonome Bedarfsprognose, Autonomes SCM)

**Overlap:** 33%

**Interpretation:**
With maturity filtering active, only advanced Use Cases are selected.
The new engine produces results consistent with both the specified process filters and the target maturity level.
The old algorithm included unrelated intralogistics Use Cases because it ignored process and maturity constraints.
This example demonstrates how the updated logic improves contextual precision and relevance.

---

## Case Study: Score Divergence Analysis

**Assessment:** `2024-06-25 / artur.retkiewicz@mahle.com`

| Aspect | Old System | New System |
|:--|:--|:--|
| **Top Use Cases** | UC-12, UC-15, UC-11 | UC-21, UC-7, UC-15 |
| **Overlap** | 33% (UC-15 only) | — |
| **Process Filtering** | Ignored | Active |
| **Maturity Filtering** | Ignored | Strict (“Autonom”) |
| **Scoring Model** | Unknown (rule-based) | Additive weighted model (0–1 scale) |

**Interpretation:**
The old engine proposed diagnostizierende intralogistics Use Cases (UC-11, UC-12) despite the customer targeting autonomous planning.
The new algorithm enforces both maturity and process filters, returning UCs relevant to the requested maturity tier and process scope.
Score differences arise from normalization (0–1 range) and additive weighting.
The 33% overlap confirms consistent top-tier Use Cases while reflecting improved contextual precision.

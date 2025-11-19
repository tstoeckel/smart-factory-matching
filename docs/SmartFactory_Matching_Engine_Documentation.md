# Smart Factory Matching Engine Documentation (Updated)

The **Smart Factory Matching Engine** helps manufacturing companies identify the most relevant digitalization Use Cases based on their operational problems, maturity goals, and strategic impact priorities.
It translates qualitative assessment data into ranked Use Case recommendations using semantic matching, contextual filtering, and quantitative weighting.

---

## Overview

The engine compares customer assessments with a library of Smart Factory Use Cases.
Each assessment describes the customer's challenges, target maturity, and strategic focus.
The engine scores all Use Cases, applies process and maturity filters, and produces a ranked recommendation list.

**Goal:**
To provide consistent, explainable, and data-driven Use Case recommendations aligned with a company's maturity and strategic priorities.

---

## Input and Output

**Input Files**
- `assessment_db.csv`:  Customer assessments containing processes, problem statements, maturity levels, and impact preferences.
- `uc_problems_db.json`: Master Use Case library including Use Case metadata, processes, maturity levels, impact tags, and problem texts.

**Output Files**
- `assessment_db.csv`: Updated with a new column `MATCHES_SCORED` containing the top-ranked Use Cases and their scores.
- Google Sheets: Export functionality allows writing results directly to Google Sheets for reporting and collaboration.
- Console output for inspection (`--inspect`) and comparison (`--compare`) modes.

---

## Example Input/Output Pair

**Input Row (from assessment_db.csv):**
CREATED_AT,EMAIL,PROCESSES,PROBLEM_TEXTS,MATURITY_LEVELS,IMPACT_PRIORITIES
2024-06-25 10:44:48,artur.retkiewicz@mahle.com,"Produktionsplanung; Prozessoptimierung","lange lieferzeiten; manuelle planung","Autonom","Kosten: 10; Liefertreue: 8"

text

**Output Row (after matching):**
2024-06-25 10:44:48,artur.retkiewicz@mahle.com,"Produktionsplanung; Prozessoptimierung","lange lieferzeiten; manuelle planung","Autonom","Kosten: 10; Liefertreue: 8","UC-21: 0.461; UC-7: 0.419; UC-15: 0.386"

text

---

## Project Files

**assessment_db.csv**  
Main database of assessments and match results. Each row represents a single customer assessment, including processes, problem statements, impact preferences, and the resulting top Use Cases.

**uc_problems_db.json**  
Master library of Smart Factory Use Cases. Contains metadata such as Use Case names, associated processes, maturity levels, impact categories, and problem text definitions.

**matching_engine.py**  
Python script implementing the matching engine. Handles Use Case matching logic, scoring and weighting, peer relation analysis, Google Sheets import/export, inspection mode (`--inspect`), and comparison mode (`--compare`).

---

## Structure of `assessment_db.csv`

- **CREATED_AT**: Timestamp in `YYYY-MM-DD HH:MM:SS` format (primary key with EMAIL).
- **EMAIL**: Customer email address.
- **PROCESSES**: Semicolon-separated list of process steps (e.g. "Produktionsplanung; Qualitätsmanagement").
- **PROBLEM_TEXTS**: List of normalized problem statements.
- **MATURITY_LEVELS**: Customer-selected target maturity levels.
- **IMPACT_PRIORITIES**: Customer strategic impact priorities with weights.
- **MATCHES_SCORED**: Computed top 10 Use Cases with scores.

---

## Matching Logic

Each Use Case is scored against customer assessments by four weighted factors:

### Scoring Formula

final_score = 0.50 * base + 0.2 * impact + 0.1 * maturity + 0.20 * process

text

| Weight | Factor   | Description                                                            |
|-------:|----------|------------------------------------------------------------------------|
| 0.50   | Base     | Problem text similarity between assessment and Use Case.              |
| 0.20   | Impact   | Alignment with customer's impact priorities.                          |
| 0.10   | Maturity | Fit of Use Case maturity to customer's selected maturity.              |
| 0.20   | Process  | Relevance of Use Case’s process domain to customer's selected processes.|

---

## Dimension Calculations

- **Base (Problem Similarity)**: Uses token overlap and Levenshtein distance for similarity between customer and Use Case problems.
- **Impact**: Compares customer impact weights with Use Case impact tags to score alignment.
- **Maturity**: Adjusted via peer relations, supports bottom-up or top-down weighting.
- **Process**: Strict filter applied if processes specified; otherwise global down-weighting.

---

## Process Filtering

- Use Cases must overlap with customer-selected processes to be included (if any specified).
- Without a process filter, all Use Cases are eligible with reduced process weight (`0.8`).
- “Digitale Datenbasis” Use Cases get special weighting to avoid dominance yet remain relevant.

---

## Digital Base Use Cases ("Digitale Datenbasis")

Foundational Use Cases for digital infrastructure:

- Neutral or reduced weighting depending on process filter status.
- Receive full weight if explicitly selected by customer.

---

## Maturity Filtering

- Hard filter on maturity levels per customer selection.
- Peer relations adjust weighting to avoid premature over-rankings.

---

## Impact Weighting

- Customer impact priorities set weights (not filters).
- Alignment score ranges **0.1 to 1.0**, affecting up to 15% of final score.

---

## Peer Relations and Modes

- Peers are Use Cases with similar problems/processes.
- Maturity weighting modified based on peer maturity:
  - `bottom-up`: favors foundational Use Cases
  - `top-down`: favors advanced Use Cases

---

## Command-Line Interface (CLI)

- Default: batch match all assessments needing scoring.
- `--recompute-all`: Force recompute for all assessments.
- `--import-sheet`: Import assessments from Google Sheets.
- `--export-to-report`: Export all matches to Google Sheet.
- `--export-in-batches`: Batch export with controls for API quota. --export-in-batches --batch-size 100 --batch-delay 10
- `--inspect <timestamp> <email>`: Show detailed single assessment scoring.
- `--compare <timestamp> <email>`: Compare old vs. new results for one assessment.
- `--aggregate-top20`: Show aggregate top 20 Use Cases from all assessments.

---

## process for Report
- `--import-sheet`: Import assessments from Google Sheets.
- `--recompute-all`: Force recompute for all assessments. 
- `--export-in-batches`: Batch export with controls for API quota. --export-in-batches --batch-size 100 --batch-delay 5

## Google Sheets Integration

- Seamless import/export with Google Sheets using API.
- Handles deduplication and batch processing to respect API limits.
- Configurable batch size and delay for exports.

---

## Performance Optimizations

- Token-based candidate filtering.
- Caching Levenshtein similarity calculations.
- Early exit for near-perfect matches.
- Batch export with progress and checkpointing.

---

## Summary

The Smart Factory Matching Engine delivers transparent, scalable, and effective Use Case recommendations aligned to customer priorities and maturity, ready for industrial deployment with integrated reporting and analysis.

---

**End of Document**
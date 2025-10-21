# Smart Factory Use Case Matching Engine – Documentation

## 1. Overview

The Smart Factory Matching Engine automatically recommends relevant
Smart Factory Use Cases based on customer assessments. It evaluates
problem statements, strategic priorities, process selections, and
maturity levels, then identifies matching Use Cases from a structured
knowledge base.

## 2. Project Files

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>File</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>assessment_db.csv</td>
<td>Main database of assessments and match results. Each row represents
a single customer assessment, including processes, problems, impacts,
and resulting top Use Cases.</td>
</tr>
<tr>
<td>uc_problems_db.json</td>
<td>Master library of Smart Factory Use Cases, including metadata such
as name, processes, maturity level, impacts, and problem texts.</td>
</tr>
<tr>
<td>matching_engine.py</td>
<td>Python script implementing the matching engine, handling matching
logic, weighting, peer relations, inspection, and comparison modes.</td>
</tr>
</tbody>
</table>

## 3. Structure of assessment\_db.csv

This CSV file stores all assessments collected from customers. Columns:

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Column</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>CREATED_AT</td>
<td>Timestamp (YYYY-MM-DD HH:MM:SS). Together with EMAIL, forms the
primary key.</td>
</tr>
<tr>
<td>EMAIL</td>
<td>Email address identifying the customer.</td>
</tr>
<tr>
<td>PROCESSES</td>
<td>Semicolon-separated list of process steps selected by the customer
(e.g. 'Produktionsplanung; Qualitätsmanagement').</td>
</tr>
<tr>
<td>PROBLEM_TEXTS</td>
<td>Normalized list of problem statements describing operational
challenges.</td>
</tr>
<tr>
<td>MATURITY_LEVELS</td>
<td>Desired maturity levels selected by the customer (e.g.
'Diagnostizierend; Vorhersagend; Autonom').</td>
</tr>
<tr>
<td>IMPACT_PRIORITIES</td>
<td>Strategic impact focus of the customer (e.g. 'Kosten: 8; Qualität:
6; Liefertreue: 5').</td>
</tr>
<tr>
<td>OLD_MATCHES_SCORED</td>
<td>Historical top-10 Use Cases with their old algorithm scores.</td>
</tr>
<tr>
<td>MATCHES_SCORED</td>
<td>New top-10 Use Cases computed by the current matching engine with
additive scoring model.</td>
</tr>
</tbody>
</table>

## 4. Structure of uc\_problems\_db.json

This JSON file defines all Smart Factory Use Cases. Each Use Case entry
contains metadata used for matching and scoring. Example structure:

```json
{
    "id": 6,
    "name": {
        "de": "Bedarfsvorhersage",
        "en": "Demand Forecasting"
    },
    "processes": [
        { "label": { "de": "Bedarfsmanagement", "en": "Demand Management" } },
        { "label": { "de": "Produktionsplanung", "en": "Production Planning" } }
    ],
    "maturity_level": { "label": { "de": "Vorhersagend", "en": "Predictive" } },
    "impact": [
        { "label": { "de": "Kosten", "en": "Cost" }, "value": 8 },
        { "label": { "de": "Qualität", "en": "Quality" }, "value": 5 },
        { "label": { "de": "Liefertreue", "en": "Delivery Reliability" }, "value": 7 }
    ],
    "problems_tackled": [
        {
            "problem_text": {
                "de": "Nachfrageschwankungen führen zu ineffizienter Planung",
                "en": "Demand fluctuations cause inefficient planning"
            }
        },
        {
            "problem_text": {
                "de": "Manuelle Anpassung der Bedarfsplanung",
                "en": "Manual adjustment of demand plans"
            }
        }
    ]
}
```

The JSON typically contains 30–50 Use Cases. Each entry contributes
problem texts for semantic matching, impact values for strategic
weighting, and maturity/process tags for structural relevance.

## 5. Matching Logic

1. Normalize problem statements for both assessment and Use Cases.

2. Perform soft matching via token overlap and sentence prefix
    similarity.

3. Build peer relations among Use Cases in the same process based
    on ≥30% Jaccard overlap.

4. Compute maturity weights (bottom-up: simpler UCs are preferred).

5. Compute impact weights aligning UC impact dimensions with
    customer strategic priorities (0.1–1.0 scale).

6. Filter Use Cases by selected processes if provided.

7. Compute additive final score: 0.5×Base + 0.25×Impact +
    0.15×Maturity + 0.10×Process.

8. Rank and output top 10 Use Cases per assessment to
    MATCHES\_SCORED.

## 6. Script Usage

Run the matching engine with different modes:

• Batch recomputation: python matching\_engine.py — recomputes missing
MATCHES\_SCORED or all (--recompute-all).

• Inspect mode: python matching\_engine.py --inspect '&lt;timestamp&gt;'
'&lt;email&gt;' — runs detailed match analysis for one assessment.

• Comparison mode: python matching\_engine.py --compare
'&lt;timestamp&gt;' '&lt;email&gt;' — compares old and new results with
visualization.

## 7. Scoring Weights

Final scoring weights:

• Base (problem similarity): 50%

• Impact alignment: 25%

• Maturity alignment: 15%

• Process relevance: 10%

The matching engine applies a strict process filter when customers
specify one or more process steps in their assessment (e.g.,
'Produktionsplanung', 'Instandhaltung', 'Qualitätsmanagement'). Only Use
Cases belonging to the selected processes are considered for scoring.
This ensures contextual relevance and prevents unrelated Use Cases
(e.g., logistics or maintenance) from appearing when the user focuses on
planning or quality. If no process filter is provided, the engine
includes all Use Cases but slightly reduces their process relevance
score through the compute\_process\_spread\_weight() function. Together
with maturity and impact filters, process filtering forms one of the
three key constraints defining which Use Cases enter the scoring stage.

## Process Filtering

## Maturity Level Filtering

The matching engine now includes a direct maturity-level filtering
feature. When a customer selects a target maturity level (e.g.,
'Autonom', 'Vorhersagend'), the engine restricts its recommendations to
Use Cases at that level. This ensures that the results align with the
customer's intended technological or process maturity tier. Optionally,
a fuzzy filter can include Use Cases one level above or below the target
tier for broader inspiration.

This replaces the earlier implicit maturity weighting approach. Now, the
maturity parameter acts as a strict filter instead of only adjusting the
score.

## Enhanced Compare Mode

The \`--compare\` function now prints all active filters for an
assessment — including process, impact, and maturity filters — and
displays each Use Case's actual maturity level directly in the
comparison table. This provides a transparent view of how the filtering
logic influenced the final rankings.

The previous rank-shift chart visualization has been removed. The
console output now focuses on a clean, readable table that shows ranks,
scores, and maturity levels side by side.

## Example: Maturity Filtering in Action

Example: Assessment 2024-06-25 / <artur.retkiewicz@mahle.com>

Customer setup:

- Processes: Bedarfsmanagement, Supply Chain Management & Beschaffung, Produktionsplanung, Prozessoptimierung
- Maturity level: Autonom
- Impact priorities: Kosten: 10; Liefertreue: 8

Old results:

- UC-12, UC-15, UC-11 (Autonome Intralogistik & Bestandsführung)

New results:

- UC-21, UC-7, UC-15 (Autonome Produktionsplanung, Autonome
Bedarfsprognose, Autonomes SCM)
- Overlap: 33%

Interpretation:

- With maturity filtering active, only advanced Use Cases are selected
- The new engine delivers results consistent with both process and
maturity filters, while the old algorithm produced unrelated
intralogistics matches. This demonstrates how the updated logic improves contextual precision and relevance.

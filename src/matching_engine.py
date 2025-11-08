"""
Smart Factory Matching Engine – Patched Version

This script computes Use Case matches for Smart Factory assessments based on problem similarity,
impact priorities, maturity levels, and process filters. It also provides batch processing,
overlap analysis, single-inspect, and old-vs-new comparison with UC names.
"""

import pandas as pd
import json
import re
import argparse

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ASSESSMENT_DB = "data/assessment_db.csv"
USECASE_DB = "data/uc_problems_db.json"
OVERLAP_THRESHOLD = 0.3
MATURITY_MODE = "bottom_up"  # or 'top_down'
TOP_LIMIT = 10

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def prepare_assessment_problems(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return [t.strip().lower() for t in text.split(";") if t.strip()]

def parse_customer_impact(text):
    if not isinstance(text, str) or not text.strip():
        return {}
    pairs = re.findall(r"([A-Za-zäöüÄÖÜß ]+):\s*([\d.]+)", text)
    return {k.strip().lower(): float(v) for k, v in pairs}

def get_uc_processes(uc):
    processes = []
    for p in uc.get("processes", []):
        label = p.get("label", {}).get("de", "").lower()
        if label:
            processes.append(label)
    return processes

def is_base_uc(uc):
    # Detects Digitale Datenbasis via maturity level label (German)
    label = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
    return "digitale datenbasis" in label

def build_peer_relations(usecases, overlap_threshold=0.3):
    uc_problem_sets = []
    for uc in usecases:
        problems = set()
        for p in uc.get("problems_tackled", []):
            de = p.get("problem_text", {}).get("de", "")
            en = p.get("problem_text", {}).get("en", "")
            if de:
                problems.add(de.lower())
            if en:
                problems.add(en.lower())
        uc_problem_sets.append(problems)

    peer_relations = {}
    for i, uc_i in enumerate(usecases):
        peers = []
        for j, uc_j in enumerate(usecases):
            if i == j:
                continue
            overlap = len(uc_problem_sets[i] & uc_problem_sets[j]) / max(
                len(uc_problem_sets[i] | uc_problem_sets[j]), 1
            )
            if overlap >= overlap_threshold:
                peers.append(j)
        peer_relations[i] = peers
    return peer_relations, uc_problem_sets

# ---------------------------------------------------------------------
# Core scoring components
# ---------------------------------------------------------------------

def compute_maturity_weight(uc, peer_relations, usecases, debug=False):
    uc_id = uc.get("id")

    base_rank = uc.get("maturity_rank", 3)

    peers = peer_relations.get(uc_id, [])
    if not peers:
        return 1.0

    peer_ranks = [usecases[p].get("maturity_rank", 3) for p in peers]
    min_peer_rank = min(peer_ranks) if peer_ranks else base_rank
    level_diff = base_rank - min_peer_rank

    factor = 1.0
    if MATURITY_MODE == "bottom_up":
        factor = 1.0 + (-0.2 * level_diff)
    elif MATURITY_MODE == "top_down":
        factor = 1.0 + (0.2 * level_diff)

    if debug:
        print(f"[DEBUG] UC-{uc_id} | diff={level_diff} | factor={factor:.2f}")

    return max(0.6, min(1.4, factor))

def compute_impact_weight(uc_impact, customer_impact):
    if not customer_impact or not uc_impact:
        return 1.0
    uc_imp = {i.get("label", {}).get("de", "").lower(): i.get("value", 0) for i in uc_impact}
    num, den = 0.0, 0.0
    for k, v in customer_impact.items():
        if k in uc_imp:
            num += uc_imp[k] * v
        den += v
    if den == 0:
        return 1.0
    return max(0.1, min(1.0, num / (10 * den)))

def compute_process_spread_weight(uc, customer_processes):
    """
    Documented behavior:
    - Base UCs (Digitale Datenbasis): neutral when no process filter (1.0).
    When a filter is active, reduced (0.8) unless explicitly selected (1.0).
    - Non-base UCs: 1.0 on overlap; 0.8 otherwise. With strict filtering, overlap is guaranteed
    for selected processes, but keep the branch for completeness.
    """
    if not customer_processes:
        return 1.0 if is_base_uc(uc) else 0.8
    uc_procs = get_uc_processes(uc)
    selected = set([p.lower() for p in customer_processes])
    overlap = bool(set(uc_procs) & selected)
    if is_base_uc(uc):
        return 1.0 if overlap else 0.8
    return 1.0 if overlap else 0.8

# ---------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------

def match_assessment(
    problems,
    customer_impact,
    usecases,
    uc_problem_sets,
    peer_relations,
    customer_processes=None,
    customer_maturity_levels=None,
    debug=False,
):
    results = []
    for uc, uc_problems in zip(usecases, uc_problem_sets):
        if customer_processes and not any(
            proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
        ):
            continue
        if customer_maturity_levels:
            uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
            if not any(level.lower() in uc_level for level in customer_maturity_levels):
                continue

        base = sum(1 for p in problems if any(p in u for u in uc_problems)) / max(len(problems), 1)
        maturity_weight = compute_maturity_weight(uc, peer_relations, usecases, debug)
        impact_weight = compute_impact_weight(uc.get("impact", []), customer_impact)
        process_weight = compute_process_spread_weight(uc, customer_processes)

        final = 0.5 * base + 0.25 * impact_weight + 0.15 * maturity_weight + 0.1 * process_weight

        results.append(
            {
                "use_case_id": uc["id"],
                "use_case_name_de": uc.get("name", {}).get("de", ""),
                "score": round(final, 3),
                "base": base,
                "impact": impact_weight,
                "maturity": maturity_weight,
                "process": process_weight,
            }
        )
    return sorted(results, key=lambda x: x["score"], reverse=True)

# ---------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------

def compute_overlap_stats(df):
    def extract_uc_ids(cell):
        if pd.isna(cell):
            return []
        return re.findall(r"UC-\d+", str(cell))

    overlaps = []
    for _, row in df.iterrows():
        old_ids = extract_uc_ids(row.get("OLD_MATCHES_SCORED", ""))
        new_ids = extract_uc_ids(row.get("MATCHES_SCORED", ""))
        if not old_ids or not new_ids:
            overlaps.append((None, None, "", "", ""))
            continue
        common = sorted(set(old_ids) & set(new_ids))
        old_only = sorted(set(old_ids) - set(new_ids))
        new_only = sorted(set(new_ids) - set(old_ids))
        overlaps.append(
            (
                len(common),
                round(len(common) / len(old_ids) * 100, 1),
                ";".join(common),
                ";".join(old_only),
                ";".join(new_only),
            )
        )

    df["OVERLAP_COUNT"], df["OVERLAP_PERCENT"], df["COMMON_UCS"], df["OLD_ONLY"], df["NEW_ONLY"] = zip(*overlaps)
    return df

# ---------------------------------------------------------------------
# Median utility (for aggregate_top20_usecases_with_rank)
# ---------------------------------------------------------------------

def median(lst):
    if not lst:
        return float('nan')
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    else:
        return s[mid]

# ---------------------------------------------------------------------
# Aggregate top20 use cases with ranks
# ---------------------------------------------------------------------

def aggregate_top20_usecases_with_rank(df):
    def parse_scores(text):
        if not isinstance(text, str):
            return {}
        pairs = re.findall(r"UC-(\d+):\s*([\d.]+)", text)
        return {"UC-"+k: float(v) for k,v in pairs}

    all_old_scores = []
    all_new_scores = []

    for _, row in df.iterrows():
        all_old_scores.append(parse_scores(row.get("OLD_MATCHES_SCORED", "")))
        all_new_scores.append(parse_scores(row.get("MATCHES_SCORED", "")))

    ucs = set()
    for d in all_old_scores + all_new_scores:
        ucs.update(d.keys())

    data = []
    for uc in sorted(ucs):
        old_vals = [d.get(uc) for d in all_old_scores if uc in d]
        new_vals = [d.get(uc) for d in all_new_scores if uc in d]
        old_median = median(old_vals)
        new_median = median(new_vals)
        old_count = len(old_vals)
        new_count = len(new_vals)
        data.append([uc, old_median, new_median, old_count, new_count])

    with open('data/uc_problems_db.json', 'r', encoding='utf-8') as f:
        usecases = json.load(f)

    uc_name_map = {f"UC-{uc['id']}": uc.get('name', {}).get('de', '') for uc in usecases}
    uc_maturity_map = {f"UC-{uc['id']}": uc.get('maturity_level', {}).get('label', {}).get('de', '') for uc in usecases}

    df_agg = pd.DataFrame(data, columns=['UC', 'Old_Score', 'New_Score', 'Old_Count', 'New_Count'])
    df_agg['Use_Case_Name'] = df_agg['UC'].map(uc_name_map)
    df_agg['Maturity_Level'] = df_agg['UC'].map(uc_maturity_map)

    # Rank by counts
    df_agg['Old_Rank'] = df_agg['Old_Count'].rank(method='dense', ascending=False).astype(int)
    df_agg['New_Rank'] = df_agg['New_Count'].rank(method='dense', ascending=False).astype(int)

    # Sort by count to get top 20 for old and new
    df_top20_new = df_agg.sort_values(by='New_Count', ascending=False).head(20).reset_index(drop=True)
    df_top20_old = df_agg.sort_values(by='Old_Count', ascending=False).head(20).reset_index(drop=True)

    print('\n[Aggregate] Top 20 Use Cases sorted by New Count:')
    print(f" {'UC':<7} {'Use_Case_Name':<40} {'Maturity_Level':<20} {'New_Rank':<8} {'Old_Rank':<8} {'Old_Count':<10} {'New_Count':<10} {'Old_Score':<10} {'New_Score':<10}")
    for _, row in df_top20_new.iterrows():
        print(f" {row['UC']:<7} {row['Use_Case_Name'][:39]:<40} {row['Maturity_Level'][:19]:<20} {row['New_Rank']:<8} {row['Old_Rank']:<8} {row['Old_Count']:<10} {row['New_Count']:<10} {row['Old_Score']:<10.3f} {row['New_Score']:<10.3f}")

    print('\n[Aggregate] Top 20 Use Cases sorted by Old Count:')
    print(f" {'UC':<7} {'Use_Case_Name':<40} {'Maturity_Level':<20} {'Old_Rank':<8} {'New_Rank':<8} {'Old_Count':<10} {'New_Count':<10} {'Old_Score':<10} {'New_Score':<10}")
    for _, row in df_top20_old.iterrows():
        print(f" {row['UC']:<7} {row['Use_Case_Name'][:39]:<40} {row['Maturity_Level'][:19]:<20} {row['Old_Rank']:<8} {row['New_Rank']:<8} {row['Old_Count']:<10} {row['New_Count']:<10} {row['Old_Score']:<10.3f} {row['New_Score']:<10.3f}")

# ---------------------------------------------------------------------
# Main CLI interface
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Factory Matching Engine CLI")
    parser.add_argument("--recompute-all", action="store_true", help="Recompute all MATCHES_SCORED values.")
    parser.add_argument("--inspect", nargs=2, metavar=("TIMESTAMP", "EMAIL"), help="Inspect a single assessment in detail.")
    parser.add_argument("--compare", nargs=2, metavar=("TIMESTAMP", "EMAIL"), help="Compare old vs new results for one assessment.")
    parser.add_argument("--aggregate-top20", action="store_true", help="Aggregate and display top 20 use cases over all assessments")
    args = parser.parse_args()

    if args.inspect:
        ts, em = args.inspect
        # Call your inspect_single_assessment(ts, em) function here if implemented
        print("Inspect feature not implemented in this snippet.")
    elif args.compare:
        ts, em = args.compare
        # Call your compare_old_new(ts, em) function here if implemented
        print("Compare feature not implemented in this snippet.")
    elif args.aggregate_top20:
        df = pd.read_csv('data/assessment_db.csv', encoding='utf-8-sig')
        aggregate_top20_usecases_with_rank(df)
    else:
        # Call your run_batch_matching() function here if implemented
        print("Batch matching feature not implemented in this snippet.")

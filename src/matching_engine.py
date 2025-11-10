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
    # peer_relations keys are zero-based indices; uc ids are assumed consecutive from 0
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

    # Patched: removed incorrect process-label penalty related to "digitale datenbasis"
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

        # Strict process filtering (doc-consistent)
        #if customer_processes and not any(
        #    proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
        #):
        #    continue

        # Strict maturity-level filtering
        #if customer_maturity_levels:
        #    uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
        #    if not any(level.lower() in uc_level for level in customer_maturity_levels):
        #        continue


        # Patched combined filtering logic, // Include Use Case if it matches either process or maturity filter (not necessarily both)
        if customer_processes and customer_maturity_levels:
            proc_match = any(
                proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
            )
            uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
            maturity_match = any(level.lower() in uc_level for level in customer_maturity_levels)

            # Use Case berücksichtigen, wenn entweder Prozess oder Maturity passt
            if not (proc_match or maturity_match):
                continue
        elif customer_processes:
            # Nur Prozessfilter prüfen
            if not any(
                proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
            ):
                continue
        elif customer_maturity_levels:
            # Nur Maturity-Filter prüfen
            uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
            if not any(level.lower() in uc_level for level in customer_maturity_levels):
                continue



        base = sum(1 for p in problems if any(p in u for u in uc_problems)) / max(len(problems), 1)
        maturity_weight = compute_maturity_weight(uc, peer_relations, usecases, debug)
        impact_weight = compute_impact_weight(uc.get("impact", []), customer_impact)
        process_weight = compute_process_spread_weight(uc, customer_processes)


        final = 0.5 * base + 0.3 * impact_weight + 0.15 * maturity_weight + 0.05 * process_weight
        # Patched: previous weighting (commented out)
        #final = 0.5 * base + 0.25 * impact_weight + 0.15 * maturity_weight + 0.1 * process_weight
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
# Batch recomputation
# ---------------------------------------------------------------------

def run_batch_matching(recompute_all=False):
    print(f"\n[Batch] Starting matching (recompute_all={recompute_all})...")

    usecases = json.load(open(USECASE_DB, "r", encoding="utf-8"))
    df = pd.read_csv(ASSESSMENT_DB, encoding="utf-8-sig")

    peer_relations, uc_problem_sets = build_peer_relations(usecases, overlap_threshold=OVERLAP_THRESHOLD)
    print(f"[Batch] Peer relations built for {len(peer_relations)} Use Cases. Mode={MATURITY_MODE}")

    if "MATCHES_SCORED" not in df.columns:
        df["MATCHES_SCORED"] = None

    if recompute_all:
        target_idx = df.index
    else:
        target_idx = df[df["MATCHES_SCORED"].isna() | (df["MATCHES_SCORED"].astype(str).str.strip() == "")].index

    for i in target_idx:
        row = df.loc[i]
        problems = prepare_assessment_problems(row.get("PROBLEM_TEXTS", ""))
        customer_impact = parse_customer_impact(row.get("IMPACT_PRIORITIES", ""))
        raw_procs = str(row.get("PROCESSES", "")).strip()
        customer_processes = [p.strip() for p in raw_procs.split(";") if p.strip()] if raw_procs and raw_procs.lower() != "nan" else None
        customer_maturity_levels = [m.strip() for m in str(row.get("MATURITY_LEVELS", "")).split(";") if m.strip()]

        top_matches = match_assessment(
            problems,
            customer_impact,
            usecases,
            uc_problem_sets,
            peer_relations,
            customer_processes,
            customer_maturity_levels,
            debug=False,
        )
        formatted = "; ".join([
            f"UC-{r.get('use_case_id', '?')}: {r.get('score', 0):.3f}"
            for r in top_matches[:TOP_LIMIT]
        ])
        df.at[i, "MATCHES_SCORED"] = formatted

    df = compute_overlap_stats(df)
    df.to_csv(ASSESSMENT_DB, index=False, encoding="utf-8-sig")
    print(f"\n[Batch] ✅ Matching completed — results written to {ASSESSMENT_DB}")

# ---------------------------------------------------------------------
# Inspect single assessment
# ---------------------------------------------------------------------

def inspect_single_assessment(timestamp, email):
    usecases = json.load(open(USECASE_DB, "r", encoding="utf-8"))
    df = pd.read_csv(ASSESSMENT_DB, encoding="utf-8-sig")

    mask = (df["CREATED_AT"] == timestamp) & (df["EMAIL"].str.lower() == email.lower())
    if not mask.any():
        print("[Inspect] ❌ No matching assessment found for", timestamp, "/", email)
        return
    row = df.loc[mask].iloc[0]

    peer_relations, uc_problem_sets = build_peer_relations(usecases, overlap_threshold=OVERLAP_THRESHOLD)
    problems = prepare_assessment_problems(row.get("PROBLEM_TEXTS", ""))
    customer_impact = parse_customer_impact(row.get("IMPACT_PRIORITIES", ""))
    raw_procs = str(row.get("PROCESSES", "")).strip()
    customer_processes = [p.strip() for p in raw_procs.split(";") if p.strip()] if raw_procs and raw_procs.lower() != "nan" else None
    customer_maturity_levels = [m.strip() for m in str(row.get("MATURITY_LEVELS", "")).split(";") if m.strip()]

    print("[Inspect] Peer relations built for", len(peer_relations), "Use Cases. Mode=", MATURITY_MODE)
    print("Impact priorities:", customer_impact if customer_impact else "(none)")
    if customer_maturity_levels:
        print("Maturity level filter detected:", ", ".join(customer_maturity_levels))
    else:
        print("No maturity level filter applied.")

    top_matches = match_assessment(
        problems,
        customer_impact,
        usecases,
        uc_problem_sets,
        peer_relations,
        customer_processes,
        customer_maturity_levels,
        debug=True,
    )
    for r in top_matches[:TOP_LIMIT]:
        print(f"UC-{r['use_case_id']} ({r['use_case_name_de']}): Score={r['score']} (Base={r['base']:.3f}, Impact={r['impact']:.2f}, Mat={r['maturity']:.2f}, Proc={r['process']:.2f})")

# ---------------------------------------------------------------------
# Compare old vs new for one assessment (with UC names)
# ---------------------------------------------------------------------

def compare_old_new(timestamp, email):
    df = pd.read_csv(ASSESSMENT_DB, encoding="utf-8-sig")
    mask = (df["CREATED_AT"] == timestamp) & (df["EMAIL"].str.lower() == email.lower())
    if not mask.any():
        print("[Compare] ❌ Assessment not found.")
        return

    row = df.loc[mask].iloc[0]

    raw_procs = str(row.get("PROCESSES", "")).strip()
    processes = ([p.strip() for p in raw_procs.split(";") if p.strip()] if raw_procs and raw_procs.lower() != "nan" else [])
    if processes:
        print(f"[Compare] Process filter detected ({len(processes)}): {', '.join(processes)}")
    else:
        print("[Compare] No process filter applied.")

    raw_impacts = str(row.get("IMPACT_PRIORITIES", "")).strip()
    if raw_impacts and raw_impacts.lower() != "nan":
        pairs = re.findall(r"([A-Za-zäöüÄÖÜß ]+):\s*([\d.]+)", raw_impacts)
        if pairs:
            print("[Compare] Impact priorities provided:", "; ".join([f"{k}: {v}" for k, v in pairs]))
    else:
        print("[Compare] No impact preferences provided.")

    raw_maturity = str(row.get("MATURITY_LEVELS", "")).strip()
    if raw_maturity and raw_maturity.lower() != "nan":
        maturities = [m.strip() for m in raw_maturity.split(";") if m.strip()]
        print(f"[Compare] Maturity level filter detected ({len(maturities)}): {', '.join(maturities)}")
    else:
        print("[Compare] No maturity level filter applied.")

    problem_texts = row.get("PROBLEM_TEXTS", "").strip()
    if problem_texts and problem_texts.lower() != "nan":
        print("\n[Compare] Provided Problem Texts:")
        print(problem_texts.replace(';', ';\n'))

    def parse_uc_scores(text):
        if not isinstance(text, str):
            return {}
        cleaned = re.sub(r"[  ]+", " ", text).replace(";", " ; ")
        pairs = re.findall(r"(UC-\d+)\s*:\s*([\d.,]+)", cleaned)
        results = {}
        for uc, val in pairs:
            try:
                results[uc] = float(val.replace(",", "."))
            except ValueError:
                continue
        return results

    old_scores = parse_uc_scores(row.get("OLD_MATCHES_SCORED", ""))
    new_scores = parse_uc_scores(row.get("MATCHES_SCORED", ""))

    all_ucs = sorted(set(old_scores.keys()) | set(new_scores.keys()))
    data = []
    for uc in all_ucs:
        old_rank = list(old_scores.keys()).index(uc) + 1 if uc in old_scores else None
        new_rank = list(new_scores.keys()).index(uc) + 1 if uc in new_scores else None
        old_score = old_scores.get(uc)
        new_score = new_scores.get(uc)
        data.append([uc, old_rank, new_rank, old_score, new_score])

    # Load Use Case metadata for maturity and name lookup
    with open(USECASE_DB, "r", encoding="utf-8") as f:
        usecases = json.load(f)
    uc_maturity_map = {f"UC-{uc['id']}": uc.get("maturity_level", {}).get("label", {}).get("de", "") for uc in usecases}
    uc_name_map = {f"UC-{uc['id']}": uc.get("name", {}).get("de", "") for uc in usecases}

    df_compare = pd.DataFrame(data, columns=["UC", "Old_Rank", "New_Rank", "Old_Score", "New_Score"])
    df_compare["Maturity_Level"] = df_compare["UC"].map(uc_maturity_map)
    df_compare["Use_Case_Name"] = df_compare["UC"].map(uc_name_map)
    df_compare = df_compare[["UC", "Use_Case_Name", "Maturity_Level", "Old_Rank", "New_Rank", "Old_Score", "New_Score"]]

    print("\n[Compare] Rank comparison:")
    print(df_compare.to_string(index=False))

    overlap = len(set(old_scores.keys()) & set(new_scores.keys()))
    denom = max(len(old_scores), 1)
    print(f"\n[Compare] Overlap: {overlap} / {len(old_scores)} ({overlap / denom:.0%})")
    if overlap > 0:
        print(f"[Compare] ✅ {overlap} shared Use Case(s) found.")
    else:
        print("[Compare] ⚠️ No overlap between old and new matches.")



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
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Factory Matching Engine CLI")
    parser.add_argument("--recompute-all", action="store_true", help="Recompute all MATCHES_SCORED values.")
    parser.add_argument(
        "--inspect", nargs=2, metavar=("TIMESTAMP", "EMAIL"), help="Inspect a single assessment in detail."
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("TIMESTAMP", "EMAIL"), help="Compare old vs new results for one assessment."
    )
    parser.add_argument("--aggregate-top20", action="store_true", help="Aggregate and display top 20 use cases over all assessments")
    args = parser.parse_args()

    if args.inspect:
        ts, em = args.inspect
        inspect_single_assessment(ts, em)
    elif args.compare:
        ts, em = args.compare
        compare_old_new(ts, em)
    elif args.aggregate_top20:
        df = pd.read_csv('data/assessment_db.csv', encoding='utf-8-sig')
        aggregate_top20_usecases_with_rank(df)
    else:
        run_batch_matching(recompute_all=args.recompute_all)

"""
Smart Factory Matching Engine – Clean Version v8.0
Last updated: 2025-10-19

This script computes Use Case matches for Smart Factory assessments based on problem similarity,
impact priorities, maturity levels, and process filters. It also provides batch processing and overlap analysis.
"""

import pandas as pd
import json
import re
import argparse

ASSESSMENT_DB = "data/assessment_db.csv"
USECASE_DB = "data/uc_problems_db.json"

OVERLAP_THRESHOLD = 0.3
MATURITY_MODE = "bottom_up"  # or 'top_down'
TOP_LIMIT = 10


# ---------------------------------------------------------------------
# Helper functions
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
    return peer_relations


# ---------------------------------------------------------------------
# Core scoring components
# ---------------------------------------------------------------------


def compute_maturity_weight(uc, peer_relations, usecases, debug=False):
    uc_id = uc.get("id")
    base_rank = uc.get("maturity_rank", 3)
    peers = peer_relations.get(uc_id - 1, [])
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

    uc_procs = get_uc_processes(uc)
    if any("digitale datenbasis" in p for p in uc_procs):
        factor *= 0.9

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
    if not customer_processes:
        return 0.8
    uc_procs = get_uc_processes(uc)
    overlap = len(set(uc_procs) & set([p.lower() for p in customer_processes]))
    return 1.0 if overlap else 0.8


# ---------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------


def match_assessment(
    problems, customer_impact, usecases, uc_problem_sets, peer_relations, customer_processes=None, debug=False
):
    results = []
    for uc, uc_problems in zip(usecases, uc_problem_sets):
        if customer_processes and not any(
            proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
        ):
            continue

        base = sum(1 for p in problems if any(p in u for u in uc_problems)) / max(len(problems), 1)
        maturity_weight = compute_maturity_weight(uc, peer_relations, usecases, debug)
        impact_weight = compute_impact_weight(uc.get("impact", []), customer_impact)
        process_weight = compute_process_spread_weight(uc, customer_processes)

        final = 0.5 * base + 0.25 * impact_weight + 0.15 * maturity_weight + 0.10 * process_weight
        results.append(
            {
                "use_case_id": uc["id"],
                "use_case_name_de": uc["name"]["de"],
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
# Batch matching
# ---------------------------------------------------------------------


def run_batch_matching(recompute_all=False):
    print(f"\n[Batch] Starting matching (recompute_all={recompute_all})...")

    usecases = json.load(open(USECASE_DB, "r", encoding="utf-8"))
    df = pd.read_csv(ASSESSMENT_DB, encoding="utf-8-sig")

    uc_problem_sets = []
    for uc in usecases:
        ps = set()
        for p in uc.get("problems_tackled", []):
            de = p.get("problem_text", {}).get("de", "")
            en = p.get("problem_text", {}).get("en", "")
            if de:
                ps.add(de.lower())
            if en:
                ps.add(en.lower())
        uc_problem_sets.append(ps)

    peer_relations = build_peer_relations(usecases, overlap_threshold=OVERLAP_THRESHOLD)
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
        customer_processes = [p.strip() for p in str(row.get("PROCESSES", "")).split(";") if p.strip()] or None

        top_matches = match_assessment(
            problems, customer_impact, usecases, uc_problem_sets, peer_relations, customer_processes, debug=False
        )
        formatted = "; ".join([f"UC-{r['use_case_id']}: {r['score']}" for r in top_matches[:TOP_LIMIT]])
        df.at[i, "MATCHES_SCORED"] = formatted

    df = compute_overlap_stats(df)
    df.to_csv(ASSESSMENT_DB, index=False, encoding="utf-8-sig")
    print(f"\n[Batch] ✅ Matching completed — results written to {ASSESSMENT_DB}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Factory Matching Engine CLI")
    parser.add_argument("--recompute-all", action="store_true", help="Recompute all MATCHES_SCORED values.")
    args = parser.parse_args()

    run_batch_matching(recompute_all=args.recompute_all)

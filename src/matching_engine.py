"""
Smart Factory Matching Engine – Patched Version

This script computes Use Case matches for Smart Factory assessments based on problem similarity,
impact priorities, maturity levels, and process filters. It also provides batch processing,
overlap analysis, single-inspect, and old-vs-new comparison with UC names.
"""

import datetime
import pandas as pd
import json
import re
import argparse
import time
import sys
import math


# ---------------------------------------------------------------------
#Import Google Sheet Data
# ---------------------------------------------------------------------
import gspread
from google.oauth2.service_account import Credentials
from dateutil import parser

def to_datetime_safe(s):
    try:
        return parser.parse(str(s).strip())
    except Exception:
        return None

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ASSESSMENT_DB = "data/assessment_db.csv"
USECASE_DB = "data/uc_problems_db.json"
GOOGLE_SHEET_CREDENTIALS = "data/praxis-backup-478106-c1-7f3f6481cd6e.json"

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

def load_usecase_db(json_path):
    """
    Load the use case DB (JSON). Expects a list of dicts, each with 'id', 'name'{'de','en'}.
    Returns a mapping: key 'UCxx' → { 'de': ..., 'en': ... }
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    lut = {}
    for entry in raw:
        ucid = int(entry['id'])  # numeric value, e.g. 16
        key = f"UC{ucid:02d}"    # two-digit (e.g. UC16)
        lut[key] = {
            "de": entry['name']['de'],
            "en": entry['name']['en']
        }
    return lut

def parse_gsheet_datetime(val):
    if pd.isna(val):
        return val
    # Versuche, ISO-Format oder andere gängige Formate zu parsen
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    # Handle Google Sheet Seriennummer (numerischer Wert)
    try:
        base = datetime.datetime(1899, 12, 30)
        n = float(val)
        dt = base + datetime.timedelta(days=n)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    # Erwartetes deutsches Datumsformat (z.B. '22.10.2025 09:05:47')
    try:
        return datetime.datetime.strptime(str(val), "%d.%m.%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    # Fallback: Originalwert zurückgeben
    return val

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
    # Damit wird die Division nicht mehr durch den Faktor 10 zusätzlich skaliert, was die Werte im Bereich [0.1, 1.0] feiner verteilt und das Matching sensibler macht.
    #return max(0.1, min(1.0, num / (10 * den)))
    return max(0.1, min(1.0, num / den))
    #return max(0.1, min(1.0, (num / den) ** 1.5))  # War 1.2



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
# Fuzzy Matching Helper Functions (NO EXTERNAL DEPENDENCIES)
# ---------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    """Berechnet Edit-Distance zwischen zwei Strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def string_similarity(s1, s2):
    """Berechnet Ähnlichkeit 0-1 basierend auf Levenshtein-Distance"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1 - (distance / max_len)

def problem_matches_optimized(problem, uc_problem, token_low=0.3, token_high=0.7, sim_thresh=0.62):
    p = problem.lower().strip()
    u = uc_problem.lower().strip()
    if not p or not u:
        return False
    if p in u or u in p:
        return True
    p_words = set(p.split())
    u_words = set(u.split())
    if not p_words or not u_words:
        return False
    overlap = len(p_words & u_words) / max(len(p_words), len(u_words))
    if overlap >= token_high:
        return True
    if overlap < token_low:
        return False
    # only now expensive similarity
    return string_similarity(p, u) >= sim_thresh

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
    uc_problem_lists = [[s.lower().strip() for s in probs if s and str(s).strip()] for probs in uc_problem_sets]
    for uc, uc_problems in zip(usecases, uc_problem_lists):

        # Patched combined filtering logic
        if customer_processes and customer_maturity_levels:
            proc_match = any(
                proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
            )
            uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
            maturity_match = any(level.lower() in uc_level for level in customer_maturity_levels)

            if not (proc_match or maturity_match):
                continue
        elif customer_processes:
            if not any(
                proc in get_uc_processes(uc) for proc in [p.lower() for p in customer_processes]
            ):
                continue
        elif customer_maturity_levels:
            uc_level = uc.get("maturity_level", {}).get("label", {}).get("de", "").lower()
            if not any(level.lower() in uc_level for level in customer_maturity_levels):
                continue

        # FUZZY MATCHING statt exaktes Substring-Matching
        matched_count = 0
        for p in problems:
            for u in uc_problems:
                if problem_matches_optimized(p, u):
                    matched_count += 1
                    break

        base = matched_count / max(len(problems), 1)
        base_adj = base

        maturity_weight = compute_maturity_weight(uc, peer_relations, usecases, debug)
        impact_weight = compute_impact_weight(uc.get("impact", []), customer_impact)
        process_weight = compute_process_spread_weight(uc, customer_processes)

        final = 0.60 * base_adj + 0.25 * impact_weight + 0.10 * maturity_weight + 0.05 * process_weight

        results.append(
            {
                "use_case_id": uc["id"],
                "use_case_name_de": uc.get("name", {}).get("de", ""),
                "score": round(final, 3),
                "base_adj": base_adj,
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

def run_batch_matching(recompute_all=False, checkpoint_every=50, progress_every=10):
    import time, traceback
    print(f"\n[Batch] Starting matching (recompute_all={recompute_all})...")
    start_all = time.time()

    try:
        usecases = json.load(open(USECASE_DB, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[Batch] ❌ Failed to load USECASE_DB: {e}")
        traceback.print_exc()
        return

    try:
        df = pd.read_csv(ASSESSMENT_DB, encoding="utf-8-sig")
    except Exception as e:
        print(f"[Batch] ❌ Failed to load ASSESSMENT_DB: {e}")
        traceback.print_exc()
        return

    peer_relations, uc_problem_sets = build_peer_relations(usecases, overlap_threshold=OVERLAP_THRESHOLD)
    # normalize uc_problem_sets -> list of lists for deterministic iteration and speed
    uc_problem_lists = [list(s) for s in uc_problem_sets]

    print(f"[Batch] Peer relations built for {len(peer_relations)} Use Cases. Mode={MATURITY_MODE}")

    if "MATCHES_SCORED" not in df.columns:
        df["MATCHES_SCORED"] = None

    if recompute_all:
        target_idx = list(df.index)
    else:
        mask_missing = df["MATCHES_SCORED"].isna() | (df["MATCHES_SCORED"].astype(str).str.strip() == "")
        target_idx = list(df[mask_missing].index)

    total_to_process = len(target_idx)
    print(f"[Batch] Rows to process: {total_to_process}")

    if total_to_process == 0:
        print("[Batch] Nothing to do.")
        return

    processed = 0
    t0 = time.time()
    for count, i in enumerate(target_idx, start=1):
        row_start = time.time()
        try:
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
                uc_problem_lists,    # use precomputed lists
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
            processed += 1

        except Exception as e:
            print(f"[Batch] ⚠️ Error processing row {i}: {e}")
            traceback.print_exc()
            # keep going

        # progress output and timing
        row_elapsed = time.time() - row_start
        if count % progress_every == 0 or count == 1 or count == total_to_process:
            avg = (time.time() - t0) / max(1, count)
            remaining = (total_to_process - count) * avg
            print(f"[Batch] {count}/{total_to_process} processed, last_row={row_elapsed:.2f}s, avg_per_row={avg:.2f}s, est_remaining={remaining/60:.1f}min")
            sys.stdout.flush()

        # checkpoint save
        if count % checkpoint_every == 0:
            try:
                df = compute_overlap_stats(df)
                df.to_csv(ASSESSMENT_DB, index=False, encoding="utf-8-sig")
                print(f"[Batch] checkpoint: wrote progress to {ASSESSMENT_DB} after {count} rows")
            except Exception as e:
                print(f"[Batch] ⚠️ Failed to write checkpoint: {e}")

    # final save
    df = compute_overlap_stats(df)
    df.to_csv(ASSESSMENT_DB, index=False, encoding="utf-8-sig")
    total_time = time.time() - start_all
    print(f"\n[Batch] ✅ Matching completed — {processed} rows processed in {total_time:.1f}s ({total_time/60:.1f}min).")

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
# import_from_Google sheet all new assessment data
# ---------------------------------------------------------------------

def import_from_sheet(sheet_id, ASSESSMENT_DB = "data/assessment_db.csv", GOOGLE_SHEET_CREDENTIALS = "data/praxis-backup-478106-c1-7f3f6481cd6e.json"):
    # Authentifizierung und Google Sheet Zugriff
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(GOOGLE_SHEET_CREDENTIALS, scopes=scopes)
    client = gspread.authorize(creds)
    
    worksheet = client.open_by_key(sheet_id).worksheet("Master")
    data = worksheet.get_all_records()  # Daten ab Zeile 2 als Dicts
    df_sheet = pd.DataFrame(data)

    mapping = {
        "Zeitstempel": "CREATED_AT",
        "E-Mail-Adresse": "EMAIL",
        "Q2_2 In diesen Prozessschritten suchen wir nach Use Cases:": "PROCESSES",
        "Q2_3 Folgende Problemstellung möchten wir lösen:": "PROBLEM_TEXTS",
        "Q2_5 Auf diesem Reifegrad / Entwicklungsstufen suchen wir Use Cases:": "MATURITY_LEVELS",
        "Q2_6 Folgende strategische Prioritäten wollen wir mit dem Use Case verfolgen:Pro Kategorie bitte nur eine Option auswählen.": "IMPACT_PRIORITIES"
    }
    df_sheet.rename(columns=mapping, inplace=True)
    relevant = list(mapping.values())
    df_sheet = df_sheet[relevant]
    df_sheet['CREATED_AT'] = df_sheet['CREATED_AT'].apply(parse_gsheet_datetime)


    # Neu: Verarbeitung von MATURITY_LEVEL, nur Text vor ":" übernehmen
    def extract_maturity_level(value):
        if pd.isna(value):
            return value
        parts = value.split(';')  # An Semikolon trennen
        # Für jedes Segment nur den Teil vor ':' extrahieren, trimmen
        clean_parts = [part.split(':')[0].strip() for part in parts if part.strip()]
        return "; ".join(clean_parts)  # Teilantworten wieder zusammensetzen
    
    df_sheet["MATURITY_LEVELS"] = df_sheet["MATURITY_LEVELS"].apply(extract_maturity_level)


    try:
        df_csv = pd.read_csv(ASSESSMENT_DB)
    except FileNotFoundError:
        df_csv = pd.DataFrame(columns=relevant)

    df_sheet['key'] = df_sheet['CREATED_AT'].astype(str) + '-' + df_sheet['EMAIL'].astype(str)
    if not df_csv.empty:
        df_csv['key'] = df_csv['CREATED_AT'].astype(str) + '-' + df_csv['EMAIL'].astype(str)
        new_rows = df_sheet[~df_sheet['key'].isin(df_csv['key'])]
    else:
        new_rows = df_sheet.copy()

    if not new_rows.empty:
        df_csv = pd.concat([df_csv, new_rows[relevant]], ignore_index=True)
        df_csv.drop(columns=['key'], errors='ignore', inplace=True)
        df_csv.to_csv(ASSESSMENT_DB, index=False)
        print(f"{len(new_rows)} new line(s) added.")
    else:
        print("no new lines in Google Sheet identified.")


# ---------------------------------------------------------------------
# Export Matchings to Google Sheet
# ---------------------------------------------------------------------
def export_to_report_csv(
    csv_path,
    uc_json_path,
    sheet_id,
    creds_path,
    worksheet_name
):
    usecase_db = load_usecase_db(uc_json_path)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)

    sheet_columns = ws.row_values(1)
    sheet_records = ws.get_all_records()

    # Use vollständigen CREATED_AT (mit Zeit) statt nur Datum
    existing_keys = set(
        (
            str(row.get("CREATED_AT", "")).strip(),
            str(row.get("EMAIL_ADRESSE_P2", "")).strip()
        )
        for row in sheet_records
    )

    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        created_at_raw = str(row["CREATED_AT"]).strip()
        email = str(row["EMAIL"]).strip()

        # Verwende den vollständigen Timestamp als Key
        key = (created_at_raw, email)

        if key in existing_keys:
            continue

        processes = row.get("PROCESSES", "")
        problem_texts = row.get("PROBLEM_TEXTS", "")
        maturity_level = row.get("MATURITY_LEVELS", "")
        impact_priorities = row.get("IMPACT_PRIORITIES", "")

        matches_scored_raw = str(row.get("MATCHES_SCORED", ""))
        if not matches_scored_raw:
            continue

        match_entries = [s.strip() for s in matches_scored_raw.split(";") if s.strip()]
        if not match_entries:
            continue

        for entry in match_entries[:10]:
            m = re.match(r"UC-?(\d+):\s*([\d.]+)", entry)
            if not m:
                continue

            num = int(m.group(1))
            score = m.group(2)

            uc_key = f"UC{num:02d}"
            uc_info = usecase_db.get(uc_key, {"de": "Unknown", "en": "Unknown"})

            name_y = f"UC{num:02d}: {uc_info['de']}"

            row_dict = {
                "CREATED_AT": created_at_raw,
                "SCORE": score,
                "FIRSTNAME_P1": uc_info["en"],
                "EMAIL_ADRESSE_P2": email,
                "Q1_0 Use Case Name:": name_y,
                "Q2_2 In diesen Prozessschritten suchen wir nach Use Cases:": processes,
                "Q2_3 Folgende Problemstellung möchten wir lösen:": problem_texts,
                "Q2_5 Auf diesem Reifegrad / Entwicklungsstufen suchen wir Use Cases:": maturity_level,
                "Q2_6 Folgende strategische Prioritäten wollen wir mit dem Use Case verfolgen:Pro Kategorie bitte nur eine Option auswählen.": impact_priorities,
            }

            values_for_insert = []
            for col in sheet_columns:
                val = row_dict.get(col, "")
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    val = ""
                values_for_insert.append(val)
            
            ws.append_row(values_for_insert)

        existing_keys.add(key)

    print("Export finished. Only existing columns were updated.")

# ---------------------------------------------------------------------
#  Export assessment_db.csv to Google Sheet in batches to avoid API quotas
# --export-in-batches --batch-size 100 --batch-delay 10
# ---------------------------------------------------------------------
def export_to_report_in_batches(
    csv_path,
    uc_json_path,
    sheet_id,
    creds_path,
    worksheet_name,
    batch_size=50,
    start_idx=0,
    delay_between_batches=60,
    delay_between_rows=0.5  # Reduziert von 2
):
    """
    Export assessment_db.csv to Google Sheet in batches to avoid API quotas.
    Uses append_rows() to write multiple rows at once instead of one per call.
    """

    usecase_db = load_usecase_db(uc_json_path)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)

    sheet_columns = ws.row_values(1)
    sheet_records = ws.get_all_records()
    
    existing_keys = set(
        (
            str(row.get("CREATED_AT", "")).strip(),
            str(row.get("EMAIL_ADRESSE_P2", "")).strip()
        )
        for row in sheet_records
    )

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    if start_idx >= total_rows:
        print("All rows processed.")
        return None

    end_idx = min(start_idx + batch_size, total_rows)
    batch_df = df.iloc[start_idx:end_idx]

    new_rows_count = 0
    rows_to_append = []  # Sammeln statt einzeln schreiben

    for idx, row in batch_df.iterrows():
        created_at_raw = str(row["CREATED_AT"]).strip()
        email = str(row["EMAIL"]).strip()
        
        key = (created_at_raw, email)

        if key in existing_keys:
            continue

        matches_scored_raw = str(row.get("MATCHES_SCORED", ""))
        if not matches_scored_raw:
            continue

        match_entries = [s.strip() for s in matches_scored_raw.split(";") if s.strip()]
        if not match_entries:
            continue

        processes = row.get("PROCESSES", "")
        problem_texts = row.get("PROBLEM_TEXTS", "")
        maturity_level = row.get("MATURITY_LEVELS", "")
        impact_priorities = row.get("IMPACT_PRIORITIES", "")

        for entry in match_entries[:10]:
            m = re.match(r"UC-?(\d+):\s*([\d.]+)", entry)
            if not m:
                continue
            num = int(m.group(1))
            score = m.group(2)
            uc_key = f"UC{num:02d}"
            uc_info = usecase_db.get(uc_key, {"de": "Unknown", "en": "Unknown"})
            name_y = f"{uc_key}: {uc_info['de']}"

            row_dict = {
                "CREATED_AT": created_at_raw,
                "SCORE": score,
                "FIRSTNAME_P1": uc_info["en"],
                "EMAIL_ADRESSE_P2": email,
                "Q1_0 Use Case Name:": name_y,
                "Q2_2 In diesen Prozessschritten suchen wir nach Use Cases:": processes,
                "Q2_3 Folgende Problemstellung möchten wir lösen:": problem_texts,
                "Q2_5 Auf diesem Reifegrad / Entwicklungsstufen suchen wir Use Cases:": maturity_level,
                "Q2_6 Folgende strategische Prioritäten wollen wir mit dem Use Case verfolgen:Pro Kategorie bitte nur eine Option auswählen.": impact_priorities,
            }

            values_for_insert = []
            for col in sheet_columns:
                val = row_dict.get(col, "")
                if isinstance(val, float) and math.isnan(val):
                    val = ""
                values_for_insert.append(val)

            rows_to_append.append(values_for_insert)
            new_rows_count += 1

        existing_keys.add(key)

    # Schreibe alle Rows auf einmal
    if rows_to_append:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ws.append_rows(rows_to_append)  # append_rows statt append_row
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 2)
                    print(f"[Batch] ⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
                    print(f"[Batch] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"[Batch] ❌ Failed after {max_retries} attempts.")
                    break

    print(f"[Batch] ✅ Exported {new_rows_count} new row(s) (rows {start_idx}-{end_idx}/{total_rows}).")
    
    if end_idx < total_rows:
        return end_idx
    return None
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
    parser.add_argument("--import-sheet", action="store_true", help="Import new data from Google Sheet into assessment_db.csv")
    parser.add_argument('--export-to-report', action='store_true', help='Export entire assessment DB to Google Sheet report (max 10 rows per assessment).')
    
    parser.add_argument("--export-in-batches", action="store_true", help="Export assessment DB to Google Sheet in batches to avoid API quota limits.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of rows to process per batch (default: 50).")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index to process from (default: 0).")
    parser.add_argument("--batch-delay", type=int, default=60, help="Delay in seconds between batch uploads (default: 60).")
    
    
    args = parser.parse_args()
    
    if args.import_sheet:
        SHEET_ID = "1RdIX9cIavqQXR2XPvW7LtlfgiR247Kvfuc8ip-cVmuw"  # Replace with your actual Google Sheet ID
        import_from_sheet(SHEET_ID, ASSESSMENT_DB, GOOGLE_SHEET_CREDENTIALS)
    
    elif args.export_to_report:
        export_to_report_csv(
            csv_path="data/assessment_db.csv",       
            uc_json_path="data/uc_problems_db.json", 
            sheet_id="1t5nQkdwaw-NZl2Abupu5gV2tE546ObEkWtHqFY-BwXw",         
            creds_path="data/praxis-backup-478106-c1-7f3f6481cd6e.json",
            worksheet_name="Worksheet") 
   

    elif args.export_in_batches:
        next_idx = args.start_idx
        batch_count = 0
        
        while next_idx is not None:
            batch_count += 1
            print(f"\n{'='*60}")
            print(f"[BATCH {batch_count}] Starting batch export from row {next_idx}")
            print(f"{'='*60}")
            
            try:
                next_idx = export_to_report_in_batches(
                    csv_path=ASSESSMENT_DB,
                    uc_json_path=USECASE_DB,
                    sheet_id="1t5nQkdwaw-NZl2Abupu5gV2tE546ObEkWtHqFY-BwXw",
                    creds_path=GOOGLE_SHEET_CREDENTIALS,
                    worksheet_name="Worksheet",
                    batch_size=args.batch_size,
                    start_idx=next_idx,
                    delay_between_batches=args.batch_delay
                )
                
                if next_idx is not None:
                    print(f"[BATCH {batch_count}] ⏳ Waiting {args.batch_delay} seconds before next batch...\n")
                    for i in range(args.batch_delay, 0, -10):
                        print(f"[BATCH {batch_count}] {i}s remaining...", end="\r")
                        time.sleep(10)
                    print()  # Newline
                else:
                    print("\n[BATCH] 🎉 All rows processed successfully!")
                    
            except Exception as e:
                print(f"[BATCH {batch_count}] ❌ Error: {e}")
                print(f"[BATCH {batch_count}] Waiting 5 minutes before retry from row {next_idx}...")
                time.sleep(300)  # 5 Minuten warten
                continue

        print(f"\n[COMPLETE] Exported {batch_count} batches total.")


    elif args.inspect:
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

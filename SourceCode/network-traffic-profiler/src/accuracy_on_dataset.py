"""
this is an automated pipeline prediction test:
it runs all pcap files from /ML/datasets through the trained model
and tests each dominant action against the expected dominant action

run from src/ directory: python accuracy_on_dataset.py

summary from 12/03/2026:
=================================================================
Action        Pass  Fail   Pass Rate
────────────────────────────────────────
Comment         96    54         64%
Like            95    55         63%
Play             0   150          0%
Search          33   117         22%
Subscribe       88    62         59%
────────────────────────────────────────
TOTAL          312   438         42%
=================================================================
"""

import os
import sys
import pandas as pd
import ipaddress
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_features_unified import extract_all_pcap_data
from ML.model_training.predict import predict_action_type

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML", "datasets")
ACTIONS = ["Comment", "Like", "Play", "Search", "Subscribe"]
YT_RANGES = [
    "172.217.0.0/16", "142.250.0.0/15", "104.237.160.0/19",
    "208.117.224.0/19", "64.15.112.0/20", "216.58.192.0/19",
    "74.125.0.0/16"
] # aligning with extract_features_unified.py

# how many files per action to test (None = all)
MAX_FILES_PER_ACTION = None

# check that flow is yt flow
def is_youtube_ip(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in ipaddress.ip_network(r) for r in YT_RANGES)
    except Exception:
        return False

# run feature extraction + prediction for a single PCAP, returns ml_features_df with action_type
def run_prediction(pcap_path):
    result = extract_all_pcap_data(pcap_path)
    if not result or result[1] is None or result[1].empty:
        return None
    return predict_action_type(result[1])

# evaluation against expected action
def evaluate_pcap(pcap_path, expected_action):

    ml_df = run_prediction(pcap_path)

    # empty flow
    if ml_df is None:
        return {"pass": False, "dominant": "NO FEATURES", "total": 0, "correct": 0}

    # check yt ips
    yt_df = ml_df[
        ml_df["src_ip"].apply(is_youtube_ip) |
        ml_df["dst_ip"].apply(is_youtube_ip)
    ]

    # if no yt ips found
    if yt_df.empty:
        return {"pass": False, "dominant": "NO YT FLOWS", "total": 0, "correct": 0}

    # store all found actions
    counts = yt_df["action_type"].value_counts()

    # ignore background as dominant action as for most flows it is the dominant
    dominant = counts.drop("Background", errors="ignore").idxmax() \
        if not counts.drop("Background", errors="ignore").empty \
        else "Background"

    correct = counts.get(expected_action, 0)

    return {
        "pass": dominant == expected_action, # true if expected action aligns with dominant
        "dominant": dominant,
        "total": len(yt_df),
        "correct": correct,
        "counts": counts.to_dict()
    }

# main test function
def run_tests():

    print("=" * 65)
    print("PIPELINE PREDICTION TEST")
    print("Dataset:", DATASET_DIR)
    print("=" * 65)

    # store results
    summary = defaultdict(lambda: {"pass": 0, "fail": 0})
    results = []

    # traverse dataset
    for action in ACTIONS:
        action_dir = os.path.join(DATASET_DIR, action)
        if not os.path.exists(action_dir):
            print(f"\n[SKIP] {action}/ directory not found")
            continue

        files = sorted(f for f in os.listdir(action_dir)
                       if f.lower().endswith((".pcap", ".pcapng")))

        if MAX_FILES_PER_ACTION:
            files = files[:MAX_FILES_PER_ACTION]

        # action header
        print(f"\n{'─'*65}")
        print(f"Action: {action.upper()} ({len(files)} files)")
        print(f"{'─'*65}")

        for fname in files:

            path = os.path.join(action_dir, fname)
            r = evaluate_pcap(path, action)

            # for each flow print either pass or fail
            status = "✅ PASS" if r["pass"] else "❌ FAIL"
            # store to dictionary
            summary[action]["pass" if r["pass"] else "fail"] += 1

            print(
                f"{status} {fname}\n"
                f"YT flows: {r['total']} | "
                f"Correct: {r['correct']} | "
                f"Dominant: {r['dominant']}"
            )

            if not r["pass"] and "counts" in r:
                print("Predictions:", r["counts"])

            results.append({
                "action": action,
                "file": fname,
                "pass": r["pass"],
                "total_flows": r["total"],
                "correct": r["correct"],
                "dominant": r["dominant"]
            })

    # total summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Action':<12}{'Pass':>6}{'Fail':>6}{'Pass Rate':>12}")
    print("─" * 40)

    total_pass = total_fail = 0

    # pass/fail ratio
    for action in ACTIONS:
        p = summary[action]["pass"]
        f = summary[action]["fail"]
        total = p + f
        rate = f"{p/total*100:.0f}%" if total else "N/A"

        print(f"{action:<12}{p:>6}{f:>6}{rate:>12}")

        total_pass += p
        total_fail += f

    # total
    grand_total = total_pass + total_fail
    grand_rate = f"{total_pass/grand_total*100:.0f}%" if grand_total else "N/A"

    print("─" * 40)
    print(f"{'TOTAL':<12}{total_pass:>6}{total_fail:>6}{grand_rate:>12}")
    print("=" * 65)

    # save results to csv file
    out_path = os.path.join(os.path.dirname(__file__), "dataset_accuracy_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("\nFull results saved to:", out_path)

if __name__ == "__main__":
    run_tests()
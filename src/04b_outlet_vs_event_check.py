"""
Step 4b: Is BERTopic grouping the constructed documents by EVENT or by OUTLET?

Reads results/constructed_dataset_assignments.csv (written by script 04) and
reports, in plain numbers:
  1. An outlet x topic table (the clearest single piece of evidence)
  2. NMI between topic and outlet, with a permutation test: how often would
     randomly shuffled topic labels line up with outlets this well?
  3. Where documents that share a real-world event landed
  4. Mean document length per outlet -- a possible FORMAT confound: if one
     outlet family writes longer summaries, topics could split on length
     and style rather than on events.

IMPORTANT: this is a small-sample, descriptive check, not a rigorous test.
Outliers (topic -1) are excluded from the statistics.

Usage:
    python src/04b_outlet_vs_event_check.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

ASSIGN_PATH = "results/constructed_dataset_assignments.csv"
CLEAN_PATH = "data/constructed_news_dataset_clean.csv"
EVENT_KEYWORDS = ["Royal Navy", "Christa Pike", "Ebola"]  # events seen in 2+ documents
N_PERMUTATIONS = 5000


def main():
    if not os.path.isfile(ASSIGN_PATH):
        print(f"No {ASSIGN_PATH} found. Run src/04_apply_to_constructed.py first.")
        return

    df = pd.read_csv(ASSIGN_PATH)
    df["manual_event_label"] = df["manual_event_label"].fillna("")
    clustered = df[df["topic"] != -1]
    print(f"{len(df)} documents: {len(df) - len(clustered)} outlier(s) excluded, "
          f"{len(clustered)} clustered into {clustered['topic'].nunique()} topic(s).")

    # 1) outlet x topic table
    print("\n[1] Outlet x topic table (rows = outlet, columns = topic id):")
    print(pd.crosstab(clustered["source"], clustered["topic"]).to_string())

    # 2) NMI + permutation test
    sources = clustered["source"].to_numpy()
    topics = clustered["topic"].to_numpy()
    observed = normalized_mutual_info_score(sources, topics)
    rng = np.random.default_rng(0)
    null = np.array([normalized_mutual_info_score(sources, rng.permutation(topics))
                     for _ in range(N_PERMUTATIONS)])
    p_value = (np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1)
    print(f"\n[2] NMI(topic, outlet) = {observed:.2f}   "
          f"(0 = unrelated, 1 = topics are exactly the outlets)")
    print(f"    Shuffled-label baseline: mean NMI {null.mean():.2f}, "
          f"95th percentile {np.percentile(null, 95):.2f}")
    print(f"    Permutation test p = {p_value:.4f} ({N_PERMUTATIONS} shuffles)")

    # 3) shared real-world events
    print("\n[3] Documents sharing a real-world event (from your manual labels):")
    for kw in EVENT_KEYWORDS:
        hits = df[df["manual_event_label"].str.contains(kw, case=False)]
        print(f"  '{kw}':")
        for _, r in hits.iterrows():
            t = "outlier" if r["topic"] == -1 else f"topic {r['topic']}"
            print(f"      {r['date']}  {r['source']:<10} -> {t}")

    # 4) format confound
    if os.path.isfile(CLEAN_PATH):
        clean = pd.read_csv(CLEAN_PATH)
        clean["chars"] = clean["pseudo_document_clean"].str.len()
        print("\n[4] Mean document length per outlet (possible format confound):")
        print(clean.groupby("source")[["chars", "n_headlines_used"]].mean().round(0).to_string())

    print("\nHow to read this: a high NMI with a small p means topics line up "
          "with outlets, i.e. style/format is a major driver and these results "
          "cannot be credited to event-tracking. Documents sharing an event "
          "only count as event-evidence if they land together AND their "
          "outlets were NOT already sharing a topic anyway.")


if __name__ == "__main__":
    main()
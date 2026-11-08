"""
Step 4: Apply BERTopic to the constructed news pseudo-document dataset.

Run this AFTER you've run 03_build_dataset.py at least a few times (ideally
across several different days) so there's enough data for clustering to
be meaningful. With very few documents, expect either very few topics or
mostly "outlier" assignments -- that's expected and fine to report
honestly as early-stage progress.

Usage:
    python src/04_apply_to_constructed.py
    python src/04_apply_to_constructed.py --offline --min_topic_size 2
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--min_topic_size", type=int, default=2,
                         help="Keep this LOW for a small, newly-collected dataset.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for UMAP, so reruns give the same topics.")
    parser.add_argument("--min_df", type=int, default=1,
                         help="Minimum document frequency for topic words "
                              "(1 for this small dataset).")
    args = parser.parse_args()

    clean_path = "data/constructed_news_dataset_clean.csv"
    raw_path = "data/constructed_news_dataset.csv"

    if os.path.isfile(clean_path):
        path, text_col = clean_path, "pseudo_document_clean"
    elif os.path.isfile(raw_path):
        print(f"    ! Using raw (unpreprocessed) data -- run "
              f"src/03b_preprocess_dataset.py first for cleaner results.")
        path, text_col = raw_path, "pseudo_document"
    else:
        print(f"No data found. Run src/03_build_dataset.py first.")
        return

    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)  # keep rows aligned with docs
    docs = df[text_col].tolist()
    print(f"[1/3] Loaded {len(docs)} pseudo-documents from "
          f"{df['date'].nunique()} day(s), {df['source'].nunique()} source(s).")

    if len(docs) < 5:
        print("WARNING: very few documents so far -- this is expected early on. "
              "Keep running 03_build_dataset.py over more days before your final "
              "submission to grow this dataset, then re-run this script.")

    print("[2/3] Fitting BERTopic on the constructed dataset...")
    # Outlet names (and words like "says" that appear in every headline)
    # are removed from topic words. Otherwise a topic labelled "bbc_npr"
    # would reflect which outlet wrote it, not which event it covers --
    # the very confound the event-vs-outlet check is testing for.
    outlet_words = [w for s in df["source"].unique() for w in str(s).lower().split()]
    extra_stop_words = outlet_words + ["aljazeera", "al", "jazeera", "guardian",
                                       "bbc", "npr", "dw", "says", "say", "said"]
    embedding_model = get_embedding_backend(offline=args.offline)
    topic_model = build_topic_model(
        embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=None,
        min_df=args.min_df,
        seed=args.seed,
        n_docs=len(docs), docs=docs,
        extra_stop_words=extra_stop_words,
    )
    topics, probs = topic_model.fit_transform(docs)

    topics_words = topic_model.get_topics()
    coherence = compute_npmi_coherence(topics_words, docs, top_n=10,
                                       extra_stop_words=extra_stop_words)
    diversity = compute_topic_diversity(topics_words, top_n=10)

    print("[3/3] Saving results...")
    os.makedirs("results", exist_ok=True)

    # --- Per-document assignments + event-vs-outlet check -----------------
    # The question this dataset exists to answer: does BERTopic group
    # pseudo-documents by REAL-WORLD EVENT (documents from different outlets
    # in the same topic) or by OUTLET STYLE (same outlet in the same topic)?
    df["topic"] = topics
    ann_path = "data/constructed_dataset_annotations.csv"
    if os.path.isfile(ann_path):
        ann = pd.read_csv(ann_path).drop_duplicates(subset=["date", "source"], keep="first")
        df = df.merge(ann[["date", "source", "manual_event_label"]],
                      on=["date", "source"], how="left")
    else:
        df["manual_event_label"] = ""
    df["manual_event_label"] = df["manual_event_label"].fillna("")
    df[["date", "source", "topic", "manual_event_label"]].to_csv(
        "results/constructed_dataset_assignments.csv", index=False)

    lines = ["Event-vs-outlet check (small-sample, descriptive -- not statistical evidence)",
             "=" * 74]
    real = df[df["topic"] != -1]
    n_cross = n_single = 0
    for t, g in real.groupby("topic"):
        srcs = sorted(g["source"].unique())
        days = sorted(g["date"].unique())
        kind = "CROSS-OUTLET" if len(srcs) > 1 else "single-outlet"
        n_cross += kind == "CROSS-OUTLET"
        n_single += kind == "single-outlet"
        lines.append(f"Topic {t}: {len(g)} docs, {kind}, {len(days)} day(s)")
        for _, r in g.iterrows():
            lines.append(f"    {r['date']}  {r['source']:<10} {r['manual_event_label']}")
    n_out = int((df["topic"] == -1).sum())
    lines.append(f"Outliers (no topic): {n_out}/{len(df)}")
    lines.append(f"Topics spanning 2+ outlets: {n_cross} | single-outlet topics: {n_single}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    with open("results/constructed_dataset_event_vs_outlet.txt", "w") as f:
        f.write(report + "\n")
    # ----------------------------------------------------------------------

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("results/constructed_dataset_topics.csv", index=False)

    with open("results/constructed_dataset_metrics.txt", "w") as f:
        f.write("Constructed news pseudo-document dataset -- early results\n")
        f.write("=" * 58 + "\n")
        f.write(f"Pseudo-documents used: {len(docs)}\n")
        f.write(f"Days covered:          {df['date'].nunique()}\n")
        f.write(f"Sources covered:       {df['source'].nunique()}\n")
        f.write(f"Topics found:          {len(topic_info) - 1}\n")
        f.write(f"Average NPMI coherence: {coherence['average']:.4f}\n")
        f.write(f"Topic diversity:        {diversity:.4f}\n")
        f.write(f"Seed:                   {args.seed}\n")

    print(f"\nDone. See results/constructed_dataset_topics.csv "
          f"and results/constructed_dataset_metrics.txt")


if __name__ == "__main__":
    main()
"""
Step 1: Reproduce the original BERTopic paper's result on 20 Newsgroups.

This directly addresses the "Replication of Original Work" rubric item --
it fits BERTopic on one of the three datasets used in Grootendorst (2022)
and reports the same two metrics the paper uses (topic coherence via NPMI,
topic diversity), so you have a genuine, comparable result to show.

Usage:
    python src/01_reproduce_original.py                # full real run
    python src/01_reproduce_original.py --offline       # fast local smoke test
    python src/01_reproduce_original.py --n_docs 3000   # cap corpus size for speed
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                         help="Use offline TF-IDF embedding instead of a real "
                              "sentence-transformer (fast smoke test only).")
    parser.add_argument("--n_docs", type=int, default=None,
                         help="Optionally cap the number of documents for a faster run.")
    parser.add_argument("--min_topic_size", type=int, default=10,
                         help="BERTopic's min_topic_size parameter.")
    parser.add_argument("--nr_topics", type=int, default=None,
                         help="Constrain the final topic count (e.g. 20 or 50) to match "
                              "the paper's tested range of 10-50, rather than using "
                              "BERTopic's auto-discovered count.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for UMAP, so reruns give the same topics.")
    parser.add_argument("--min_df", type=int, default=2,
                         help="Minimum document frequency for topic words.")
    args = parser.parse_args()

    print("[1/4] Loading 20 Newsgroups (same dataset family used in the paper)...")
    docs = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )["data"]
    docs = [d for d in docs if len(d.strip()) > 20]  # drop near-empty docs
    if args.n_docs:
        docs = docs[: args.n_docs]
    print(f"    -> {len(docs)} documents loaded.")

    print("[2/4] Fitting BERTopic (embed -> reduce -> cluster -> label)...")
    embedding_model = get_embedding_backend(offline=args.offline)
    topic_model = build_topic_model(
        embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        min_df=args.min_df,
        seed=args.seed,
        n_docs=len(docs), docs=docs,
    )
    topics, probs = topic_model.fit_transform(docs)

    print("[3/4] Computing evaluation metrics (NPMI coherence, topic diversity)...")
    topics_words = topic_model.get_topics()
    coherence = compute_npmi_coherence(topics_words, docs, top_n=10)
    diversity = compute_topic_diversity(topics_words, top_n=10)

    print(f"    -> Average NPMI coherence: {coherence['average']:.4f}")
    print(f"    -> Topic diversity:        {diversity:.4f}")
    print(f"    -> Number of topics found: {len(topic_model.get_topic_info()) - 1} "
          f"(plus one outlier group)")

    print("[4/4] Saving results...")
    os.makedirs("results", exist_ok=True)
    topic_info = topic_model.get_topic_info()
    n_topics_found = len(topic_info) - 1

    tag = f"k{args.nr_topics}" if args.nr_topics else "auto"
    csv_path = f"results/20newsgroups_topics_{tag}.csv"
    metrics_path = f"results/20newsgroups_metrics_{tag}.txt"
    chart_path = f"results/20newsgroups_topic_barchart_{tag}.html"

    topic_info.to_csv(csv_path, index=False)

    with open(metrics_path, "w") as f:
        f.write(f"20 Newsgroups reproduction -- evaluation metrics (nr_topics={tag})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Documents used:        {len(docs)}\n")
        f.write(f"Topics requested:      {args.nr_topics if args.nr_topics else 'auto (no constraint)'}\n")
        f.write(f"Topics found:          {n_topics_found}\n")
        f.write(f"Average NPMI coherence: {coherence['average']:.4f}\n")
        f.write(f"Topic diversity:        {diversity:.4f}\n")
        f.write(f"Seed:                   {args.seed}\n")

    summary_path = "results/20newsgroups_sweep_summary.csv"
    update_sweep_summary(summary_path, {
        "n_docs": len(docs),
        "nr_topics_requested": args.nr_topics if args.nr_topics else "auto",
        "seed": args.seed,
        "nr_topics_found": n_topics_found,
        "npmi_coherence": round(coherence["average"], 4),
        "topic_diversity": round(diversity, 4),
    }, key_cols=["n_docs", "nr_topics_requested", "seed"])

    try:
        fig = topic_model.visualize_barchart(top_n_topics=12)
        fig.write_html(chart_path)
        print(f"    -> Saved {chart_path}")
    except Exception as e:
        print(f"    (Visualization skipped: {e})")

    print(f"\nDone. See {csv_path}, {metrics_path}, and the running "
          f"comparison table at {summary_path}")

if __name__ == "__main__":
    main()

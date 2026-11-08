"""
Step 5: Reproduce the original paper's result on BBC News -- the second of
the three datasets used in Grootendorst (2022), and the exact source your
proposal cites (Greene and Cunningham, 2006).

Downloads the official raw-text distribution directly from the authors'
own site (http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), which is the
same source your references.bib entry points to -- so this is a faithful
reproduction, not a third-party reformatted copy.

Usage:
    python src/05_bbc_news.py                      # full real run
    python src/05_bbc_news.py --offline             # fast local smoke test
    python src/05_bbc_news.py --nr_topics 5          # constrain to the paper's known 5 categories
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)

BBC_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
RAW_DIR = "data/bbc_raw"


def download_and_extract_bbc():
    if os.path.isdir(os.path.join(RAW_DIR, "bbc")):
        print(f"    -> Already downloaded, reusing {RAW_DIR}/bbc")
        return os.path.join(RAW_DIR, "bbc")

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"    -> Downloading {BBC_URL} ...")
    with urllib.request.urlopen(BBC_URL, timeout=60) as response:
        zip_bytes = response.read()

    print("    -> Extracting...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(RAW_DIR)

    return os.path.join(RAW_DIR, "bbc")


def load_bbc_documents(bbc_dir: str):
    """Returns (docs, categories) -- categories are the known ground-truth
    labels (business/entertainment/politics/sport/tech), useful later for
    an optional sanity check of whether discovered topics line up with
    real categories."""
    docs, categories = [], []
    for category in sorted(os.listdir(bbc_dir)):
        cat_path = os.path.join(bbc_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            fpath = os.path.join(cat_path, fname)
            try:
                # Original files are latin-1 encoded, as noted by the dataset authors
                with open(fpath, "r", encoding="latin-1") as f:
                    text = f.read().strip()
                if len(text) > 20:
                    docs.append(text)
                    categories.append(category)
            except Exception:
                continue
    return docs, categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                         help="Use offline TF-IDF embedding instead of a real "
                              "sentence-transformer (fast smoke test only).")
    parser.add_argument("--min_topic_size", type=int, default=10)
    parser.add_argument("--nr_topics", type=int, default=None,
                         help="Constrain topic count. Try 5 to match the dataset's "
                              "known 5 categories (business/entertainment/politics/sport/tech).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for UMAP, so reruns give the same topics.")
    parser.add_argument("--min_df", type=int, default=2,
                         help="Minimum document frequency for topic words.")
    args = parser.parse_args()

    print("[1/5] Downloading / loading BBC News dataset (Greene & Cunningham, 2006)...")
    bbc_dir = download_and_extract_bbc()
    docs, categories = load_bbc_documents(bbc_dir)
    print(f"    -> {len(docs)} articles loaded across {len(set(categories))} known categories: "
          f"{sorted(set(categories))}")

    print("[2/5] Fitting BERTopic (embed -> reduce -> cluster -> label)...")
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

    print("[3/5] Computing evaluation metrics (NPMI coherence, topic diversity)...")
    topics_words = topic_model.get_topics()
    coherence = compute_npmi_coherence(topics_words, docs, top_n=10)
    diversity = compute_topic_diversity(topics_words, top_n=10)
    n_topics_found = len(topic_model.get_topic_info()) - 1

    print(f"    -> Average NPMI coherence: {coherence['average']:.4f}")
    print(f"    -> Topic diversity:        {diversity:.4f}")
    print(f"    -> Number of topics found: {n_topics_found} (plus one outlier group)")

    print("[4/5] Checking alignment with known categories (bonus sanity check)...")
    df = pd.DataFrame({"category": categories, "topic": topics})
    crosstab = pd.crosstab(df["category"], df["topic"])

    real_topic_cols = [c for c in crosstab.columns if c != -1]
    n_outliers_per_cat = crosstab[-1] if -1 in crosstab.columns else crosstab.iloc[:, 0] * 0
    dominant_topic_per_category = crosstab[real_topic_cols].idxmax(axis=1)

    print("    -> Dominant REAL discovered topic per known category "
          "(excluding the outlier bucket):")
    for cat in dominant_topic_per_category.index:
        top = dominant_topic_per_category[cat]
        n_in_dominant = crosstab.loc[cat, top]
        n_outlier = n_outliers_per_cat.get(cat, 0)
        n_total = crosstab.loc[cat].sum()
        print(f"       {cat:15s} -> topic {top} "
              f"({n_in_dominant}/{n_total} articles; {n_outlier} were outliers)")

    print("[5/5] Saving results...")
    os.makedirs("results", exist_ok=True)
    tag = f"k{args.nr_topics}" if args.nr_topics else "auto"
    csv_path = f"results/bbcnews_topics_{tag}.csv"
    metrics_path = f"results/bbcnews_metrics_{tag}.txt"
    chart_path = f"results/bbcnews_topic_barchart_{tag}.html"
    crosstab_path = f"results/bbcnews_category_alignment_{tag}.csv"

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(csv_path, index=False)
    crosstab.to_csv(crosstab_path)

    with open(metrics_path, "w") as f:
        f.write(f"BBC News reproduction -- evaluation metrics (nr_topics={tag})\n")
        f.write("=" * 55 + "\n")
        f.write(f"Documents used:        {len(docs)}\n")
        f.write(f"Known categories:      {len(set(categories))}\n")
        f.write(f"Topics found:          {n_topics_found}\n")
        f.write(f"Average NPMI coherence: {coherence['average']:.4f}\n")
        f.write(f"Topic diversity:        {diversity:.4f}\n")
        f.write(f"Seed:                   {args.seed}\n")

    summary_path = "results/bbcnews_sweep_summary.csv"
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

    print(f"\nDone. See {csv_path}, {metrics_path}, {crosstab_path}, "
          f"and the running comparison table at {summary_path}")


if __name__ == "__main__":
    main()
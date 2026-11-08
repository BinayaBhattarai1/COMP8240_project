"""
Step 11 (v2 MODELLING): fit BERTopic on the article-level dataset.

Run it on both text versions so you can test the length confound:
    python src/11_bertopic_articles.py --text full
    python src/11_bertopic_articles.py --text capped

Each run writes:
  results/v2/assignments/<run>.csv   article_id -> topic (used by step 13)
  results/v2/topics_<run>.csv        BERTopic topic table
  results/v2/runs_summary.csv        one row per run: topics, NPMI, diversity
where <run> = bertopic_<text>_mts<min_topic_size>_seed<seed>.

Options:
  --min_topic_size  smaller -> more, narrower topics (closer to single events).
                    Try 3, 5 (default) and 10.
  --seed            UMAP seed; the paper averages 3 runs, so also try --seed 1/2/3.
  --offline         fast TF-IDF stand-in embedder, ONLY to test the pipeline.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)
from v2_common import CLEAN_PATH, RESULTS_DIR, OUTLET_STOP_WORDS, TEXT_VARIANTS, ensure_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", choices=TEXT_VARIANTS, default="full")
    p.add_argument("--min_topic_size", type=int, default=5)
    p.add_argument("--nr_topics", type=int, default=None,
                   help="optionally reduce to this many topics (default: let BERTopic decide)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--embedding", default=None,
                   help="sentence-transformer name, e.g. all-mpnet-base-v2 (the paper's model); "
                        "default all-MiniLM-L6-v2")
    a = p.parse_args()

    if not os.path.isfile(CLEAN_PATH):
        print(f"No {CLEAN_PATH}. Run src/08_preprocess_articles.py first.")
        return
    ensure_dirs()
    os.makedirs(os.path.join(RESULTS_DIR, "assignments"), exist_ok=True)

    col = TEXT_VARIANTS[a.text]
    df = pd.read_csv(CLEAN_PATH, dtype={"article_id": str})
    df = df[df[col].fillna("").str.strip() != ""].reset_index(drop=True)
    docs = df[col].tolist()
    run = f"bertopic_{a.text}_mts{a.min_topic_size}_seed{a.seed}"
    if a.nr_topics:
        run += f"_k{a.nr_topics}"
    if a.embedding and a.embedding != "all-MiniLM-L6-v2":
        run += "_" + a.embedding.split("/")[-1].split("-")[1]   # e.g. _mpnet
    if a.offline:
        run += "_OFFLINE"
    print(f"[1/3] {len(docs)} articles, {df['published_date'].nunique()} days, "
          f"{df['outlet'].nunique()} outlets  |  text = {col}  |  run = {run}")

    print("[2/3] Fitting BERTopic...")
    model = build_topic_model(get_embedding_backend(offline=a.offline, model_name=a.embedding),
                              min_topic_size=a.min_topic_size, nr_topics=a.nr_topics,
                              min_df=2, seed=a.seed, n_docs=len(docs), docs=docs,
                              extra_stop_words=OUTLET_STOP_WORDS)
    topics, _ = model.fit_transform(docs)

    words = model.get_topics()
    coh = compute_npmi_coherence(words, docs, top_n=10, extra_stop_words=OUTLET_STOP_WORDS)
    div = compute_topic_diversity(words, top_n=10)
    n_topics = len([t for t in words if t != -1])
    outlier_share = float((pd.Series(topics) == -1).mean())

    print("[3/3] Saving")
    pd.DataFrame({"article_id": df["article_id"], "topic": topics}).to_csv(
        os.path.join(RESULTS_DIR, "assignments", f"{run}.csv"), index=False)
    model.get_topic_info().to_csv(os.path.join(RESULTS_DIR, f"topics_{run}.csv"), index=False)
    if not a.offline:
        try:
            model.visualize_barchart(top_n_topics=12).write_html(
                os.path.join(RESULTS_DIR, f"barchart_{run}.html"))
        except Exception as e:  # plotting is optional
            print(f"    (bar chart skipped: {e})")
    row = {"run": run, "model": "BERTopic", "embedding": a.embedding or "all-MiniLM-L6-v2",
           "text": a.text, "min_topic_size": a.min_topic_size,
           "nr_topics": a.nr_topics or "auto", "seed": a.seed, "offline": a.offline,
           "n_docs": len(docs), "n_topics": n_topics, "outlier_share": round(outlier_share, 4),
           "npmi": round(coh["average"], 4), "diversity": round(div, 4)}
    update_sweep_summary(os.path.join(RESULTS_DIR, "runs_summary.csv"), row, ["run"])

    print(f"\n{run}: {n_topics} topics, {outlier_share:.0%} outliers, "
          f"NPMI {coh['average']:.4f}, diversity {div:.4f}")
    print(model.get_topic_info().head(10)[["Topic", "Count", "Name"]].to_string(index=False))


if __name__ == "__main__":
    main()

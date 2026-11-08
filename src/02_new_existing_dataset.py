"""
Step 2: Apply BERTopic to a new existing dataset not used in the original
paper (Amazon product reviews), to test generalisation to a new domain.

This directly addresses the "New Data" rubric item, specifically the
"at least one new existing dataset" requirement -- and produces a real,
comparable result rather than just a plan.

Usage:
    python src/02_new_existing_dataset.py
    python src/02_new_existing_dataset.py --offline --n_docs 500
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--n_docs", type=int, default=2000,
                         help="Number of reviews to use (keep modest for speed).")
    parser.add_argument("--min_topic_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for UMAP, so reruns give the same topics.")
    parser.add_argument("--min_df", type=int, default=2,
                         help="Minimum document frequency for topic words.")
    args = parser.parse_args()

    print("[1/4] Loading Amazon product reviews (fancyzhx/amazon_polarity, via Hugging Face)...")
    from datasets import load_dataset
    # NOTE: recent datasets/huggingface_hub versions require namespaced repo
    # IDs -- the old bare name "amazon_polarity" no longer resolves, since
    # HF now requires "org/dataset-name" format. This is the same dataset,
    # just under its current namespaced location.
    dataset = load_dataset("fancyzhx/amazon_polarity", split=f"train[:{args.n_docs}]")
    docs = [d for d in dataset["content"] if len(d.strip()) > 20]
    print(f"    -> {len(docs)} reviews loaded.")

    print("[2/4] Fitting BERTopic...")
    embedding_model = get_embedding_backend(offline=args.offline)
    topic_model = build_topic_model(
        embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=None,
        min_df=args.min_df,
        seed=args.seed,
        n_docs=len(docs), docs=docs,
    )
    topics, probs = topic_model.fit_transform(docs)

    print("[3/4] Computing evaluation metrics...")
    topics_words = topic_model.get_topics()
    coherence = compute_npmi_coherence(topics_words, docs, top_n=10)
    diversity = compute_topic_diversity(topics_words, top_n=10)
    print(f"    -> Average NPMI coherence: {coherence['average']:.4f}")
    print(f"    -> Topic diversity:        {diversity:.4f}")

    print("[4/4] Saving results...")
    os.makedirs("results", exist_ok=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv("results/amazon_reviews_topics.csv", index=False)

    with open("results/amazon_reviews_metrics.txt", "w") as f:
        f.write("Amazon product reviews -- new-domain generalisation test\n")
        f.write("=" * 55 + "\n")
        f.write(f"Documents used:        {len(docs)}\n")
        f.write(f"Topics found:          {len(topic_info) - 1}\n")
        f.write(f"Average NPMI coherence: {coherence['average']:.4f}\n")
        f.write(f"Topic diversity:        {diversity:.4f}\n")
        f.write(f"Seed:                   {args.seed}\n")
        f.write("\nContext: commerce-domain, short informal text -- compare against\n")
        f.write("the 20 Newsgroups result to see whether BERTopic's performance\n")
        f.write("holds up outside the news/political domains used in the paper.\n")

    print("\nDone. See results/amazon_reviews_topics.csv and "
          "results/amazon_reviews_metrics.txt")


if __name__ == "__main__":
    main()

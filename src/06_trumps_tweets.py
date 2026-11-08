"""
Step 6: Reproduce the original paper's result on Trump's tweets -- the
third and final of the three datasets used in Grootendorst (2022).

Downloads real tweets directly from bpb27/trump_tweet_data_archive on
GitHub (www.trumptwitterarchive.com's underlying data source), a
widely-used, publicly available archive with no authentication required.

IMPORTANT HONESTY NOTE: this archive only contains condensed yearly files
for 2009-2018 (confirmed by checking directly -- 2019/2020/2021 files do
not exist in this particular repo, even though the account was active
until suspension in January 2021). Your proposal describes the paper's
dataset as covering 2009-2021 with 44,253 tweets; this script will give
you a real, substantial, genuinely comparable dataset from the same
account and era, but the exact tweet count and date range will differ
from the paper's reported numbers. Report your ACTUAL numbers honestly
rather than claiming an exact match -- that's a stronger, more credible
claim than pretending to match a number you didn't actually reproduce.

Usage:
    python src/06_trumps_tweets.py                 # full real run
    python src/06_trumps_tweets.py --offline        # fast local smoke test
    python src/06_trumps_tweets.py --years 2016 2017 2018  # specific years only
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import (get_embedding_backend, build_topic_model, compute_npmi_coherence,
                   compute_topic_diversity, update_sweep_summary)

BASE_URL = "https://raw.githubusercontent.com/bpb27/trump_tweet_data_archive/master/condensed_{year}.json.zip"
AVAILABLE_YEARS = list(range(2009, 2019))  # 2009-2018 inclusive -- confirmed available
RAW_DIR = "data/trump_tweets_raw"


def download_year(year: int) -> list:
    cache_path = os.path.join(RAW_DIR, f"condensed_{year}.json")
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = BASE_URL.format(year=year)
    print(f"    -> Downloading {year}...")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            zip_bytes = response.read()
    except Exception as e:
        print(f"    ! Could not download {year}: {e}")
        return []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            data = json.load(f)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def clean_tweet_text(text: str) -> str:
    """Light preprocessing only, matching the proposal's description of the
    paper's approach ('lowercased only') -- deliberately minimal so the
    noisy, informal character of tweets is preserved for the method to
    handle, rather than cleaned away."""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)  # strip URLs, keep everything else
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                         help="Use offline TF-IDF embedding instead of a real "
                              "sentence-transformer (fast smoke test only).")
    parser.add_argument("--years", type=int, nargs="+", default=AVAILABLE_YEARS,
                         help=f"Which years to include. Available: {AVAILABLE_YEARS}")
    parser.add_argument("--min_topic_size", type=int, default=10)
    parser.add_argument("--nr_topics", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for UMAP, so reruns give the same topics.")
    parser.add_argument("--min_df", type=int, default=2,
                         help="Minimum document frequency for topic words.")
    args = parser.parse_args()

    invalid_years = [y for y in args.years if y not in AVAILABLE_YEARS]
    if invalid_years:
        print(f"    ! Warning: years {invalid_years} are not available in this "
              f"archive (only {AVAILABLE_YEARS[0]}-{AVAILABLE_YEARS[-1]} exist). Skipping those.")

    print(f"[1/5] Downloading / loading tweets for years: "
          f"{[y for y in args.years if y in AVAILABLE_YEARS]}...")
    all_tweets = []
    for year in args.years:
        if year not in AVAILABLE_YEARS:
            continue
        all_tweets.extend(download_year(year))

    print(f"    -> {len(all_tweets)} total tweets loaded (including retweets).")

    non_retweets = [t for t in all_tweets if not t.get("is_retweet", False)]
    docs = [clean_tweet_text(t["text"]) for t in non_retweets if t.get("text")]
    docs = [d for d in docs if len(d) > 10]

    print(f"    -> {len(non_retweets)} non-retweet tweets, "
          f"{len(docs)} after removing near-empty ones.")
    print(f"    NOTE: this will differ from the paper's reported 44,253 tweets "
          f"(2009-2021) since this archive only covers 2009-2018 -- report your "
          f"actual count honestly rather than the paper's number.")

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

    print("[4/5] Saving results...")
    os.makedirs("results", exist_ok=True)
    tag = f"k{args.nr_topics}" if args.nr_topics else "auto"
    csv_path = f"results/trumptweets_topics_{tag}.csv"
    metrics_path = f"results/trumptweets_metrics_{tag}.txt"
    chart_path = f"results/trumptweets_topic_barchart_{tag}.html"

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(csv_path, index=False)

    with open(metrics_path, "w") as f:
        f.write(f"Trump's tweets reproduction -- evaluation metrics (nr_topics={tag})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Years included:        {[y for y in args.years if y in AVAILABLE_YEARS]}\n")
        f.write(f"Non-retweet tweets:    {len(docs)}\n")
        f.write(f"Topics found:          {n_topics_found}\n")
        f.write(f"Average NPMI coherence: {coherence['average']:.4f}\n")
        f.write(f"Topic diversity:        {diversity:.4f}\n")
        f.write(f"Seed:                   {args.seed}\n")
        f.write("\nNOTE: paper reports 44,253 tweets from 2009-2021; this archive\n")
        f.write("only covers 2009-2018, so counts differ. This is disclosed honestly\n")
        f.write("rather than claiming an exact match to the paper's dataset.\n")

    summary_path = "results/trumptweets_sweep_summary.csv"
    update_sweep_summary(summary_path, {
        "n_docs": len(docs),
        "years": str([y for y in args.years if y in AVAILABLE_YEARS]),
        "nr_topics_requested": args.nr_topics if args.nr_topics else "auto",
        "seed": args.seed,
        "nr_topics_found": n_topics_found,
        "npmi_coherence": round(coherence["average"], 4),
        "topic_diversity": round(diversity, 4),
    }, key_cols=["n_docs", "years", "nr_topics_requested", "seed"])

    print("[5/5] Saving visualization...")
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
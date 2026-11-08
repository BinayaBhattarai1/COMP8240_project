"""
Step 3: Construct the new "news pseudo-document" dataset described in the
proposal's New Data section.

The idea: collect same-day headlines + summaries from several news
outlets, and concatenate each outlet's items into one per-day
"pseudo-document". Because we know which real-world events happened on a
given day, this lets you later sanity-check whether BERTopic's discovered
topics track genuine events rather than just each outlet's writing style.

IMPORTANT -- this is meant to be run MORE THAN ONCE over time:
Running it once gives you one day's snapshot per source. For a proper
"newly constructed dataset" you should run this script daily (or every
few days) over the remaining weeks of the course -- each run appends a
new row per source, so the dataset grows into genuine longitudinal data.
That accumulation process IS the "progress" the rubric is asking you to
show evidence of (collection, preprocessing) -- so keep re-running this
and let the CSV grow before your final submission.

Usage:
    python src/03_build_dataset.py
    (re-run this on different days to grow data/constructed_news_dataset.csv)
"""

import csv
import datetime
import os

import feedparser

FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    # Reuters shut down its public RSS feeds in 2020, so that URL always
    # failed. Replaced with Deutsche Welle's world-news feed. If a feed
    # stops working, the script prints a warning and carries on.
    "DW": "https://rss.dw.com/rdf/rss-en-world",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
}

OUTPUT_PATH = "data/constructed_news_dataset.csv"
FIELDNAMES = ["date", "source", "pseudo_document", "n_headlines_used"]


def build_pseudo_documents(max_items_per_source: int = 15) -> list:
    today = datetime.date.today().isoformat()
    rows = []
    for source_name, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"    ! Could not fetch {source_name}: {e}")
            continue

        entries = feed.entries[:max_items_per_source]
        if not entries:
            print(f"    ! No entries returned for {source_name} (feed may have changed URL).")
            continue

        headline_texts = []
        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            headline_texts.append(f"{title}. {summary}".strip())

        pseudo_doc = " ".join(headline_texts)
        rows.append({
            "date": today,
            "source": source_name,
            "pseudo_document": pseudo_doc,
            "n_headlines_used": len(entries),
        })
        print(f"    -> {source_name}: {len(entries)} headlines collected.")
    return rows


def append_to_csv(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    print("[1/2] Fetching today's headlines from each news source...")
    rows = build_pseudo_documents()

    if not rows:
        print("No data collected -- check your internet connection and feed URLs.")
        return

    print(f"[2/2] Appending {len(rows)} new pseudo-documents to {OUTPUT_PATH} ...")
    append_to_csv(rows, OUTPUT_PATH)

    # Report cumulative progress so you can see the dataset growing over time
    import pandas as pd
    df = pd.read_csv(OUTPUT_PATH)
    print(f"\nDataset now contains {len(df)} pseudo-documents across "
          f"{df['date'].nunique()} day(s) and {df['source'].nunique()} source(s).")
    print("Re-run this script on future days to keep growing the dataset before submission.")


if __name__ == "__main__":
    main()

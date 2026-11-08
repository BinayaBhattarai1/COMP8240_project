"""
Step 7 (v2 COLLECTION): collect news ARTICLES, one row per article.

Run this at least once a day (twice is better) until you stop collecting.
RSS feeds only show the latest items, so a day you miss can never be
filled in later. Running it more often is harmless: articles already in
the dataset are skipped (matched by URL).

Every run also appends one line per outlet to data/v2/collection_log.csv
(how many items the feed returned and how many were new). That log is your
evidence of a systematic, repeated collection process.

Usage:
    python src/07_collect_articles.py
    python src/07_collect_articles.py --feed BBC=path/or/url.xml   # override/test a feed
"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import sys

import feedparser
import requests

try:
    import certifi
    CA_BUNDLE = certifi.where()
except ImportError:  # requests normally brings certifi with it
    CA_BUNDLE = True

# Some news sites refuse requests that don't look like a normal browser/reader.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
                         "COMP8240-news-collector",
           "Accept": "application/rss+xml, application/rdf+xml, application/xml, text/xml, */*"}

sys.path.append(os.path.dirname(__file__))
from v2_common import (FEEDS, RAW_PATH, RAW_FIELDS, COLLECTION_LOG_PATH, ensure_dirs)


def article_id(url: str, outlet: str, title: str) -> str:
    key = url.strip() if url else f"{outlet}|{title.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def fetch_feed(url):
    """Download the feed with requests (uses certifi's certificates, which avoids
    the common macOS 'CERTIFICATE_VERIFY_FAILED' problem), then parse it.
    Local file paths (used for testing) are parsed directly.
    Returns (parsed_feed, error_message_or_None)."""
    if not url.startswith(("http://", "https://")):
        return feedparser.parse(url), None
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, verify=CA_BUNDLE)
        r.raise_for_status()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    feed = feedparser.parse(r.content)
    if not feed.entries:
        return feed, f"downloaded OK (HTTP {r.status_code}) but no articles found in it"
    return feed, None


def published_iso(entry):
    """Feed publication time in UTC, or None if the feed doesn't give one."""
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)
    return None


def load_existing_ids(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["article_id"] for row in csv.DictReader(f)}


def append_rows(path, rows, fields):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", action="append", default=[],
                        help="NAME=URL to override or add a feed (repeatable).")
    args = parser.parse_args()

    feeds = dict(FEEDS)
    for spec in args.feed:
        name, url = spec.split("=", 1)
        feeds[name] = url

    ensure_dirs()
    seen = load_existing_ids(RAW_PATH)
    now = dt.datetime.now(dt.timezone.utc)
    new_rows, log_rows = [], []

    print(f"Collecting at {now.isoformat(timespec='seconds')} (UTC)")
    for outlet, url in feeds.items():
        feed, error = fetch_feed(url)
        entries = feed.entries if feed is not None else []
        status = "ok" if entries else f"FAILED: {error}"
        n_new = 0
        for e in entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title:
                continue
            aid = article_id(link, outlet, title)
            if aid in seen:
                continue
            seen.add(aid)
            pub = published_iso(e)
            new_rows.append({
                "article_id": aid,
                "outlet": outlet,
                "title": title,
                "summary": (e.get("summary") or e.get("description") or "").strip(),
                "url": link,
                "published": pub.isoformat() if pub else "",
                # Date the story belongs to: publication date if the feed gives
                # one, otherwise the day it was collected.
                "published_date": (pub or now).date().isoformat(),
                "collected_at": now.isoformat(timespec="seconds"),
            })
            n_new += 1
        log_rows.append({"collected_at": now.isoformat(timespec="seconds"),
                         "outlet": outlet, "items_in_feed": len(entries),
                         "new_articles": n_new, "status": status})
        print(f"  {outlet:<10} {len(entries):>3} in feed, {n_new:>3} new")
        if not entries:
            print(f"      ! {outlet} failed -> {str(error)[:300]}")

    if new_rows:
        append_rows(RAW_PATH, new_rows, RAW_FIELDS)
    append_rows(COLLECTION_LOG_PATH, log_rows,
                ["collected_at", "outlet", "items_in_feed", "new_articles", "status"])

    import pandas as pd
    if os.path.isfile(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        print(f"\nAdded {len(new_rows)} new articles. Dataset now: {len(df)} articles, "
              f"{df['published_date'].nunique()} day(s), {df['outlet'].nunique()} outlet(s).")
        print(df.groupby("outlet").size().rename("articles").to_string())
    else:
        print("\nNo articles collected. Read the '!' lines above for the reason and "
              "send them to Claude.")


if __name__ == "__main__":
    main()
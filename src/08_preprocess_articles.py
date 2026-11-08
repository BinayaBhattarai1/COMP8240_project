"""
Step 8 (v2 PREPROCESSING): clean the raw articles and build two text versions.

Steps (each prints how many rows it changed or removed):
  1. Strip HTML tags/entities and feed boilerplate ("Continue reading...",
     "The post ... appeared first on ...", "Read more").
  2. Remove duplicates: same URL, and the same normalised headline from the
     same outlet (feeds re-publish a story under a new URL when it is updated).
  3. Drop rows that are too short to be a news story, and non-story items
     (live blogs, video/audio-only, quizzes) if --drop_nonstory is on.
  4. Keep only a date window if --start / --end are given.
  5. Build two text columns:
       text_full   = title + full summary            (raw lengths)
       text_capped = title + first sentence of summary, cut to --cap_words
     v1 found the Guardian's documents were ~4x longer than the others, so
     length/format could explain outlet-based topics. text_capped removes
     most of that difference, so you can test whether it matters.

Usage:
    python src/08_preprocess_articles.py
    python src/08_preprocess_articles.py --start 2026-10-15 --end 2026-11-01
"""

import argparse
import html
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from v2_common import RAW_PATH, CLEAN_PATH, ensure_dirs

HTML_TAG_RE = re.compile(r"<[^>]+>")
BOILERPLATE_RES = [
    re.compile(r"continue reading(\.\.\.|…)?", re.I),
    re.compile(r"the post .{0,200}? appeared first on .{0,100}?\.?$", re.I),
    re.compile(r"\bread more\b\.?", re.I),
]
WS_RE = re.compile(r"\s+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
NONSTORY_RE = re.compile(r"^(?:live|watch|video|listen|podcast|quiz|in pictures|photos?)\b[:\s-]"
                         r"|\blive updates?\b|\bas it happened\b", re.I)


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(HTML_TAG_RE.sub(" ", text))
    for rx in BOILERPLATE_RES:
        text = rx.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def capped_text(title: str, summary: str, cap_words: int) -> str:
    first = SENT_RE.split(summary, maxsplit=1)[0] if summary else ""
    words = f"{title}. {first}".split() if first else title.split()
    return " ".join(words[:cap_words])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cap_words", type=int, default=40)
    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--start", help="first date to keep (YYYY-MM-DD)")
    p.add_argument("--end", help="last date to keep (YYYY-MM-DD)")
    p.add_argument("--keep_nonstory", action="store_true",
                   help="keep live blogs / videos / quizzes (dropped by default)")
    args = p.parse_args()

    if not os.path.isfile(RAW_PATH):
        print(f"No {RAW_PATH}. Run src/07_collect_articles.py first.")
        return
    ensure_dirs()
    df = pd.read_csv(RAW_PATH, dtype=str).fillna("")
    n0 = len(df)
    print(f"[1/5] Loaded {n0} raw articles")

    df["title_clean"] = df["title"].apply(clean)
    df["summary_clean"] = df["summary"].apply(clean)
    changed = (df["summary_clean"] != df["summary"].str.strip()).sum()
    print(f"      HTML/boilerplate cleaning changed {changed} summaries")
    # A summary that just repeats the headline adds nothing
    same = df["summary_clean"].str.lower() == df["title_clean"].str.lower()
    df.loc[same, "summary_clean"] = ""

    print("[2/5] Removing duplicates")
    n = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first") if (df["url"] != "").all() else df
    d_url = n - len(df)
    df["_nt"] = df["title_clean"].apply(norm_title)
    n = len(df)
    df = df.sort_values("collected_at").drop_duplicates(subset=["outlet", "_nt"], keep="first")
    d_title = n - len(df)
    print(f"      same URL: {d_url} | same headline, same outlet: {d_title}")

    print("[3/5] Removing non-story items and very short rows")
    n = len(df)
    if not args.keep_nonstory:
        df = df[~df["title_clean"].str.contains(NONSTORY_RE)]
    d_nonstory = n - len(df)
    df["text_full"] = (df["title_clean"] + ". " + df["summary_clean"]).str.strip(" .")
    n = len(df)
    df = df[df["text_full"].str.len() >= args.min_chars]
    d_short = n - len(df)
    print(f"      non-story (live/video/quiz): {d_nonstory} | shorter than {args.min_chars} chars: {d_short}")

    print("[4/5] Date window")
    n = len(df)
    if args.start:
        df = df[df["published_date"] >= args.start]
    if args.end:
        df = df[df["published_date"] <= args.end]
    print(f"      outside {args.start or 'start'}..{args.end or 'end'}: {n - len(df)}")

    print(f"[5/5] Building text_capped (title + first sentence, max {args.cap_words} words)")
    df["text_capped"] = [capped_text(t, s, args.cap_words)
                         for t, s in zip(df["title_clean"], df["summary_clean"])]
    df["words_full"] = df["text_full"].str.split().str.len()
    df["words_capped"] = df["text_capped"].str.split().str.len()

    cols = ["article_id", "outlet", "published_date", "published", "collected_at", "url",
            "title_clean", "summary_clean", "text_full", "text_capped",
            "words_full", "words_capped"]
    df = df.sort_values(["published_date", "outlet", "published"])[cols].reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"\nSummary: {n0} raw -> {len(df)} clean articles "
          f"({df['published_date'].nunique()} days, {df['outlet'].nunique()} outlets)")
    print("\nMean words per article, by outlet (the length confound, before and after capping):")
    print(df.groupby("outlet")[["words_full", "words_capped"]].agg(["mean", "count"])
            .round(1).to_string())
    print(f"\nSaved {CLEAN_PATH}")


if __name__ == "__main__":
    main()

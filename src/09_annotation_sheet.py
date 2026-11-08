"""
Step 9 (v2 ANNOTATION): build the event-labelling sheet(s).

Ground truth for "did BERTopic find real-world events?" has to come from a
human. You give every sampled article an EVENT ID: articles about the same
real-world story get the same ID, whichever outlet wrote them. See
ANNOTATION_GUIDE.md for the exact rules.

Commands:
  --generate   Create data/v2/annotations_A.csv (your sheet).
               Takes ALL clean articles from whole days, adding days until
               about --n articles (default 250). Whole days, not a random
               sample, so every outlet's version of a story is in the sheet.
               A "suggested_group" column (TF-IDF similarity) puts likely
               same-story articles next to each other to save you time --
               it is only a hint; you decide the event_id.
  --second     Create data/v2/annotations_B.csv for a second annotator:
               a random --n_b rows (default 75) of your sheet with NO
               suggestions and NO labels, so their judgement is independent.
  --status     Show annotation progress (your evidence of annotation work).

Usage:
    python src/09_annotation_sheet.py --generate
    python src/09_annotation_sheet.py --generate --days 2026-10-20 2026-10-21 2026-10-22
    python src/09_annotation_sheet.py --second
    python src/09_annotation_sheet.py --status
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from v2_common import CLEAN_PATH, ANNOTATION_PATH, ANNOTATION_B_PATH, ensure_dirs

LABEL_COLS = ["event_id", "event_description", "notes"]


def suggest_groups(texts, threshold=0.15):
    """Average-linkage grouping of articles by TF-IDF cosine similarity.

    Average linkage (not single linkage) so one loosely related article
    cannot chain two different stories into one giant group.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=int)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - threshold).fit_predict(X.toarray())


def generate(n_target, days, threshold, force):
    if os.path.isfile(ANNOTATION_PATH) and not force:
        print(f"{ANNOTATION_PATH} already exists -- not overwriting your labels. "
              f"Use --force to rebuild it (this DELETES existing labels).")
        return
    df = pd.read_csv(CLEAN_PATH, dtype=str).fillna("")
    if days:
        chosen = [d for d in days if d in set(df["published_date"])]
    else:
        # Most recent days with all outlets present first, then fill up to n_target
        per_day = df.groupby("published_date").agg(n=("article_id", "size"),
                                                   outlets=("outlet", "nunique"))
        full = per_day[per_day["outlets"] == df["outlet"].nunique()].index.tolist()
        order = sorted(full, reverse=True) + sorted(set(per_day.index) - set(full), reverse=True)
        chosen, total = [], 0
        for d in order:
            if total >= n_target:
                break
            chosen.append(d)
            total += per_day.loc[d, "n"]
    sheet = df[df["published_date"].isin(chosen)].copy()
    if sheet.empty:
        print("No articles for the chosen days.")
        return

    # Group per day pair window: stories run over several days, so group across the whole sheet
    sheet["suggested_group"] = suggest_groups(sheet["text_full"].tolist(), threshold)
    sizes = sheet["suggested_group"].map(sheet["suggested_group"].value_counts())
    sheet["_size"] = sizes
    sheet = sheet.sort_values(["_size", "suggested_group", "published_date", "outlet"],
                              ascending=[False, True, True, True])
    # Renumber groups in display order; singletons get blank so the hint stays readable
    remap = {g: i + 1 for i, g in enumerate(sheet["suggested_group"].unique())}
    sheet["suggested_group"] = [f"G{remap[g]:03d}" if s > 1 else ""
                                for g, s in zip(sheet["suggested_group"], sheet["_size"])]
    for c in LABEL_COLS:
        sheet[c] = ""
    out = sheet[["article_id", "published_date", "outlet", "title_clean", "summary_clean",
                 "suggested_group"] + LABEL_COLS].rename(
        columns={"title_clean": "title", "summary_clean": "summary"})
    ensure_dirs()
    out.to_csv(ANNOTATION_PATH, index=False)
    n_groups = (out["suggested_group"] != "").sum()
    print(f"Wrote {ANNOTATION_PATH}: {len(out)} articles from {len(chosen)} day(s) "
          f"{sorted(chosen)}")
    print(f"  {n_groups} articles are in suggested multi-article groups; the rest look unique.")
    print("Next: open it in Excel/Google Sheets, fill event_id (+ event_description) for "
          "every row, save as CSV with the same name. Rules: ANNOTATION_GUIDE.md")


def second(n_b, seed, force):
    if os.path.isfile(ANNOTATION_B_PATH) and not force:
        print(f"{ANNOTATION_B_PATH} already exists. Use --force to rebuild it.")
        return
    a = pd.read_csv(ANNOTATION_PATH, dtype=str).fillna("")
    # Sample whole suggested groups as well as singletons, so the subset contains
    # same-event pairs (otherwise agreement would be measured on easy "all different" cases).
    rng = np.random.default_rng(seed)
    groups = [g for g in a["suggested_group"].unique() if g]
    rng.shuffle(groups)
    picked = []
    for g in groups:
        idx = a.index[a["suggested_group"] == g].tolist()
        if len(picked) + len(idx) > n_b * 0.6:
            continue
        picked += idx
    rest = a.index[~a.index.isin(picked)].tolist()
    picked += list(rng.choice(rest, size=min(len(rest), n_b - len(picked)), replace=False))
    b = a.loc[picked, ["article_id", "published_date", "outlet", "title", "summary"]]
    b = b.sample(frac=1, random_state=seed)  # shuffle so groups aren't adjacent
    for c in LABEL_COLS:
        b[c] = ""
    b.to_csv(ANNOTATION_B_PATH, index=False)
    print(f"Wrote {ANNOTATION_B_PATH}: {len(b)} articles, shuffled, no hints or labels.")
    print("Give it to your second annotator together with ANNOTATION_GUIDE.md.")


def status():
    for name, path in [("A (you)", ANNOTATION_PATH), ("B (second annotator)", ANNOTATION_B_PATH)]:
        if not os.path.isfile(path):
            print(f"{name}: no sheet yet ({path})")
            continue
        s = pd.read_csv(path, dtype=str).fillna("")
        done = s["event_id"].str.strip() != ""
        ev = s.loc[done, "event_id"].str.strip().str.upper()
        counts = ev.value_counts()
        print(f"{name}: {done.sum()}/{len(s)} labelled ({done.mean():.0%})")
        if done.any():
            multi = counts[counts > 1]
            lab = s.loc[done].assign(event_id=ev)
            cross = lab.groupby("event_id")["outlet"].nunique()
            print(f"    distinct events: {len(counts)} | events with 2+ articles: {len(multi)} "
                  f"| events covered by 2+ outlets: {(cross > 1).sum()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument("--second", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--n_b", type=int, default=75)
    p.add_argument("--days", nargs="*")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="TF-IDF cosine for suggested groups; 0.15 suits short headline+summary text")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if not os.path.isfile(CLEAN_PATH):
        print(f"No {CLEAN_PATH}. Run src/08_preprocess_articles.py first.")
        return
    if a.generate:
        generate(a.n, a.days, a.threshold, a.force)
    if a.second:
        second(a.n_b, a.seed, a.force)
    if a.status or not (a.generate or a.second):
        status()


if __name__ == "__main__":
    main()

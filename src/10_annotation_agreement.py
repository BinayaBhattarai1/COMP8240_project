"""
Step 10 (v2 ANNOTATION QUALITY): agreement between two annotators.

Event IDs are arbitrary names ("E012" for you might be "E3" for your second
annotator), so comparing the IDs directly makes no sense. Instead, for every
PAIR of articles both people labelled, ask: "did each annotator say these two
are the same event?" Cohen's kappa is computed on those yes/no pair decisions.
Adjusted Rand Index (ARI) between the two labelings is reported as well.

Rough reading of kappa: >0.8 strong, 0.6-0.8 substantial, 0.4-0.6 moderate.

Usage:
    python src/10_annotation_agreement.py
"""

import itertools
import os
import sys

import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

sys.path.append(os.path.dirname(__file__))
from v2_common import ANNOTATION_PATH, ANNOTATION_B_PATH, RESULTS_DIR, ensure_dirs


def load(path):
    s = pd.read_csv(path, dtype=str).fillna("")
    s["event_id"] = s["event_id"].str.strip().str.upper()
    # NONNEWS rows stay in: agreeing that an item is not news is still agreement.
    return s[s["event_id"] != ""][["article_id", "event_id", "title", "outlet"]]


def main():
    if not (os.path.isfile(ANNOTATION_PATH) and os.path.isfile(ANNOTATION_B_PATH)):
        print("Need both data/v2/annotations_A.csv and data/v2/annotations_B.csv "
              "(run 09_annotation_sheet.py --second, then have both labelled).")
        return
    a, b = load(ANNOTATION_PATH), load(ANNOTATION_B_PATH)
    m = a.merge(b[["article_id", "event_id"]], on="article_id", suffixes=("_A", "_B"))
    if len(m) < 10:
        print(f"Only {len(m)} articles labelled by both -- label more before computing agreement.")
        return

    same_a, same_b, disagreements = [], [], []
    rows = m.to_dict("records")
    for r1, r2 in itertools.combinations(rows, 2):
        sa = r1["event_id_A"] == r2["event_id_A"]
        sb = r1["event_id_B"] == r2["event_id_B"]
        same_a.append(sa)
        same_b.append(sb)
        if sa != sb:
            disagreements.append((r1, r2, sa))

    kappa = cohen_kappa_score(same_a, same_b)
    ari = adjusted_rand_score(m["event_id_A"], m["event_id_B"])
    pos_a, pos_b = sum(same_a), sum(same_b)
    both = sum(x and y for x, y in zip(same_a, same_b))

    lines = [
        "Inter-annotator agreement (v2 event labels)",
        "=" * 44,
        f"Articles labelled by both annotators: {len(m)}",
        f"Article pairs compared:               {len(same_a)}",
        f"'Same event' pairs  A: {pos_a}  B: {pos_b}  both: {both}",
        f"Cohen's kappa (pairwise same-event):  {kappa:.3f}",
        f"Adjusted Rand Index (A vs B):         {ari:.3f}",
        "",
        "Disagreements (first 15) -- discuss these and refine the guideline:",
    ]
    for r1, r2, sa in disagreements[:15]:
        who = "A says SAME, B says DIFFERENT" if sa else "A says DIFFERENT, B says SAME"
        lines.append(f"  [{who}]")
        lines.append(f"     {r1['outlet']:<9} {r1['title'][:90]}")
        lines.append(f"     {r2['outlet']:<9} {r2['title'][:90]}")
    report = "\n".join(lines)
    print(report)
    ensure_dirs()
    with open(os.path.join(RESULTS_DIR, "annotation_agreement.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nSaved {RESULTS_DIR}/annotation_agreement.txt")


if __name__ == "__main__":
    main()

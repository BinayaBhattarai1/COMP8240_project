"""
Step 13 (v2 EVALUATION): does each model group articles by EVENT or by OUTLET?

Evaluates every run in results/v2/assignments/ (BERTopic and LDA, full and
capped text) and, if present, the v1 pseudo-document result for reference.

Two families of measures:

A. OUTLET effect -- uses ALL articles, no labels needed
   * NMI(topic, outlet) with a permutation test (shuffled topic labels).
     High NMI + small p = topics line up with outlets.

B. EVENT tracking -- uses the articles you labelled in annotations_A.csv
   * NMI and ARI between topics and your event IDs.
   * Cross-outlet event pairs kept together: of all pairs of articles that
     cover the SAME event but come from DIFFERENT outlets, the share placed
     in the same topic. High = the model tracks events across outlets.
   * Same-outlet unrelated pairs grouped: of all pairs from the SAME outlet
     about DIFFERENT events, the share placed in the same topic. Compare it
     with the same rate for different-outlet unrelated pairs: if same-outlet
     pairs are grouped much more often, outlet style is pulling topics together.
   Outliers (BERTopic topic -1) never count as "together".

Outputs:
  results/v2/evaluation.csv           one row per run
  results/v2/evaluation_report.txt    readable report
  results/v2/event_vs_outlet_pairs.png  figure for the report/video

Usage:
    python src/13_evaluate_event_vs_outlet.py
"""

import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.append(os.path.dirname(__file__))
from v2_common import CLEAN_PATH, ANNOTATION_PATH, RESULTS_DIR, ensure_dirs

N_PERM = 2000
V1_ASSIGN = "results/constructed_dataset_assignments.csv"


def perm_test(labels, topics, rng):
    obs = normalized_mutual_info_score(labels, topics)
    null = np.array([normalized_mutual_info_score(labels, rng.permutation(topics))
                     for _ in range(N_PERM)])
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (N_PERM + 1))


def pair_rates(df):
    """df: rows with outlet, event_id, topic (labelled articles only)."""
    c = {k: [0, 0] for k in ["same_event_cross", "same_event_same", "diff_event_same", "diff_event_cross"]}
    recs = df[["outlet", "event_id", "topic"]].to_numpy()
    for (o1, e1, t1), (o2, e2, t2) in itertools.combinations(recs, 2):
        key = ("same_event_" if e1 == e2 else "diff_event_") + ("same" if o1 == o2 else "cross")
        c[key][1] += 1
        c[key][0] += int(t1 == t2 and t1 != -1)
    return {k: (v[0] / v[1] if v[1] else np.nan, v[1]) for k, v in c.items()}


def evaluate_run(name, assign, clean, ann, rng):
    d = clean.merge(assign, on="article_id", how="inner")
    row = {"run": name, "n_articles": len(d),
           "n_topics": d.loc[d["topic"] != -1, "topic"].nunique(),
           "outlier_share": round(float((d["topic"] == -1).mean()), 3)}
    cl = d[d["topic"] != -1]
    nmi_o, null_o, p_o = perm_test(cl["outlet"].to_numpy(), cl["topic"].to_numpy(), rng)
    row.update({"nmi_outlet": round(nmi_o, 3), "nmi_outlet_shuffled": round(null_o, 3),
                "p_outlet": round(p_o, 4)})
    if ann is not None:
        la = d.merge(ann, on="article_id", how="inner")
        row["n_labelled"] = len(la)
        lc = la[la["topic"] != -1]
        if len(lc) > 5:
            nmi_e, null_e, p_e = perm_test(lc["event_id"].to_numpy(), lc["topic"].to_numpy(), rng)
            row.update({"nmi_event": round(nmi_e, 3), "nmi_event_shuffled": round(null_e, 3),
                        "p_event": round(p_e, 4),
                        "ari_event": round(adjusted_rand_score(lc["event_id"], lc["topic"]), 3)})
        rates = pair_rates(la)
        for k, (r, n) in rates.items():
            row[f"together_{k}"] = round(r, 3) if not np.isnan(r) else np.nan
            row[f"pairs_{k}"] = n
    return row


def v1_reference(rng):
    if not os.path.isfile(V1_ASSIGN):
        return None
    v1 = pd.read_csv(V1_ASSIGN)
    cl = v1[v1["topic"] != -1]
    if cl["topic"].nunique() < 2:
        return None
    nmi, null, p = perm_test(cl["source"].to_numpy(), cl["topic"].to_numpy(), rng)
    return {"run": "v1_pseudo_documents (reference)", "n_articles": len(v1),
            "n_topics": cl["topic"].nunique(),
            "outlier_share": round(float((v1["topic"] == -1).mean()), 3),
            "nmi_outlet": round(nmi, 3), "nmi_outlet_shuffled": round(null, 3),
            "p_outlet": round(p, 4)}


def make_figure(ev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    need = ["together_same_event_cross", "together_diff_event_same", "together_diff_event_cross"]
    sub = ev.dropna(subset=need)
    sub = sub[~sub["run"].str.startswith("v1")]
    if sub.empty:
        return False
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
    ink, ink2 = "#0b0b0b", "#52514e"
    labels = sub["run"].str.replace("_seed42", "").tolist()
    y = np.arange(len(sub))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(sub) + 1.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    series = [("Same event, different outlets", sub[need[0]], BLUE),
              ("Different events, same outlet", sub[need[1]], ORANGE),
              ("Different events, different outlets", sub[need[2]], GREY)]
    for i, (lab, vals, col) in enumerate(series):
        bars = ax.barh(y + (i - 1) * h, vals, height=h - 0.04, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                    va="center", fontsize=8, color=ink2)
    ax.set_yticks(y, labels, fontsize=9, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of article pairs placed in the same topic", color=ink2, fontsize=9)
    ax.tick_params(colors=ink2, labelsize=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Do topics follow events or outlets?", loc="left", fontsize=12, color=ink, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
              labelcolor=ink2, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "assignments", "*.csv")))
    if not files:
        print("No runs found. Run src/11_bertopic_articles.py (and 12_lda_baseline.py) first.")
        return
    ensure_dirs()
    rng = np.random.default_rng(0)
    clean = pd.read_csv(CLEAN_PATH, dtype={"article_id": str})[["article_id", "outlet"]]
    ann = None
    if os.path.isfile(ANNOTATION_PATH):
        a = pd.read_csv(ANNOTATION_PATH, dtype=str).fillna("")
        a["event_id"] = a["event_id"].str.strip().str.upper()
        # NONNEWS = items that are not a news story (see ANNOTATION_GUIDE.md)
        a = a[(a["event_id"] != "") & (a["event_id"] != "NONNEWS")][["article_id", "event_id"]]
        ann = a if len(a) else None
    if ann is None:
        print("No event labels yet -- reporting the OUTLET effect only. "
              "Label data/v2/annotations_A.csv for the event measures.\n")

    rows = []
    v1 = v1_reference(rng)
    if v1:
        rows.append(v1)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        rows.append(evaluate_run(name, pd.read_csv(f, dtype={"article_id": str}), clean, ann, rng))
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RESULTS_DIR, "evaluation.csv"), index=False)

    L = ["v2 evaluation: events or outlets?", "=" * 34, ""]
    L.append("A. OUTLET effect (all articles, outliers excluded)")
    L.append(f"   {'run':<42}{'topics':>7}{'outl.':>7}{'NMI':>7}{'shuf.':>7}{'p':>8}")
    for r in rows:
        L.append(f"   {r['run'][:41]:<42}{r['n_topics']:>7}{r['outlier_share']:>7.0%}"
                 f"{r['nmi_outlet']:>7.3f}{r['nmi_outlet_shuffled']:>7.3f}{r['p_outlet']:>8.4f}")
    if ann is not None and "nmi_event" in ev:
        L += ["", f"B. EVENT tracking ({len(ann)} labelled articles, "
                  f"{ann['event_id'].nunique()} events)"]
        L.append(f"   {'run':<42}{'NMI ev':>7}{'ARI':>7}{'SE/XO':>8}{'DE/SO':>8}{'DE/XO':>8}")
        for r in rows:
            if "nmi_event" not in r or pd.isna(r.get("nmi_event")):
                continue
            L.append(f"   {r['run'][:41]:<42}{r['nmi_event']:>7.3f}{r['ari_event']:>7.3f}"
                     f"{r['together_same_event_cross']:>8.0%}{r['together_diff_event_same']:>8.0%}"
                     f"{r['together_diff_event_cross']:>8.0%}")
        any_r = next(r for r in rows if "pairs_same_event_cross" in r)
        L += ["",
              "   SE/XO = same event, different outlets: share of pairs in the same topic (higher = tracks events)",
              "   DE/SO = different events, same outlet: share grouped together",
              "   DE/XO = different events, different outlets: share grouped together (background rate)",
              "   If DE/SO is well above DE/XO, outlet style is still pulling articles together.",
              f"   Pair counts: SE/XO {any_r['pairs_same_event_cross']}, "
              f"DE/SO {any_r['pairs_diff_event_same']}, DE/XO {any_r['pairs_diff_event_cross']}"]
    L += ["", "Compare 'full' vs 'capped' runs: if the outlet NMI drops when text length is",
          "equalised, part of the v1 outlet effect was a length/format artefact."]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
        f.write(report + "\n")
    if ann is not None and make_figure(ev, os.path.join(RESULTS_DIR, "event_vs_outlet_pairs.png")):
        print(f"\nFigure: {RESULTS_DIR}/event_vs_outlet_pairs.png")
    print(f"Saved {RESULTS_DIR}/evaluation.csv and evaluation_report.txt")


if __name__ == "__main__":
    main()

"""
Step 12 (v2 BASELINE): LDA on the same articles, for comparison with BERTopic.

The BERTopic paper compares against LDA (among others). Running LDA on the
same documents answers: is the event-vs-outlet pattern specific to
BERTopic's embeddings, or does a bag-of-words model show it too?

By default k = the number of topics the matching BERTopic run found
(read from results/v2/runs_summary.csv), so the two are comparable.

Usage:
    python src/12_lda_baseline.py --text full
    python src/12_lda_baseline.py --text capped --k 40
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

sys.path.append(os.path.dirname(__file__))
from utils import compute_npmi_coherence, compute_topic_diversity, update_sweep_summary
from v2_common import CLEAN_PATH, RESULTS_DIR, OUTLET_STOP_WORDS, TEXT_VARIANTS, ensure_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", choices=TEXT_VARIANTS, default="full")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    summ_path = os.path.join(RESULTS_DIR, "runs_summary.csv")
    k = a.k
    if k is None and os.path.isfile(summ_path):
        s = pd.read_csv(summ_path)
        s = s[(s["model"] == "BERTopic") & (s["text"] == a.text) & (~s["offline"].astype(bool))]
        if len(s):
            k = int(s.iloc[-1]["n_topics"])
    k = k or 30
    ensure_dirs()
    os.makedirs(os.path.join(RESULTS_DIR, "assignments"), exist_ok=True)

    col = TEXT_VARIANTS[a.text]
    df = pd.read_csv(CLEAN_PATH, dtype={"article_id": str})
    df = df[df[col].fillna("").str.strip() != ""].reset_index(drop=True)
    docs = df[col].tolist()
    stop = sorted(ENGLISH_STOP_WORDS.union(OUTLET_STOP_WORDS))
    vec = CountVectorizer(stop_words=stop, min_df=2)
    X = vec.fit_transform(docs)
    print(f"LDA: {len(docs)} articles, vocabulary {X.shape[1]}, k = {k}, text = {a.text}")
    lda = LatentDirichletAllocation(n_components=k, random_state=a.seed, learning_method="batch",
                                    max_iter=50)
    theta = lda.fit_transform(X)
    topics = theta.argmax(axis=1)

    vocab = np.array(vec.get_feature_names_out())
    words = {t: [(w, float(lda.components_[t][i]))
                 for i, w in zip(lda.components_[t].argsort()[::-1][:10],
                                 vocab[lda.components_[t].argsort()[::-1][:10]])]
             for t in range(k)}
    coh = compute_npmi_coherence(words, docs, top_n=10, extra_stop_words=OUTLET_STOP_WORDS)
    div = compute_topic_diversity(words, top_n=10)

    run = f"lda_{a.text}_k{k}_seed{a.seed}"
    pd.DataFrame({"article_id": df["article_id"], "topic": topics}).to_csv(
        os.path.join(RESULTS_DIR, "assignments", f"{run}.csv"), index=False)
    pd.DataFrame([{"topic": t, "count": int((topics == t).sum()),
                   "words": "_".join(w for w, _ in words[t])} for t in range(k)]).to_csv(
        os.path.join(RESULTS_DIR, f"topics_{run}.csv"), index=False)
    update_sweep_summary(summ_path, {
        "run": run, "model": "LDA", "text": a.text, "min_topic_size": "",
        "nr_topics": k, "seed": a.seed, "offline": False, "n_docs": len(docs),
        "n_topics": k, "outlier_share": 0.0, "npmi": round(coh["average"], 4),
        "diversity": round(div, 4)}, ["run"])
    print(f"{run}: NPMI {coh['average']:.4f}, diversity {div:.4f}")


if __name__ == "__main__":
    main()

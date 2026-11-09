"""
Step 15 (EVALUATION): LLM-as-judge for topic quality (proposal Section 2.3).

NPMI only measures word co-occurrence. Following the proposal, an LLM is
asked two questions about every topic, without being told which model made it:

  Q1 Interpretability (words only, 1-3), as in Stammbach et al. (2023):
       3 = very related words, a clear theme; 2 = somewhat related; 1 = not related.
  Q2 Granularity (v2 constructed data only; words + 3 sample headlines):
       EVENT  = one specific real-world news story
       THEME  = a broad subject spanning several stories (e.g. "Middle East")
       MIXED  = unrelated stories / no coherent subject
     Q2 is the LLM version of this project's research question: do topics
     follow events?

Q2 is then checked against the human labels: for each topic, "purity" = share
of its labelled articles that belong to its most common human event. If the
judge is useful, EVENT topics should have clearly higher purity than MIXED ones.

Providers:
  --provider hf         open instruct model via transformers (default
                        Qwen/Qwen2.5-3B-Instruct; fits a free Colab T4 GPU)
  --provider anthropic  Claude API; needs ANTHROPIC_API_KEY (pip install anthropic)
  --provider mock       random answers, ONLY to test the pipeline offline

Usage:
    python src/15_llm_judge.py                              # all real v2 runs
    python src/15_llm_judge.py --runs bertopic_capped_mts5_seed1 lda_capped_k40_seed42
    python src/15_llm_judge.py --extra "results/*_topics_k*.csv"   # replication topics, Q1 only
    python src/15_llm_judge.py --provider anthropic --model claude-haiku-5-5
"""

import argparse
import ast
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from v2_common import CLEAN_PATH, ANNOTATION_PATH, RESULTS_DIR, ensure_dirs

OUT_DIR = os.path.join(RESULTS_DIR, "llm_judge")

Q1 = ("You are evaluating topics produced by a topic model.\n"
      "Here are the top words of one topic:\n{words}\n\n"
      "Rate how related these words are to each other:\n"
      "3 = very related, they clearly describe one theme\n"
      "2 = somewhat related\n"
      "1 = not very related\n"
      "Answer with a single digit (1, 2 or 3) and nothing else.")

Q2 = ("You are evaluating a topic found in a collection of world news articles.\n"
      "Top words: {words}\n"
      "Example headlines from the topic:\n{heads}\n\n"
      "Which description fits this topic best?\n"
      "EVENT = the articles are about one specific real-world news story\n"
      "THEME = a broad subject covering several different stories\n"
      "MIXED = unrelated stories with no coherent subject\n"
      "Answer with one word: EVENT, THEME or MIXED.")


# ---------------------------------------------------------------- providers
class Judge:
    def __init__(self, provider, model, seed=0):
        self.provider = provider
        self.rng = np.random.default_rng(seed)
        if provider == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            try:   # newer transformers call it dtype, older ones torch_dtype
                self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map="auto")
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype,
                                                                  device_map="auto")
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        self.model_name = model

    def ask(self, prompt, max_new_tokens=5):
        if self.provider == "mock":
            return str(self.rng.choice(["1", "2", "3"])) if "single digit" in prompt else \
                str(self.rng.choice(["EVENT", "THEME", "MIXED"]))
        if self.provider == "anthropic":
            r = self.client.messages.create(model=self.model_name, max_tokens=max_new_tokens,
                                            temperature=0,
                                            messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        import torch
        msgs = [{"role": "user", "content": prompt}]
        # Build the prompt as text, then tokenise: works the same on old and new
        # transformers (newer versions return a dict from apply_chat_template).
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_rating(text):
    m = re.search(r"[123]", text)
    return int(m.group()) if m else np.nan


def parse_kind(text):
    m = re.search(r"EVENT|THEME|MIXED", text.upper())
    return m.group() if m else ""


# ---------------------------------------------------------------- topic loading
def topic_words_from_csv(path):
    """{topic_id: [words]} from a BERTopic topic_info CSV or an LDA topics CSV."""
    t = pd.read_csv(path)
    out = {}
    if "Representation" in t:
        for _, r in t.iterrows():
            if int(r["Topic"]) == -1:
                continue
            words = [w for w in ast.literal_eval(r["Representation"]) if w][:10]
            out[int(r["Topic"])] = words
    elif "words" in t:
        for _, r in t.iterrows():
            out[int(r["topic"])] = str(r["words"]).split("_")[:10]
    return out


def sample_headlines(run, clean, k=3, seed=0):
    """3 random headlines per topic from the run's assignments (same rule for every model)."""
    p = os.path.join(RESULTS_DIR, "assignments", f"{run}.csv")
    if not os.path.isfile(p):
        return {}
    a = pd.read_csv(p, dtype={"article_id": str}).merge(clean, on="article_id")
    return {int(t): g.sample(min(k, len(g)), random_state=seed)["title_clean"].tolist()
            for t, g in a.groupby("topic") if t != -1}


def purity_by_topic(run, ann):
    p = os.path.join(RESULTS_DIR, "assignments", f"{run}.csv")
    if ann is None or not os.path.isfile(p):
        return {}
    a = pd.read_csv(p, dtype={"article_id": str}).merge(ann, on="article_id")
    res = {}
    for t, g in a.groupby("topic"):
        if t != -1 and len(g) >= 2:
            res[int(t)] = (g["event_id"].value_counts().iloc[0] / len(g), len(g))
    return res


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["hf", "anthropic", "mock"], default="hf")
    p.add_argument("--model", default=None)
    p.add_argument("--runs", nargs="*", help="v2 run names (default: all non-offline runs)")
    p.add_argument("--extra", nargs="*", default=[],
                   help="glob(s) of other topic CSVs (e.g. replication runs) -- Q1 only")
    p.add_argument("--extra_only", action="store_true",
                   help="judge only the --extra files (writes llm_judge_summary_extra.csv)")
    p.add_argument("--max_topics", type=int, default=60,
                   help="judge at most this many topics per run (largest first)")
    a = p.parse_args()
    model = a.model or {"hf": "Qwen/Qwen2.5-3B-Instruct", "anthropic": "claude-haiku-5-5",
                        "mock": "mock"}[a.provider]

    ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = pd.read_csv(CLEAN_PATH, dtype={"article_id": str})[["article_id", "title_clean"]] \
        if os.path.isfile(CLEAN_PATH) else None
    ann = None
    if os.path.isfile(ANNOTATION_PATH):
        s = pd.read_csv(ANNOTATION_PATH, dtype=str).fillna("")
        s["event_id"] = s["event_id"].str.strip().str.upper()
        s = s[(s["event_id"] != "") & (s["event_id"] != "NONNEWS")][["article_id", "event_id"]]
        ann = s if len(s) else None

    jobs = []  # (run_name, topics_csv, is_v2)
    if a.extra_only:
        names = []
    elif a.runs:
        names = a.runs
    else:
        names = [os.path.basename(f)[7:-4] for f in glob.glob(os.path.join(RESULTS_DIR, "topics_*.csv"))]
        if a.provider != "mock":
            names = [n for n in names if "OFFLINE" not in n]
    for n in sorted(names):
        jobs.append((n, os.path.join(RESULTS_DIR, f"topics_{n}.csv"), True))
    for g in a.extra:
        for f in sorted(glob.glob(g)):
            jobs.append((os.path.basename(f)[:-4], f, False))
    if not jobs:
        print("No topic files found. Run 11_bertopic_articles.py / 12_lda_baseline.py first.")
        return

    judge = Judge(a.provider, model)
    print(f"Judge: {a.provider} / {model}  |  {len(jobs)} run(s)")
    summary = []
    for run, path, is_v2 in jobs:
        words = topic_words_from_csv(path)
        heads = sample_headlines(run, clean) if (is_v2 and clean is not None) else {}
        pur = purity_by_topic(run, ann) if is_v2 else {}
        # largest topics first, capped -- keeps runtime and API cost bounded
        sizes = {}
        if is_v2:
            ap = os.path.join(RESULTS_DIR, "assignments", f"{run}.csv")
            if os.path.isfile(ap):
                sizes = pd.read_csv(ap)["topic"].value_counts().to_dict()
        order = sorted(words, key=lambda t: -sizes.get(t, 0))[:a.max_topics]
        rows = []
        for t in order:
            r = {"run": run, "topic": t, "words": " ".join(words[t]),
                 "rating": parse_rating(judge.ask(Q1.format(words=", ".join(words[t]))))}
            if t in heads:
                hs = "\n".join(f"- {h}" for h in heads[t])
                r["kind"] = parse_kind(judge.ask(Q2.format(words=", ".join(words[t]), heads=hs)))
                r["headlines"] = " | ".join(heads[t])
            if t in pur:
                r["human_purity"], r["n_labelled"] = round(pur[t][0], 3), pur[t][1]
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"{run}.csv"), index=False)
        s = {"run": run, "judge": f"{a.provider}:{model}", "topics_judged": len(df),
             "mean_rating": round(df["rating"].mean(), 3),
             "share_rating_3": round((df["rating"] == 3).mean(), 3)}
        if "kind" in df:
            for k in ["EVENT", "THEME", "MIXED"]:
                s[f"share_{k}"] = round((df["kind"] == k).mean(), 3)
        if "human_purity" in df:
            for k in ["EVENT", "THEME", "MIXED"]:
                v = df.loc[df.get("kind", pd.Series(dtype=str)) == k, "human_purity"]
                s[f"purity_{k}"] = round(v.mean(), 3) if len(v) else np.nan
        summary.append(s)
        print(f"  {run}: mean rating {s['mean_rating']:.2f}"
              + (f", EVENT {s['share_EVENT']:.0%} THEME {s['share_THEME']:.0%} "
                 f"MIXED {s['share_MIXED']:.0%}" if "share_EVENT" in s else ""))

    out = pd.DataFrame(summary)
    name = "llm_judge_summary_extra.csv" if a.extra_only else "llm_judge_summary.csv"
    out.to_csv(os.path.join(OUT_DIR, name), index=False)
    print(f"\nSaved per-topic judgements and {OUT_DIR}/{name}")
    if "purity_EVENT" in out:
        print("Check: purity_EVENT should be well above purity_MIXED if the judge agrees "
              "with your human event labels.")


if __name__ == "__main__":
    main()

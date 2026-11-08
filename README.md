# Reproducing BERTopic (COMP8240, Macquarie University, Session 2 2026)

Replication of Grootendorst (2022), *BERTopic: Neural topic modeling with a
class-based TF-IDF procedure*, on its three original datasets, plus two new
datasets: Amazon product reviews (existing) and a **newly constructed
multi-outlet world-news dataset** with human event labels.

Author: Binaya Bhattarai (48745219)

## Quick start

```bash
pip install bertopic sentence-transformers feedparser accelerate tabulate
```

Every script accepts `--offline` (TF-IDF stand-in embeddings, no download) to test
the pipeline in seconds. Reported numbers use real sentence-transformer embeddings.
`run_v2_pipeline.ipynb` runs the whole constructed-dataset pipeline on Google Colab.

## Scripts (`src/`)

| Step | Script | What it does |
| --- | --- | --- |
| Replication | `01_reproduce_original.py` | 20 Newsgroups, NPMI + diversity (`--nr_topics`, `--seed`) |
| | `05_bbc_news.py` | BBC News (Greene & Cunningham, 2006) + category alignment |
| | `06_trumps_tweets.py` | Trump's tweets 2009–2018 (public archive) |
| New existing data | `02_new_existing_dataset.py` | Amazon Polarity reviews (Hugging Face) |
| Constructed data v1 | `03`, `03b`, `03c`, `04`, `04b` | Daily per-outlet pseudo-documents (pilot; see report) |
| Constructed data v2 | `07_collect_articles.py` | Collect RSS articles from BBC, Guardian, Al Jazeera, DW, NPR (run daily) |
| | `08_preprocess_articles.py` | Clean, de-duplicate, drop non-stories; full and length-capped text |
| | `09_annotation_sheet.py` | Build event-labelling sheets A (main) and B (second annotator) |
| | `10_annotation_agreement.py` | Cohen's kappa (pairwise same-event) and ARI |
| | `11_bertopic_articles.py` | BERTopic on articles (`--text full/capped`, `--seed`, `--embedding`) |
| | `12_lda_baseline.py` | LDA on the same articles |
| | `13_evaluate_event_vs_outlet.py` | Do topics follow events or outlets? NMI + permutation tests, pair rates |
| | `14_report_tables.py` | Report tables (CSV + LaTeX), figures, `REPORT_NUMBERS.md` |
| | `15_llm_judge.py` | LLM-as-judge: topic interpretability (1–3) and EVENT/THEME/MIXED |
| Shared | `utils.py`, `v2_common.py` | Model construction, metrics, paths |

Annotation rules: `ANNOTATION_GUIDE.md`.

## Constructed dataset: order of operations

```bash
python src/07_collect_articles.py                 # once or twice a day over the collection period
python src/08_preprocess_articles.py
python src/09_annotation_sheet.py --generate      # label data/v2/annotations_A.csv by hand
python src/09_annotation_sheet.py --second        # second annotator labels annotations_B.csv
python src/10_annotation_agreement.py
python src/11_bertopic_articles.py --text full   --seed 1   # repeat: seeds 1-3, text full/capped
python src/12_lda_baseline.py      --text capped --k 40 --seed 1
python src/13_evaluate_event_vs_outlet.py
python src/15_llm_judge.py
python src/14_report_tables.py
```

## Implementation notes

- **Metrics.** NPMI is computed with document-level co-occurrence over the
  modelled corpus (not OCTIS's sliding window over a reference corpus), so
  absolute values are not directly comparable with the paper; pairs that never
  co-occur score −1 rather than being skipped. Diversity = unique share of the
  top-10 words across topics.
- **`min_df` fix.** BERTopic fits its vectorizer on one joined document per
  topic, so passing `min_df=2` to it removes words unique to a single topic.
  `utils.document_level_vectorizer` applies `min_df` to the original documents
  and gives BERTopic a fixed vocabulary.
- **Reproducibility.** UMAP is seeded (`--seed`); results are reported as
  mean ± sd over seeds 1–3.

## Data

`data/v2/` holds the constructed dataset: headlines and the short summaries
published in each outlet's public RSS feed, with source URLs and dates,
collected for non-commercial research. Full article text was never collected.
`data/v2/annotations_*.csv` hold the human event labels. The original datasets
(20 Newsgroups, BBC News, Trump's tweets, Amazon Polarity) are downloaded by the
scripts from their public sources and are not stored in this repository.

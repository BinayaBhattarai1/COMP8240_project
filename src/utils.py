"""
Shared utilities for the BERTopic reproduction project.

This module provides:
  1. get_embedding_backend() - a swappable embedding model. By default it
     uses a real sentence-transformer (as in the original paper). Pass
     offline=True to use a fast, fully local TF-IDF + SVD fallback that
     needs no internet connection -- useful for smoke-testing the whole
     pipeline before committing to a slow real run.
  2. build_topic_model() - one place where every script builds BERTopic,
     so all runs share the same settings: an English-stopword
     CountVectorizer for topic representations (without it, topic labels
     come out as "the_to_of_and") and a seeded UMAP for reproducibility.
  3. compute_npmi_coherence() / compute_topic_diversity() - manual
     re-implementations of the two metrics described in the proposal
     (Section 2.1), so you are not dependent on the heavier OCTIS toolkit
     just to get a coherence/diversity number for your video.

NOTE ON THE NPMI APPROXIMATION:
The original paper (and OCTIS) computes NPMI using a sliding window over
a large reference corpus. Implementing a true sliding-window co-occurrence
counter is a reasonable next step but is not required for this project --
here we use the standard, widely-used simplification where two words
"co-occur" if they both appear anywhere in the same document (this is the
same approximation gensim's CoherenceModel uses when window_size=None).
This is worth mentioning explicitly in your video/report so it's clear
you understand the metric rather than treating it as a black box.

Two details make this version defensible:
  - Documents are tokenised with the SAME analyzer BERTopic uses to build
    topic words (lowercase, punctuation stripped, stopwords removed), so
    "god," and "god" are not treated as different words.
  - A word pair that never co-occurs gets NPMI = -1 (its theoretical
    minimum) instead of being skipped. Skipping such pairs only keeps the
    pairs that happen to co-occur, which biases coherence upwards --
    especially for runs with hundreds of small topics.
"""

import itertools
import math
import os
from collections import Counter

import numpy as np


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # the paper used all-mpnet-base-v2


def get_embedding_backend(offline: bool = False, model_name: str = None):
    """
    Returns an object with an .encode(list_of_strings) -> np.ndarray method,
    which is what BERTopic expects for its embedding_model argument.

    offline=False (default): real sentence-transformer, matching the
        original paper's methodology. Requires internet access the first
        time it runs (to download the pretrained model).
    offline=True: a local TF-IDF + TruncatedSVD embedding. No internet
        required, runs in seconds. Good for a fast end-to-end smoke test
        of the whole pipeline, but is NOT a faithful reproduction of the
        paper's method -- only use offline mode to check your code works,
        then re-run with offline=False for your actual reported results.
    """
    if not offline:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name or DEFAULT_EMBEDDING_MODEL)

    # --- Offline fallback: TF-IDF -> dense vectors via truncated SVD ---
    # BERTopic requires custom embedders to subclass its BaseEmbedder and
    # implement .embed() (NOT a duck-typed .encode() method) -- passing a
    # plain object silently fails BERTopic's backend-type check and it
    # falls back to trying to download the real default model, causing a
    # confusing network error even though you asked for offline mode.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from bertopic.backend import BaseEmbedder

    class OfflineEmbedder(BaseEmbedder):
        def __init__(self, n_components=50):
            super().__init__()
            self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
            self.n_components = n_components
            self._fitted = False

        def embed(self, documents, verbose=False):
            documents = list(documents)
            if not self._fitted:
                tfidf = self.vectorizer.fit_transform(documents)
                self._fitted = True
            else:
                tfidf = self.vectorizer.transform(documents)
            # SVD needs n_components < min(n_samples, n_features); guard small corpora
            n_comp = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            return svd.fit_transform(tfidf)

    return OfflineEmbedder()


def get_vectorizer(min_df: int = 2, extra_stop_words=None):
    """
    CountVectorizer used for BERTopic's topic representations (c-TF-IDF).

    BERTopic's default vectorizer keeps stopwords, which is why earlier runs
    produced topic labels like "the_to_of_and" and low topic diversity.
    min_df=2 drops words seen in only one document (typos, IDs); use
    min_df=1 for very small corpora such as the constructed dataset.
    """
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    stop_words = "english"
    if extra_stop_words:
        stop_words = sorted(ENGLISH_STOP_WORDS.union(w.lower() for w in extra_stop_words))
    return CountVectorizer(stop_words=stop_words, min_df=min_df)


def document_level_vectorizer(docs, min_df: int = 2, extra_stop_words=None):
    """
    CountVectorizer whose vocabulary keeps words found in >= min_df DOCUMENTS.

    Why this exists (bug fix, Nov 2026): BERTopic fits its vectorizer_model on
    ONE joined document per topic, not on the original documents. Passing
    min_df=2 straight to BERTopic therefore keeps only words that occur in at
    least two TOPICS -- which deletes exactly the words unique to one topic
    (e.g. a person or place that names a news event) and leaves topics
    labelled with words shared across topics. Here the vocabulary is decided
    on the original documents, and BERTopic receives a fixed vocabulary.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    base = get_vectorizer(min_df=min_df, extra_stop_words=extra_stop_words)
    base.fit(docs)
    return CountVectorizer(stop_words=base.stop_words, vocabulary=sorted(base.vocabulary_))


def build_topic_model(embedding_model, min_topic_size: int = 10, nr_topics=None,
                      min_df: int = 2, seed: int = 42, n_docs=None,
                      extra_stop_words=None, docs=None):
    """
    Builds BERTopic with the project's standard settings.

    The UMAP settings are BERTopic's own defaults, plus random_state so a
    rerun gives the same topics (UMAP is otherwise non-deterministic, which
    is why repeated "auto" runs gave slightly different numbers before).
    For tiny corpora, n_neighbors is capped below the number of documents.

    Pass docs= so min_df is applied per document (see document_level_vectorizer).
    Without docs, min_df is NOT passed to BERTopic (min_df=1), because inside
    BERTopic it would count topics, not documents.
    """
    from bertopic import BERTopic
    from umap import UMAP

    if docs is not None and n_docs is None:
        n_docs = len(docs)
    n_neighbors = 15
    if n_docs is not None:
        n_neighbors = max(2, min(15, n_docs - 1))
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    if docs is not None and min_df > 1:
        vectorizer = document_level_vectorizer(docs, min_df, extra_stop_words)
    else:
        vectorizer = get_vectorizer(min_df=1, extra_stop_words=extra_stop_words)
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        verbose=True,
    )


def compute_npmi_coherence(topics_words: dict, docs: list, top_n: int = 10,
                           extra_stop_words=None) -> dict:
    """
    topics_words: dict mapping topic_id -> [(word, score), ...], as returned
                  by BERTopic's get_topics().
    docs: the documents used to fit the model (the reference corpus for
          co-occurrence statistics).
    top_n: how many top words per topic to use.

    Returns: {topic_id: coherence_score, ..., "average": overall_average}
    """
    analyzer = get_vectorizer(min_df=1, extra_stop_words=extra_stop_words).build_analyzer()

    # Inverted index: word -> set of document ids containing it
    word_docs = {}
    for i, doc in enumerate(docs):
        for w in set(analyzer(doc)):
            word_docs.setdefault(w, set()).add(i)
    n_docs = len(docs)

    results = {}
    all_scores = []
    for topic_id, word_score_pairs in topics_words.items():
        if topic_id == -1:
            continue  # skip BERTopic's outlier "topic"
        words = [w for w, _ in word_score_pairs[:top_n] if w]
        pair_scores = []
        for w1, w2 in itertools.combinations(words, 2):
            d1, d2 = word_docs.get(w1, set()), word_docs.get(w2, set())
            if not d1 or not d2:
                continue  # word not in reference corpus (e.g. a bigram); no evidence either way
            d12 = len(d1 & d2)
            if d12 == 0:
                pair_scores.append(-1.0)  # never co-occur: minimum NPMI, not skipped
                continue
            p_w1, p_w2, p_w12 = len(d1) / n_docs, len(d2) / n_docs, d12 / n_docs
            if p_w12 == 1.0:
                pair_scores.append(1.0)  # both words in every document
                continue
            npmi = math.log(p_w12 / (p_w1 * p_w2)) / -math.log(p_w12)
            pair_scores.append(npmi)
        topic_score = float(np.mean(pair_scores)) if pair_scores else 0.0
        results[topic_id] = topic_score
        all_scores.append(topic_score)

    results["average"] = float(np.mean(all_scores)) if all_scores else 0.0
    return results


def compute_topic_diversity(topics_words: dict, top_n: int = 10) -> float:
    """
    Proportion of unique words across the top-N words of every topic
    (excluding the outlier topic -1). 1.0 = no redundancy, 0.0 = every
    topic shares the exact same words.
    """
    all_words = []
    for topic_id, word_score_pairs in topics_words.items():
        if topic_id == -1:
            continue
        # BERTopic pads topics that have fewer than top_n words with "" -- those
        # placeholders are not words and must not count as repeated words.
        all_words.extend(w for w, _ in word_score_pairs[:top_n] if w)

    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def update_sweep_summary(path: str, row: dict, key_cols: list):
    """
    Writes one result row to a sweep-summary CSV, REPLACING any earlier row
    with the same configuration (key_cols) instead of appending a duplicate.
    """
    import pandas as pd
    new = pd.DataFrame([row])
    if os.path.isfile(path):
        old = pd.read_csv(path)
        if all(c in old.columns for c in key_cols):
            mask = np.ones(len(old), dtype=bool)
            for c in key_cols:
                mask &= old[c].astype(str) == str(row[c])
            old = old[~mask]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)

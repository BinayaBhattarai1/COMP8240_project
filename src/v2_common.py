"""
Shared settings for the v2 (article-level) constructed dataset.

v1 (scripts 03-04b) joined ~15 headlines from one outlet into a single
per-day "pseudo-document". Each v1 document therefore mixed many unrelated
stories, and the only thing guaranteed to be shared inside a document was
the outlet -- which builds an outlet effect into the design. v2 keeps one
news article (headline + summary) per row, so every document is about one
story and can be given a single real-world event label.
"""

import os

# Same five outlets as v1, so v1 and v2 can be compared directly.
FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "DW": "https://rss.dw.com/rdf/rss-en-world",
    "NPR": "https://feeds.npr.org/1004/rss.xml",  # NPR World
}

DATA_DIR = "data/v2"
RESULTS_DIR = "results/v2"

RAW_PATH = os.path.join(DATA_DIR, "articles_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "articles_clean.csv")
ANNOTATION_PATH = os.path.join(DATA_DIR, "annotations_A.csv")
ANNOTATION_B_PATH = os.path.join(DATA_DIR, "annotations_B.csv")
COLLECTION_LOG_PATH = os.path.join(DATA_DIR, "collection_log.csv")

RAW_FIELDS = ["article_id", "outlet", "title", "summary", "url",
              "published", "published_date", "collected_at"]

# Removed from topic words so a topic cannot be "named" by the outlet that
# wrote it (the confound this dataset is designed to test).
OUTLET_STOP_WORDS = ["bbc", "guardian", "aljazeera", "al", "jazeera", "dw",
                     "deutsche", "welle", "npr", "says", "say", "said", "news",
                     "live", "latest", "updates", "watch", "video"]

# Text variants built by 08_preprocess_articles.py
TEXT_VARIANTS = {
    "full": "text_full",      # title + full RSS summary (lengths differ by outlet)
    "capped": "text_capped",  # title + first sentence, capped to N words (length-controlled)
}


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

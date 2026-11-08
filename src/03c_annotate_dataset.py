"""
Step 3c: Create and manage manual annotations for the constructed dataset.

This exists specifically for the "annotation" component of the rubric's
"New Data -- Progress" criterion. Your proposal describes checking whether
BERTopic's discovered topics track genuine real-world events -- that check
requires a human to label what the real event(s) were for each day, since
there's no automatic ground truth for "what actually happened in the news
today." This script makes that labeling process concrete and trackable.

Workflow:
  1. Run with --generate to create/update an annotation spreadsheet
     (data/constructed_dataset_annotations.csv) with one row per
     day+source, showing a text preview and an empty label column.
  2. Open that CSV (Excel, Google Sheets, or directly in VS Code) and
     fill in the "manual_event_label" column yourself -- a few words
     naming the real event(s) that pseudo-document covers, e.g.
     "US government shutdown" or "football transfer window".
  3. Run with --status at any time to see how many rows are annotated
     so far -- this count IS your "annotation progress" evidence.

Usage:
    python src/03c_annotate_dataset.py --generate
    python src/03c_annotate_dataset.py --status
"""

import argparse
import os

import pandas as pd

CLEAN_PATH = "data/constructed_news_dataset_clean.csv"
ANNOTATION_PATH = "data/constructed_dataset_annotations.csv"


def generate(preview_chars: int = 200):
    if not os.path.isfile(CLEAN_PATH):
        print(f"No cleaned data found at {CLEAN_PATH}. Run src/03b_preprocess_dataset.py first.")
        return

    clean_df = pd.read_csv(CLEAN_PATH)

    if os.path.isfile(ANNOTATION_PATH):
        existing = pd.read_csv(ANNOTATION_PATH)
        # Remove duplicate date+source rows (keep the first, labelled one)
        n_dup = existing.duplicated(subset=["date", "source"]).sum()
        if n_dup:
            existing = existing.drop_duplicates(subset=["date", "source"], keep="first")
            print(f"Removed {n_dup} duplicate annotation row(s).")
        existing_keys = set(zip(existing["date"], existing["source"]))
    else:
        existing = pd.DataFrame(columns=["date", "source", "preview", "manual_event_label"])
        existing_keys = set()

    new_rows = []
    for _, row in clean_df.iterrows():
        key = (row["date"], row["source"])
        if key in existing_keys:
            continue
        new_rows.append({
            "date": row["date"],
            "source": row["source"],
            "preview": str(row["pseudo_document_clean"])[:preview_chars],
            "manual_event_label": "",
        })
        existing_keys.add(key)  # prevents a second row with the same date+source in one run

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        combined = existing

    combined.to_csv(ANNOTATION_PATH, index=False)
    print(f"Added {len(new_rows)} new row(s) to annotate.")
    print(f"Open {ANNOTATION_PATH} and fill in the 'manual_event_label' column by hand.")
    _print_status(combined)


def status():
    if not os.path.isfile(ANNOTATION_PATH):
        print(f"No annotation file yet. Run with --generate first.")
        return
    df = pd.read_csv(ANNOTATION_PATH)
    _print_status(df)


def _print_status(df: pd.DataFrame):
    total = len(df)
    labeled = df["manual_event_label"].fillna("").astype(str).str.strip().ne("").sum()
    pct = (labeled / total * 100) if total else 0
    print(f"\nAnnotation progress: {labeled}/{total} rows labeled ({pct:.0f}%)")
    if labeled < total:
        print(f"{total - labeled} row(s) still need a manual_event_label.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                         help="Create/update the annotation CSV with any newly collected rows.")
    parser.add_argument("--status", action="store_true",
                         help="Show how many rows have been manually labeled so far.")
    args = parser.parse_args()

    if args.generate:
        generate()
    elif args.status:
        status()
    else:
        print("Specify --generate or --status. See the file's docstring for the workflow.")


if __name__ == "__main__":
    main()
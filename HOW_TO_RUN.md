# How to run (Google Colab)

1. Copy your own data into this folder:
   - `data/v2/articles_raw.csv`
   - `data/v2/collection_log.csv`
   - optional: `results/constructed_dataset_assignments.csv` (the v1 result)
2. Upload the whole `COMP8240_project` folder to Google Drive, directly under `MyDrive/`.
3. In Drive, double-click `run_v2_pipeline.ipynb` and open it with Google Colab.
4. Choose Runtime → Change runtime type → **T4 GPU**.
5. Run the cells from the top. Allow Drive access when Colab asks.
6. **Step 3 stops for labelling.** Label `data/v2/annotations_A.csv` yourself, and have a
   second person label `annotations_B.csv`. Follow `ANNOTATION_GUIDE.md`. Save both as CSV
   with the same names in `data/v2/`.
   Step 5 (the topic models) doesn't need labels, so you can run it while labelling.
7. When the labels are done, run Step 4 and Steps 6–11. Step 11 downloads `results_v2.zip`.
   Send that file back for the report.

Optional: Step 10 refreshes the replication numbers with the fixed code (about 1–2 hours).
Set `REFRESH_REPLICATION = True` in that cell to run it.

Quick test with no GPU and no downloads: set `OFFLINE_TEST = True` in the settings cell.
Test results can't go in the report.

# Annotation guide — event labels for the constructed news dataset (v2)

Goal: give every article in `data/v2/annotations_A.csv` (and `annotations_B.csv`
for the second annotator) an **event ID** so we can test whether BERTopic groups
articles by the real-world story they report, or by the outlet that wrote them.

## What counts as one event

An *event* is one specific real-world happening or ongoing story that a reader
would recognise as "the same news", whichever outlet wrote it.

- Same event: BBC "Israel and Hamas agree first phase of Gaza ceasefire" and
  Al Jazeera "Gaza truce: hostages to be freed on Monday" (same deal, same days).
- Different events: two articles that share only a *theme* — e.g. an Ebola
  outbreak in Congo and a measles outbreak in Canada are both "health", but they
  are different events.
- A follow-up on the same story within the sheet's window (e.g. day 2 of the same
  ceasefire) **is the same event**. A clearly new development with its own news
  hook (e.g. a separate air strike in Lebanon) is a new event.
- If an article mainly covers two events, label the one in its **headline**.

## How to fill the columns

| Column | What to write |
| --- | --- |
| `event_id` | A short code: `E001`, `E002`, … Reuse the same code for every article about that event. Use `SINGLE` only if you prefer — but a unique code per singleton (`E057`) is better. |
| `event_description` | 3–8 words naming the event the first time you create the code, e.g. `Gaza ceasefire phase one`. |
| `notes` | Optional: why a borderline case was decided the way it was. |

`suggested_group` (sheet A only) is an automatic TF-IDF hint that puts likely
same-story articles next to each other. **It is only a hint** — split or merge
groups wherever your judgement differs.

## Rules that keep the labels usable

1. Label every row. Don't skip hard ones; use the headline rule above.
2. Judge from title + summary only, not from which outlet wrote it.
3. Keep a running list of codes and descriptions so you reuse codes consistently.
4. Non-news items that slipped through (quizzes, newsletters, "what we know")
   get the code `NONNEWS`; they are dropped from the event measures.
5. The second annotator (sheet B) should label **without** seeing sheet A or the
   suggested groups, using only this guide.

## After labelling

```
python src/09_annotation_sheet.py --status      # progress evidence
python src/10_annotation_agreement.py           # Cohen's kappa + ARI (needs sheet B)
python src/13_evaluate_event_vs_outlet.py       # event-tracking measures
```

Agreement reading: kappa > 0.8 strong, 0.6–0.8 substantial, 0.4–0.6 moderate.
Discuss disagreements listed by script 10, refine this guide if needed, and say
in the report how many disagreements there were and how they were resolved.

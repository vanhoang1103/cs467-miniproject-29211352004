# Job and Internship Description Matcher

- Student: Nguyễn Văn Hoàng
- Student ID: 29211352004
- Section: CS467A
- Project: MP-A06 – Job and Internship Description Matcher

## Stage 1: Dataset
This repository currently contains a synthetic teaching dataset:
- `data/items.csv`: 40 internship/job postings.
- `data/ratings.csv`: 15 candidate profiles represented by user IDs and 75 interest ratings (5 ratings per candidate).

The job postings and candidate ratings are synthetic examples created for this coursework. They are not claims of live vacancies or real candidate reviews. If a public dataset is used later, its source must be cited here.

## Dataset schema
`items.csv`: item_id, role_title, company, location, job_description, required_skills

`ratings.csv`: user_id, item_id, interest_rating (1–5), note

Note: The assignment's two sparsity instructions conflict mathematically: 12–15 candidates × 4–6 ratings each cannot produce a 40–65% missing matrix when there are 35–45 jobs. This dataset follows the explicit 4–6 ratings-per-candidate instruction; with 15 candidates, 40 jobs, and 75 ratings, 87.5% of the full matrix is missing.\n\n## Stage 1 validation
Run:
```bash
python check_dataset.py
```

Later stages will add NLP, recommendation methods, evaluation, charts, and demo instructions.

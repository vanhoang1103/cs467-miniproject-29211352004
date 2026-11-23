# Job and Internship Description Matcher

An NLP-based job and internship recommendation system that matches candidate resumes and interests with relevant job postings.

## Project Information

- **Student:** Nguyen Van Hoang
- **Student ID:** 29211352004
- **Course:** CS467 - Advanced Artificial Intelligence
- **Section:** CS467A
- **Project:** MP-A06 - Job and Internship Description Matcher

## 1. Problem Statement

Finding suitable internships and entry-level jobs can be difficult for university students.

This project develops a recommendation system that analyzes job descriptions, required technical skills, candidate resumes, and candidate ratings to recommend relevant opportunities.

The system combines Natural Language Processing (NLP), content-based filtering, popularity-based recommendation, and collaborative filtering.

## 2. Dataset Description

The project uses a synthetic dataset created for coursework.

### Job postings

File: `data/items.csv`

- 40 job and internship postings.
- Includes job title, company, location, description, and required skills.

Columns:

`item_id, role_title, company, location, job_description, required_skills`

### Candidate ratings

File: `data/ratings.csv`

- 15 candidate profiles.
- 75 ratings in total.
- Each candidate has 5 ratings.
- Rating scale: 1 to 5.

Columns:

`user_id, item_id, interest_rating, note`

The dataset is synthetic and does not represent real vacancies or actual candidate reviews.

## 3. NLP Pipeline

The NLP pipeline is implemented in `src/nlp_pipeline.py`.

Main processing steps:

1. Convert text to lowercase.
2. Remove punctuation and unnecessary characters.
3. Tokenize text.
4. Remove English and custom stop words.
5. Extract technical skills.
6. Convert job descriptions into TF-IDF vectors.
7. Calculate cosine similarity between resumes and job postings.

### TF-IDF

The project uses unigram and bigram features:

`ngram_range=(1, 2)`

TF-IDF represents textual information numerically so that job descriptions and resumes can be compared.

### Technical skill extraction

The system recognizes technical skills such as:

- Python
- Java
- JavaScript
- SQL
- MySQL
- PostgreSQL
- MongoDB
- React
- Machine Learning
- Deep Learning
- NLP
- Git
- Docker
- AWS
- Azure

## 4. Recommendation Algorithms

Implemented in `src/recommender.py`.

### 4.1 Popularity Baseline

Ranks jobs using a weighted average rating.

The score considers both the average rating of a job and the number of ratings it receives.

### 4.2 Content-Based Filtering

Builds a candidate profile from previously highly rated jobs.

The system compares the candidate profile with job TF-IDF vectors using cosine similarity.

### 4.3 Resume-to-Job Matching

Accepts free-text resume information and ranks jobs according to textual similarity and matched technical skills.

### 4.4 Item-Based Collaborative Filtering

Calculates similarities between jobs based on candidate rating patterns.

### 4.5 Hybrid Recommendation

Combines content-based information and collaborative filtering predictions to produce a ranked recommendation list.

## 5. Evaluation

The evaluation script is `src/evaluate.py`.

The evaluation uses a holdout strategy:

- Training ratings: 60
- Testing ratings: 15

One positive rating is held out for each candidate when available.

### Precision@5 results

| Method        | Precision@5 |
| ------------- | ----------: |
| Popularity    |      0.0000 |
| Content-Based |      0.1733 |
| Hybrid        |      0.0933 |

Content-Based achieved the highest Precision@5 in the current experiment.

The results are based on a small synthetic dataset and should not be interpreted as production-level performance.

### Evaluation charts

- `results/precision_at_5_comparison.png`
- `results/rating_distribution.png`

## 6. Project Structure

```text
cs467-miniproject-29211352004/
│
├── data/
│   ├── items.csv
│   └── ratings.csv
│
├── src/
│   ├── main.py
│   ├── nlp_pipeline.py
│   ├── recommender.py
│   └── evaluate.py
│
├── results/
│   ├── evaluation_metrics.csv
│   ├── precision_at_5_comparison.png
│   ├── rating_distribution.png
│   └── sample_outputs.txt
│
├── notebooks/
├── check_dataset.py
├── test_nlp_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Installation

Python 3.11 or a compatible Python version is recommended.

Clone the repository:

```bash
git clone https://github.com/vanhoang1103/cs467-miniproject-29211352004.git
cd cs467-miniproject-29211352004
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 8. How to Run

### Validate the dataset

```bash
python check_dataset.py
```

### Test the NLP pipeline

```bash
python test_nlp_pipeline.py
```

### Run the recommendation engine

```bash
python src/recommender.py
```

### Run evaluation

```bash
python src/evaluate.py
```

### Run the interactive demo

```bash
python src/main.py
```

The demo provides four options:

1. Resume-to-job matching.
2. Returning candidate recommendation.
3. Cold-start new job test.
4. Run all sample tests and save outputs.

## 9. Test Scenarios

### Scenario 1: Resume matching

A candidate enters resume information containing technical skills such as Python, SQL, Machine Learning, and Git.

The system returns the top five matching job postings.

### Scenario 2: Returning candidate

An existing candidate enters their candidate ID.

The system uses their previous ratings to build a preference profile and recommends jobs they have not rated.

### Scenario 3: Cold-start job posting

A new job posting is temporarily added to the in-memory dataset with zero ratings.

The NLP model is refitted to include the new posting, and content-based matching is used to test whether it can be retrieved.

The original CSV files are not modified by this demonstration.

## 10. Limitations

- The dataset is small and synthetic.
- Recommendation quality may change with larger and more diverse datasets.
- Popularity and collaborative filtering can be affected by sparse ratings.
- The current evaluation focuses on Precision@5.
- More real-world candidate feedback is needed for reliable performance evaluation.

## 11. Conclusion

This project demonstrates how NLP and recommendation algorithms can support job and internship discovery.

It combines TF-IDF, technical skill extraction, cosine similarity, candidate profiles, popularity-based ranking, and collaborative filtering.

The system also demonstrates resume matching, returning-candidate recommendations, and a content-based cold-start test for new job postings.

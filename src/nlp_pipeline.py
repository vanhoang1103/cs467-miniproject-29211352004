
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Làm sạch văn bản
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# 2. Tách từ
def tokenize(text):
    return clean_text(text).split()


# 3. Loại bỏ stop words

CUSTOM_STOP_WORDS = {
    "experience",
    "experienced",
    "work",
    "working",
    "team",
    "teams",
    "candidate",
    "candidates",
    "job",
    "role",
    "company",
    "including",
    "ability",
    "strong",
    "good",
    "excellent",
    "knowledge",
    "skills",
    "skill",
    "required",
    "requirements",
    "responsibilities",
    "using",
    "used",
    "will",
    "must",
    "may",
    "preferred"
}

def remove_stopwords(tokens):
    return [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS
        and word not in CUSTOM_STOP_WORDS
        and len(word) > 1
    ]


# 4. Tiền xử lý hoàn chỉnh
def preprocess_text(text):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    return " ".join(tokens)


# 5. Nhận diện kỹ năng kỹ thuật
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript",
    "c#", "c++", ".net", "sql", "mysql",
    "postgresql", "mongodb", "html", "css",
    "react", "angular", "vue", "node.js",
    "django", "flask", "fastapi", "asp.net",
    "machine learning", "deep learning",
    "natural language processing",
    "computer vision", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy",
    "git", "github", "docker", "kubernetes",
    "linux", "aws", "azure", "rest api",
    "graphql", "selenium", "flutter",
    "php", "laravel", "wordpress"
]


def extract_technical_skills(text):
    text = str(text).lower()
    found = []

    for skill in TECHNICAL_SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found.append(skill)

    return found


# 6. Xây dựng NLP Pipeline
class NLPProcessor:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            stop_words=None,
            ngram_range=(1, 2)
        )

        self.item_vectors = None
        self.items = None

    # Huấn luyện TF-IDF bằng dữ liệu công việc
    def fit(self, items_df):

        self.items = items_df.copy()

        corpus = (
            self.items["role_title"].fillna("").astype(str)
            + " "
            + self.items["job_description"].fillna("").astype(str)
            + " "
            + self.items["required_skills"].fillna("").astype(str)
        )

        self.item_vectors = self.vectorizer.fit_transform(corpus)

        return self

    # 7. Vector hóa truy vấn mới
    def transform_query(self, query):

        if self.item_vectors is None:
            raise ValueError("Please fit the NLP model first.")

        return self.vectorizer.transform([query])

    # 8. Trích xuất từ khóa quan trọng
    def top_keywords(self, text, top_n=10):

        vector = self.transform_query(text)

        scores = vector.toarray()[0]

        feature_names = self.vectorizer.get_feature_names_out()

        ranked_indices = scores.argsort()[::-1]

        keywords = []

        for index in ranked_indices:
            if scores[index] > 0:
                keywords.append(feature_names[index])

            if len(keywords) >= top_n:
                break

        return keywords

    # 9. So sánh CV với công việc
    def compare_resume_to_job(self, resume, job):

        job_text = (
            str(job["role_title"]) + " "
            + str(job["job_description"]) + " "
            + str(job["required_skills"])
        )

        resume_vector = self.transform_query(resume)
        job_vector = self.transform_query(job_text)

        similarity = cosine_similarity(
            resume_vector,
            job_vector
        )[0][0]

        resume_skills = set(extract_technical_skills(resume))
        job_skills = set(extract_technical_skills(job_text))

        matched_skills = sorted(resume_skills & job_skills)
        missing_skills = sorted(job_skills - resume_skills)

        return {
            "similarity": round(float(similarity), 4),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }

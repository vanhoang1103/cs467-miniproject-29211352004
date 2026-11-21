
import pandas as pd

from src.nlp_pipeline import NLPProcessor
from src.nlp_pipeline import (
    clean_text,
    tokenize,
    remove_stopwords,
    extract_technical_skills
)

# Đọc dữ liệu Giai đoạn 1
items = pd.read_csv("data/items.csv")

# Kiểm tra tiền xử lý
print("1. CLEAN TEXT")
print(clean_text("PYTHON, SQL!!! Machine Learning"))

print("\n2. TOKENIZATION")
print(tokenize("Python, SQL and Machine Learning"))

print("\n3. STOP WORD REMOVAL")
print(remove_stopwords(tokenize(
    "I have experience with Python and SQL"
)))

print("\n4. TECHNICAL SKILLS")
print(extract_technical_skills(
    "Python, SQL, React, Git and Machine Learning"
))

# Huấn luyện NLP
nlp = NLPProcessor()
nlp.fit(items)

print("\n5. TF-IDF")
print("Number of jobs:", len(items))
print("Matrix shape:", nlp.item_vectors.shape)
print("Vocabulary size:",
      len(nlp.vectorizer.get_feature_names_out()))

# Thử CV
resume = """
Computer science student with Python, SQL,
machine learning and Git experience.
"""

print("\n6. RESUME KEYWORDS")
print(nlp.top_keywords(resume))

# Kiểm tra vector hóa
query_vector = nlp.transform_query(resume)

print("\n7. QUERY VECTOR")
print(query_vector.shape)

# So sánh với công việc đầu tiên
result = nlp.compare_resume_to_job(
    resume,
    items.iloc[0]
)

print("\n8. RESUME JOB MATCH")
print(result)

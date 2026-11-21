
"""
Job and Internship Description Matcher
Stage 3: Recommendation Engine

Methods:
1. Popularity Baseline
2. NLP Content-Based Filtering
3. Resume-to-Job + Item-Based CF Hybrid
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import NLPProcessor, extract_technical_skills


class JobRecommender:
    def __init__(self, items_path="data/items.csv",
                 ratings_path="data/ratings.csv"):

        self.items = pd.read_csv(items_path)
        self.ratings = pd.read_csv(ratings_path)

        self.items["item_id"] = self.items["item_id"].astype(str)
        self.ratings["item_id"] = self.ratings["item_id"].astype(str)
        self.ratings["user_id"] = self.ratings["user_id"].astype(str)

        self.items = self.items.reset_index(drop=True)

        # Build text for each job
        # NLP model
        self.nlp = NLPProcessor()
        self.nlp.fit(self.items)

        self.item_vectors = self.nlp.item_vectors

        # TF-IDF vectors for all jobs
        self.item_vectors = self.nlp.item_vectors

        # User-item rating matrix
        self.rating_matrix = self.ratings.pivot_table(
            index="user_id",
            columns="item_id",
            values="interest_rating",
            aggfunc="mean"
        )

        # Item-item collaborative similarity
        self.item_cf_similarity = self._build_item_cf_matrix()

        # Average rating and number of ratings
        self.popularity = self.ratings.groupby("item_id")[
            "interest_rating"
        ].agg(["mean", "count"])

        self.global_mean = self.ratings["interest_rating"].mean()

    # --------------------------------------------------
    # Helper: get item information
    # --------------------------------------------------

    def _get_item(self, item_id):
        result = self.items[
            self.items["item_id"] == str(item_id)
        ]

        if result.empty:
            return None

        return result.iloc[0]

    def _item_text(self, item):
        return (
            str(item["role_title"]) + " " +
            str(item["job_description"]) + " " +
            str(item["required_skills"])
        )

    def _get_skills(self, text):
        return set(extract_technical_skills(str(text)))

    def _get_matched_skills(self, query, job_text):
        query_skills = self._get_skills(query)
        job_skills = self._get_skills(job_text)

        return sorted(query_skills.intersection(job_skills))

    # --------------------------------------------------
    # 1. POPULARITY BASELINE
    # --------------------------------------------------

    def recommend_popular(self, user_id=None, top_n=5,
                          min_ratings=2):

        scores = self.popularity.copy()

        # Bayesian weighted rating
        scores["score"] = (
            scores["count"] / (scores["count"] + min_ratings)
            * scores["mean"]
            +
            min_ratings / (scores["count"] + min_ratings)
            * self.global_mean
        )

        # Remove items already rated by this user
        if user_id is not None:
            user_id = str(user_id)

            rated_items = self.ratings.loc[
                self.ratings["user_id"] == user_id,
                "item_id"
            ].tolist()

            scores = scores.drop(
                index=rated_items,
                errors="ignore"
            )

        # Only keep jobs with enough ratings
        scores = scores[scores["count"] >= min_ratings]

        scores = scores.sort_values(
            "score", ascending=False
        ).head(top_n)

        return self._format_results(
            scores.index.tolist(),
            scores["score"].tolist(),
            method="Popularity"
        )

    # --------------------------------------------------
    # 2. NLP CONTENT-BASED FILTERING
    # --------------------------------------------------

    def recommend_by_text(self, query, top_n=5,
                           exclude_items=None):

        query_vector = self.nlp.transform_query(query)

        similarities = cosine_similarity(
            query_vector,
            self.item_vectors
        ).flatten()

        exclude_items = set(
            str(x) for x in (exclude_items or [])
        )

        results = []

        for index, similarity in enumerate(similarities):
            item = self.items.iloc[index]
            item_id = str(item["item_id"])

            if item_id in exclude_items:
                continue

            job_text = self._item_text(item)

            matched_skills = self._get_matched_skills(
                query, job_text
            )

            results.append({
                "item_id": item_id,
                "score": float(similarity),
                "matched_skills": matched_skills
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return self._format_results(
            [x["item_id"] for x in results[:top_n]],
            [x["score"] for x in results[:top_n]],
            method="Content-Based",
            matched_skills=[
                x["matched_skills"] for x in results[:top_n]
            ]
        )

    # --------------------------------------------------
    # Item-to-item recommendation
    # --------------------------------------------------

    def recommend_similar_items(self, item_id, top_n=5):

        item_id = str(item_id)

        if item_id not in self.items["item_id"].values:
            raise ValueError(f"Unknown item_id: {item_id}")

        item_index = self.items.index[
            self.items["item_id"] == item_id
        ][0]

        similarities = cosine_similarity(
            self.item_vectors[item_index],
            self.item_vectors
        ).flatten()

        results = []

        for index, score in enumerate(similarities):
            other_id = str(self.items.iloc[index]["item_id"])

            if other_id == item_id:
                continue

            results.append((other_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

        return self._format_results(
            [x[0] for x in results[:top_n]],
            [x[1] for x in results[:top_n]],
            method="Item-to-Item"
        )

    # --------------------------------------------------
    # User profile recommendation
    # --------------------------------------------------

    def recommend_for_user_profile(self, user_id, top_n=5):

        user_id = str(user_id)

        user_ratings = self.ratings[
            self.ratings["user_id"] == user_id
        ]

        if user_ratings.empty:
            raise ValueError(f"No ratings found for user {user_id}")

        # Use jobs rated 4 or 5 as positive preferences
        positive = user_ratings[
            user_ratings["interest_rating"] >= 4
        ]

        if positive.empty:
            positive = user_ratings

        rated_items = user_ratings["item_id"].tolist()

        vectors = []
        weights = []

        for _, rating in positive.iterrows():
            item_id = str(rating["item_id"])

            index = self.items.index[
                self.items["item_id"] == item_id
            ]

            if len(index) == 0:
                continue

            vectors.append(self.item_vectors[index[0]].toarray()[0])
            weights.append(float(rating["interest_rating"]))

        if not vectors:
            raise ValueError("Cannot build user profile")

        profile = np.average(
            np.array(vectors),
            axis=0,
            weights=np.array(weights)
        )

        similarities = cosine_similarity(
            profile.reshape(1, -1),
            self.item_vectors
        ).flatten()

        return self._rank_scores(
            similarities,
            top_n=top_n,
            exclude_items=rated_items,
            method="User-Profile"
        )

    # --------------------------------------------------
    # 3. ITEM-BASED COLLABORATIVE FILTERING
    # --------------------------------------------------

    def _build_item_cf_matrix(self):

        matrix = self.rating_matrix.fillna(0)

        if matrix.empty:
            return pd.DataFrame()

        similarity = cosine_similarity(matrix.T)

        return pd.DataFrame(
            similarity,
            index=matrix.columns,
            columns=matrix.columns
        )

    def recommend_item_cf(self, user_id, top_n=5):

        user_id = str(user_id)

        if user_id not in self.rating_matrix.index:
            raise ValueError(f"Unknown user_id: {user_id}")

        user_ratings = self.rating_matrix.loc[user_id].dropna()

        rated_items = user_ratings.index.tolist()

        predictions = {}

        for candidate in self.item_cf_similarity.columns:
            if candidate in rated_items:
                continue

            numerator = 0.0
            denominator = 0.0

            for rated_item, rating in user_ratings.items():
                similarity = self.item_cf_similarity.loc[
                    candidate, rated_item
                ]

                numerator += similarity * float(rating)
                denominator += abs(similarity)

            if denominator > 0:
                predictions[candidate] = numerator / denominator

        ranked = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return self._format_results(
            [x[0] for x in ranked],
            [x[1] for x in ranked],
            method="Item-Based CF"
        )

    # --------------------------------------------------
    # HYBRID: Resume + Content + Collaborative Filtering
    # --------------------------------------------------

    def recommend_hybrid(self, user_id, resume_text=None,
                         top_n=5):

        user_id = str(user_id)

        # Get resume/content scores
        if resume_text:
            query_vector = self.nlp.transform_query(resume_text)

            content_scores = cosine_similarity(
                query_vector,
                self.item_vectors
            ).flatten()
        else:
            content_scores = np.zeros(len(self.items))

        # Get CF predictions
        cf_results = self.recommend_item_cf(
            user_id=user_id,
            top_n=len(self.items)
        )

        cf_scores = {
            x["item_id"]: x["score"]
            for x in cf_results
        }

        # Normalize scores to [0, 1]
        def normalize(values):
            values = np.asarray(values, dtype=float)

            if len(values) == 0 or values.max() == values.min():
                return np.zeros_like(values)

            return (
                values - values.min()
            ) / (
                values.max() - values.min()
            )

        content_norm = normalize(content_scores)

        cf_values = np.array([
            cf_scores.get(str(item_id), 0.0)
            for item_id in self.items["item_id"]
        ])

        cf_norm = normalize(cf_values)

        # Weighted hybrid score
        # 70% resume/content + 30% collaborative filtering
        hybrid_scores = (
            0.70 * content_norm +
            0.30 * cf_norm
        )

        rated_items = self.ratings.loc[
            self.ratings["user_id"] == user_id,
            "item_id"
        ].tolist()

        return self._rank_scores(
            hybrid_scores,
            top_n=top_n,
            exclude_items=rated_items,
            method="Hybrid"
        )

    # --------------------------------------------------
    # Resume-to-job matching
    # --------------------------------------------------

    def match_resume(self, resume_text, top_n=5):

        recommendations = self.recommend_by_text(
            query=resume_text,
            top_n=top_n
        )

        return recommendations

    # --------------------------------------------------
    # Internal ranking and formatting
    # --------------------------------------------------

    def _rank_scores(self, scores, top_n=5,
                     exclude_items=None, method="Unknown"):

        exclude_items = set(
            str(x) for x in (exclude_items or [])
        )

        results = []

        for index, score in enumerate(scores):
            item_id = str(self.items.iloc[index]["item_id"])

            if item_id in exclude_items:
                continue

            results.append((item_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

        return self._format_results(
            [x[0] for x in results[:top_n]],
            [x[1] for x in results[:top_n]],
            method=method
        )

    def _format_results(self, item_ids, scores,
                        method="Unknown", matched_skills=None):

        results = []

        for index, (item_id, score) in enumerate(
            zip(item_ids, scores)
        ):
            item = self._get_item(item_id)

            if item is None:
                continue

            job_text = self._item_text(item)

            skills = (
                matched_skills[index]
                if matched_skills is not None
                else []
            )

            results.append({
                "rank": index + 1,
                "item_id": str(item["item_id"]),
                "role_title": item["role_title"],
                "company": item["company"],
                "location": item["location"],
                "score": round(float(score), 4),
                "matched_skills": skills,
                "method": method
            })

        return results


# ------------------------------------------------------
# Quick manual test
# ------------------------------------------------------

if __name__ == "__main__":

    recommender = JobRecommender()

    print("\n=== POPULARITY BASELINE ===")
    popular = recommender.recommend_popular(
        user_id="U001",
        top_n=5
    )

    for job in popular:
        print(job)

    print("\n=== RESUME MATCH ===")

    resume = """
    I am a Computer Science student.
    I have knowledge of Python, SQL, machine learning and Git.
    I am looking for a software development internship.
    """

    resume_results = recommender.match_resume(
        resume_text=resume,
        top_n=5
    )

    for job in resume_results:
        print(job)

    print("\n=== RETURNING CANDIDATE ===")

    user_results = recommender.recommend_for_user_profile(
        user_id="U001",
        top_n=5
    )

    for job in user_results:
        print(job)

    print("\n=== HYBRID RECOMMENDATION ===")

    hybrid_results = recommender.recommend_hybrid(
        user_id="U001",
        resume_text=resume,
        top_n=5
    )

    for job in hybrid_results:
        print(job)

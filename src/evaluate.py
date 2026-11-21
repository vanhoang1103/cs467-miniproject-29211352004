
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Xác định thư mục gốc của project
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender import JobRecommender


DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ITEMS_PATH = DATA_DIR / "items.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def precision_at_k(recommended, relevant, k=5):
    """
    Tính Precision@K.
    recommended: danh sách item_id được hệ thống gợi ý.
    relevant: danh sách item_id thực sự phù hợp.
    """
    if k <= 0:
        return 0.0

    top_k = recommended[:k]

    if not top_k:
        return 0.0

    hits = len(set(top_k) & set(relevant))
    return hits / k


def create_holdout_split(ratings):
    """
    Giữ lại một đánh giá 4-5 sao của mỗi ứng viên
    làm dữ liệu kiểm tra.
    """
    train_parts = []
    test_parts = []

    for user_id, group in ratings.groupby("user_id"):
        positive_ratings = group[
            group["interest_rating"] >= 4
        ]

        if positive_ratings.empty:
            train_parts.append(group)
            continue

        # Chọn một đánh giá tích cực để ẩn
        hidden = positive_ratings.sample(
            n=1,
            random_state=42
        )

        train = group.drop(index=hidden.index)

        train_parts.append(train)
        test_parts.append(hidden)

    train_df = pd.concat(train_parts).reset_index(drop=True)

    if test_parts:
        test_df = pd.concat(test_parts).reset_index(drop=True)
    else:
        test_df = pd.DataFrame(columns=ratings.columns)

    return train_df, test_df


def get_recommendations(recommender, method, user_id, top_n=5):
    """
    Gọi phương pháp gợi ý tương ứng.
    """
    method_candidates = {
        "Popularity": ["recommend_popular"],
        "Content-Based": [
            "recommend_user_profile",
            "recommend_by_user_profile"
        ],
        "Hybrid": ["recommend_hybrid"]
    }

    for method_name in method_candidates[method]:
        function = getattr(recommender, method_name, None)

        if callable(function):
            results = function(user_id=user_id, top_n=top_n)

            if results is None:
                return []

            return [
                str(item["item_id"])
                for item in results
            ]

    raise AttributeError(
        f"Không tìm thấy hàm gợi ý cho phương pháp {method}. "
        "Hãy kiểm tra tên hàm trong recommender.py."
    )


def evaluate_models(recommender, test_df):
    """
    Đánh giá ba phương pháp bằng Precision@5.
    """
    methods = [
        "Popularity",
        "Content-Based",
        "Hybrid"
    ]

    scores = {method: [] for method in methods}

    for user_id, group in test_df.groupby("user_id"):
        # Chỉ xem các công việc được đánh giá 4-5 sao
        relevant_items = group.loc[
            group["interest_rating"] >= 4,
            "item_id"
        ].astype(str).tolist()

        if not relevant_items:
            continue

        for method in methods:
            recommended = get_recommendations(
                recommender,
                method,
                user_id,
                top_n=5
            )

            score = precision_at_k(
                recommended,
                relevant_items,
                k=5
            )

            scores[method].append(score)

    metrics = {}

    for method in methods:
        if scores[method]:
            metrics[method] = sum(scores[method]) / len(scores[method])
        else:
            metrics[method] = 0.0

    return metrics


def plot_rating_distribution(ratings):
    """
    Biểu đồ phân phối điểm đánh giá.
    """
    counts = (
        ratings["interest_rating"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Distribution of Candidate Ratings")
    plt.xlabel("Interest Rating")
    plt.ylabel("Number of Ratings")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = RESULTS_DIR / "rating_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_evaluation_comparison(metrics):
    """
    Biểu đồ so sánh Precision@5.
    """
    methods = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    plt.bar(methods, values)

    plt.title("Recommendation Methods - Precision@5")
    plt.xlabel("Recommendation Method")
    plt.ylabel("Precision@5")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path = RESULTS_DIR / "precision_at_5_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu ban đầu
    items = pd.read_csv(ITEMS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)

    print("=" * 55)
    print("JOB AND INTERNSHIP RECOMMENDER - EVALUATION")
    print("=" * 55)

    print(f"Total jobs: {len(items)}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Total candidates: {ratings['user_id'].nunique()}")

    # Chia tập train/test
    train_df, test_df = create_holdout_split(ratings)

    print("\nDATA SPLIT")
    print(f"Training ratings: {len(train_df)}")
    print(f"Testing ratings: {len(test_df)}")

    if test_df.empty:
        raise ValueError(
            "Không có đánh giá tích cực 4-5 sao để tạo tập kiểm tra."
        )

    # Tạm thay ratings.csv bằng tập train.
    # Luôn khôi phục dữ liệu gốc sau khi đánh giá.
    original_data = RATINGS_PATH.read_bytes()

    try:
        train_df.to_csv(RATINGS_PATH, index=False)

        # Huấn luyện recommender chỉ bằng dữ liệu train
        recommender = JobRecommender()

        metrics = evaluate_models(
            recommender,
            test_df
        )

    finally:
        RATINGS_PATH.write_bytes(original_data)

    # In kết quả
    print("\nEVALUATION RESULTS")
    print("-" * 55)

    for method, score in metrics.items():
        print(f"{method:20s}: {score:.4f}")

    # Lưu kết quả đánh giá
    metrics_df = pd.DataFrame(
        [
            {
                "method": method,
                "precision_at_5": score
            }
            for method, score in metrics.items()
        ]
    )

    metrics_df.to_csv(
        RESULTS_DIR / "evaluation_metrics.csv",
        index=False
    )

    # Tạo hai biểu đồ
    plot_rating_distribution(ratings)
    plot_evaluation_comparison(metrics)

    print("\nEVALUATION COMPLETED")
    print("Original ratings.csv has been restored.")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

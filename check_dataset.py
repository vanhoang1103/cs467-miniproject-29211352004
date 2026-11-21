import pandas as pd

items = pd.read_csv("data/items.csv")
ratings = pd.read_csv("data/ratings.csv")

required_items = ["item_id", "role_title", "company", "location", "job_description", "required_skills"]
required_ratings = ["user_id", "item_id", "interest_rating", "note"]

assert list(items.columns) == required_items, "Sai schema items.csv"
assert list(ratings.columns) == required_ratings, "Sai schema ratings.csv"
assert 35 <= len(items) <= 45, "Số lượng items phải từ 35 đến 45"
assert 12 <= ratings["user_id"].nunique() <= 15, "Số ứng viên phải từ 12 đến 15"
assert ratings["interest_rating"].between(1, 5).all(), "Rating phải nằm trong 1..5"
assert not items.isnull().any().any(), "items.csv có ô trống"
assert not ratings.isnull().any().any(), "ratings.csv có ô trống"
assert not ratings.duplicated(["user_id", "item_id"]).any(), "Một ứng viên đánh giá trùng một công việc"

per_user = ratings.groupby("user_id")["item_id"].nunique()
assert per_user.between(4, 6).all(), "Mỗi ứng viên phải đánh giá 4-6 công việc"

matrix = ratings.pivot(index="user_id", columns="item_id", values="interest_rating").reindex(columns=items["item_id"])
total_cells = matrix.shape[0] * matrix.shape[1]
missing_cells = int(matrix.isna().sum().sum())
missing_pct = missing_cells / total_cells * 100

print("=== DATASET VALIDATION: PASSED ===")
print(f"Job postings: {len(items)}")
print(f"Candidates: {ratings['user_id'].nunique()}")
print(f"Ratings: {len(ratings)}")
print(f"Ratings per candidate: min={per_user.min()}, max={per_user.max()}")
print(f"User-item matrix: {matrix.shape[0]} x {matrix.shape[1]} = {total_cells} cells")
print(f"Missing cells: {missing_cells} ({missing_pct:.2f}%)")
print(f"Observed ratings: {len(ratings)} ({100-missing_pct:.2f}%)")
print(f"Items with >=2 ratings: {(ratings.groupby('item_id').size() >= 2).sum()}")
print(f"Items with zero ratings: {(ratings.groupby('item_id').size().reindex(items['item_id'], fill_value=0) == 0).sum()}")
print("\nCandidate rating counts:")
print(per_user.to_string())

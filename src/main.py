
"""
Job and Internship Description Matcher
Stage 5: Interactive Demo

Features:
1. Resume-to-job matching
2. Returning candidate recommendation
3. Cold-start recommendation for a new job posting
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from recommender import JobRecommender
from nlp_pipeline import NLPProcessor


ITEMS_PATH = ROOT / "data" / "items.csv"
RATINGS_PATH = ROOT / "data" / "ratings.csv"
RESULTS_DIR = ROOT / "results"


def print_results(results):
    """Display recommendation results."""
    if not results:
        print("No recommendations found.")
        return

    for item in results:
        print("-" * 60)
        print(f"Rank: {item.get('rank', '-')}")
        print(f"Job ID: {item.get('item_id', '-')}")
        print(f"Role: {item.get('role_title', '-')}")
        print(f"Company: {item.get('company', '-')}")
        print(f"Location: {item.get('location', '-')}")
        print(f"Score: {item.get('score', 0):.4f}")
        print(f"Method: {item.get('method', '-')}")
        print(
            "Matched skills: "
            + ", ".join(item.get("matched_skills", []))
        )


def resume_demo(recommender, resume_text=None):
    """Recommend jobs based on resume text."""
    print("\n=== RESUME-TO-JOB MATCHING ===")

    if not resume_text:
        resume_text = input(
            "Enter your resume skills and experience: "
        ).strip()

    if not resume_text:
        print("Resume text cannot be empty.")
        return []

    results = recommender.recommend_by_text(
        resume_text,
        top_n=5
    )

    print_results(results)
    return results


def returning_candidate_demo(recommender, user_id=None):
    """Recommend jobs for an existing candidate."""
    print("\n=== RETURNING CANDIDATE ===")

    available_users = sorted(
        recommender.ratings["user_id"].astype(str).unique()
    )

    print("Available candidate IDs:")
    print(", ".join(available_users))

    if user_id is None:
        user_id = input("Enter candidate ID: ").strip()

    if user_id not in available_users:
        print(f"Candidate {user_id} was not found.")
        return []

    results = recommender.recommend_for_user_profile(
        user_id=user_id,
        top_n=5
    )

    print_results(results)
    return results


def cold_start_demo(recommender):
    """
    Test a new job posting with zero ratings.

    The new job is added only in memory.
    Original CSV files are not modified.
    """
    print("\n=== COLD-START: NEW JOB POSTING ===")

    new_job = {
        "item_id": "DEMO_NEW_001",
        "role_title": "Junior Python AI Developer",
        "company": "Demo Technology",
        "location": "Da Nang",
        "job_description": (
            "Looking for a junior developer with Python, "
            "SQL, machine learning, data analysis and Git "
            "knowledge. Candidates will work on AI projects."
        ),
        "required_skills": (
            "Python, SQL, Machine Learning, Data Analysis, Git"
        )
    }

    original_items = recommender.items.copy()
    original_nlp = recommender.nlp
    original_vectors = recommender.item_vectors

    try:
        import pandas as pd

        combined_items = pd.concat(
            [
                original_items,
                pd.DataFrame([new_job])
            ],
            ignore_index=True
        )

        # Refit TF-IDF to include the new job posting.
        new_nlp = NLPProcessor()
        new_nlp.fit(combined_items)

        recommender.items = combined_items
        recommender.nlp = new_nlp
        recommender.item_vectors = new_nlp.item_vectors

        print("New job added temporarily:")
        print(f"Job ID: {new_job['item_id']}")
        print(f"Role: {new_job['role_title']}")
        print(f"Company: {new_job['company']}")
        print("Existing ratings for this job: 0")

        results = recommender.recommend_by_text(
            "Python SQL machine learning Git AI developer",
            top_n=5
        )

        print("\nRecommendations including the new job:")
        print_results(results)

        return results

    finally:
        # Restore the original in-memory recommender.
        recommender.items = original_items
        recommender.nlp = original_nlp
        recommender.item_vectors = original_vectors


def run_sample_tests(recommender):
    """Run all three demo scenarios and save their outputs."""
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()

    with redirect_stdout(output):
        print("=" * 65)
        print("JOB AND INTERNSHIP DESCRIPTION MATCHER")
        print("STAGE 5 - SAMPLE DEMONSTRATION")
        print("=" * 65)

        resume_demo(
            recommender,
            resume_text=(
                "Computer Science student with Python, SQL, "
                "machine learning, Git and data analysis skills."
            )
        )

        sample_user = sorted(
            recommender.ratings["user_id"].astype(str).unique()
        )[0]

        returning_candidate_demo(
            recommender,
            user_id=sample_user
        )

        cold_start_demo(recommender)

    content = output.getvalue()
    print(content)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "sample_outputs.txt"
    output_path.write_text(content, encoding="utf-8")

    print(f"\nSample outputs saved to: {output_path}")


def main():
    print("=" * 65)
    print("JOB AND INTERNSHIP DESCRIPTION MATCHER")
    print("=" * 65)

    recommender = JobRecommender(
        items_path=str(ITEMS_PATH),
        ratings_path=str(RATINGS_PATH)
    )

    print(f"Loaded jobs: {len(recommender.items)}")
    print(f"Loaded ratings: {len(recommender.ratings)}")
    print(
        "Loaded candidates: "
        f"{recommender.ratings['user_id'].nunique()}"
    )

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Resume-to-job matching")
        print("2. Returning candidate recommendation")
        print("3. Cold-start new job test")
        print("4. Run all sample tests and save outputs")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        try:
            if choice == "1":
                resume_demo(recommender)

            elif choice == "2":
                returning_candidate_demo(recommender)

            elif choice == "3":
                cold_start_demo(recommender)

            elif choice == "4":
                run_sample_tests(recommender)

            elif choice == "0":
                print("Exiting demo. Goodbye!")
                break

            else:
                print("Invalid option. Please choose 0-4.")

        except Exception as error:
            print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()

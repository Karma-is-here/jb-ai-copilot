import json
from pathlib import Path
from collections import defaultdict
import statistics


FILE = Path("data/benchmarks/generation_evaluation.json")


def main():
    data = json.loads(
        FILE.read_text(encoding="utf-8")
    )

    scores = []
    top_scores = []
    second_scores = []
    third_scores = []
    fourth_scores = []
    fifth_scores = []

    questions_with_scores = 0

    for result in data["results"]:
        chunks = result.get("retrieved_chunks", [])

        if not chunks:
            continue

        questions_with_scores += 1

        reranker_scores = [
            c.get("reranker_score")
            for c in chunks[:5]
            if c.get("reranker_score") is not None
        ]

        scores.extend(reranker_scores)

        if len(reranker_scores) >= 1:
            top_scores.append(reranker_scores[0])

        if len(reranker_scores) >= 2:
            second_scores.append(reranker_scores[1])

        if len(reranker_scores) >= 3:
            third_scores.append(reranker_scores[2])

        if len(reranker_scores) >= 4:
            fourth_scores.append(reranker_scores[3])

        if len(reranker_scores) >= 5:
            fifth_scores.append(reranker_scores[4])

    print("=" * 80)
    print("RERANKER SCORE DISTRIBUTION")
    print("=" * 80)

    print(f"Questions with retrieved chunks : {questions_with_scores}")
    print(f"Total scores analyzed            : {len(scores)}")

    if not scores:
        print("No reranker scores found.")
        return

    print()
    print("ALL TOP-5 RERANKER SCORES")
    print("-" * 80)

    print(f"Min      : {min(scores):.4f}")
    print(f"Max      : {max(scores):.4f}")
    print(f"Mean     : {statistics.mean(scores):.4f}")
    print(f"Median   : {statistics.median(scores):.4f}")

    print()
    print("POSITION-WISE DISTRIBUTION")
    print("-" * 80)

    positions = [
        ("Rank 1", top_scores),
        ("Rank 2", second_scores),
        ("Rank 3", third_scores),
        ("Rank 4", fourth_scores),
        ("Rank 5", fifth_scores),
    ]

    for name, values in positions:
        if not values:
            continue

        print(
            f"{name:<8} "
            f"mean={statistics.mean(values):8.4f}  "
            f"median={statistics.median(values):8.4f}  "
            f"min={min(values):8.4f}  "
            f"max={max(values):8.4f}"
        )

    print()
    print("THRESHOLD COUNTS")
    print("-" * 80)

    for threshold in [
        -10,
        -5,
        -2,
        0,
        2,
        4,
        6,
        8,
    ]:
        count = sum(
            score >= threshold
            for score in scores
        )

        percentage = (
            count / len(scores) * 100
        )

        print(
            f">= {threshold:>3}: "
            f"{count:4}/{len(scores)} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("=" * 80)
    print("IMPORTANT")
    print("=" * 80)
    print(
        "These statistics are diagnostic only. "
        "They should NOT be used to choose an evidence "
        "threshold yet."
    )


if __name__ == "__main__":
    main()
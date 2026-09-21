import json
from pathlib import Path
from collections import defaultdict

from src.retrieval.vector import VectorRetriever
from src.retrieval.lexical import LexicalRetriever
from src.retrieval.hybrid import HybridRetriever


BENCHMARK_DIR = Path("data/benchmarks")
OUTPUT_FILE = BENCHMARK_DIR / "retrieval_evaluation.json"

TOP_K = 20
CANDIDATE_K = 20


def load_benchmarks():
    """
    Load all benchmark *_questions.json files.

    Each benchmark item has:
        question
        answer
        page

    The benchmark file path determines the expected document_id.
    """
    benchmarks = []

    for path in BENCHMARK_DIR.rglob("*_questions.json"):
        relative = path.relative_to(BENCHMARK_DIR)

        # Convert:
        # investment/advisory/advisory-mandates_questions.json
        #
        # into:
        # investment/advisory/advisory-mandates
        document_id = str(relative.with_suffix("")).replace("\\", "/")
        document_id = document_id.removesuffix("_questions")

        with path.open("r", encoding="utf-8") as f:
            questions = json.load(f)

        for index, item in enumerate(questions, start=1):
            benchmarks.append(
                {
                    "benchmark_file": str(relative).replace("\\", "/"),
                    "question_id": f"{document_id}::q{index}",
                    "document_id": document_id,
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                    "expected_page": item["page"],
                }
            )

    return benchmarks


def is_hit(result, expected_document_id, expected_page):
    """
    A retrieval result is relevant when:
      1. document_id matches the benchmark document
      2. expected page falls within the chunk's page range
    """
    if result["document_id"] != expected_document_id:
        return False

    page_start = result.get("page_start")
    page_end = result.get("page_end")

    if page_start is None or page_end is None:
        return False

    return page_start <= expected_page <= page_end


def first_hit_rank(results, expected_document_id, expected_page):
    """
    Return 1-based rank of the first relevant result.
    Return None if no relevant result exists.
    """
    for rank, result in enumerate(results, start=1):
        if is_hit(result, expected_document_id, expected_page):
            return rank

    return None


def recall_at(results, expected_document_id, expected_page, k):
    """
    Binary recall for a single query:
    1 if a relevant chunk appears in top-k, else 0.
    """
    return int(
        any(
            is_hit(result, expected_document_id, expected_page)
            for result in results[:k]
        )
    )


def reciprocal_rank(results, expected_document_id, expected_page):
    """
    Reciprocal rank for a single query.
    """
    rank = first_hit_rank(
        results,
        expected_document_id,
        expected_page,
    )

    if rank is None:
        return 0.0

    return 1.0 / rank


def evaluate_results(results, benchmark):
    return {
        "recall_at_1": recall_at(
            results,
            benchmark["document_id"],
            benchmark["expected_page"],
            1,
        ),
        "recall_at_3": recall_at(
            results,
            benchmark["document_id"],
            benchmark["expected_page"],
            3,
        ),
        "recall_at_5": recall_at(
            results,
            benchmark["document_id"],
            benchmark["expected_page"],
            5,
        ),
        "reciprocal_rank": reciprocal_rank(
            results,
            benchmark["document_id"],
            benchmark["expected_page"],
        ),
        "hit_rank": first_hit_rank(
            results,
            benchmark["document_id"],
            benchmark["expected_page"],
        ),
    }


def aggregate(metrics):
    count = len(metrics)

    if count == 0:
        return {
            "questions": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "MRR@20": 0.0,
        }

    return {
        "questions": count,
        "recall@1": sum(x["recall_at_1"] for x in metrics) / count,
        "recall@3": sum(x["recall_at_3"] for x in metrics) / count,
        "recall@5": sum(x["recall_at_5"] for x in metrics) / count,
        "MRR@20": sum(x["reciprocal_rank"] for x in metrics) / count,
    }


def print_summary(summary):
    print()
    print("=" * 72)
    print("RETRIEVAL EVALUATION")
    print("=" * 72)

    print(
        f"{'Method':<12}"
        f"{'Questions':>12}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'Recall@5':>12}"
        f"{'MRR@20':>12}"
    )

    print("-" * 72)

    for method, metrics in summary.items():
        print(
            f"{method:<12}"
            f"{metrics['questions']:>12}"
            f"{metrics['recall@1']:>12.3f}"
            f"{metrics['recall@3']:>12.3f}"
            f"{metrics['recall@5']:>12.3f}"
            f"{metrics['MRR@20']:>12.3f}"
        )

    print("=" * 72)


def main():
    benchmarks = load_benchmarks()

    if not benchmarks:
        raise RuntimeError(
            "No *_questions.json benchmark files found."
        )

    print(f"Loaded {len(benchmarks)} benchmark questions.")
    print(
        f"Across "
        f"{len(set(b['document_id'] for b in benchmarks))} documents."
    )

    print()
    print("Loading retrievers...")

    vector = VectorRetriever()
    lexical = LexicalRetriever()
    hybrid = HybridRetriever()

    retrievers = {
        "vector": vector,
        "lexical": lexical,
        "hybrid": hybrid,
    }

    per_question = []
    grouped_metrics = defaultdict(lambda: defaultdict(list))

    total = len(benchmarks)

    for i, benchmark in enumerate(benchmarks, start=1):

        print(
            f"[{i}/{total}] "
            f"{benchmark['question']}"
        )

        for method, retriever in retrievers.items():

            if method == "hybrid":
                results = retriever.search(
                    benchmark["question"],
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                )
            else:
                results = retriever.search(
                    benchmark["question"],
                    top_k=TOP_K,
                )

            metrics = evaluate_results(
                results,
                benchmark,
            )

            grouped_metrics[method]["metrics"].append(metrics)

            per_question.append(
                {
                    **benchmark,
                    "method": method,
                    "hit_rank": metrics["hit_rank"],
                    "recall@1": metrics["recall_at_1"],
                    "recall@3": metrics["recall_at_3"],
                    "recall@5": metrics["recall_at_5"],
                    "reciprocal_rank": metrics["reciprocal_rank"],
                    "top_results": [
                        {
                            "rank": rank,
                            "chunk_id": result["chunk_id"],
                            "document_id": result["document_id"],
                            "page_start": result.get("page_start"),
                            "page_end": result.get("page_end"),
                            "score": result.get("similarity")
                            if method == "vector"
                            else result.get("lexical_score")
                            if method == "lexical"
                            else result.get("rrf_score"),
                        }
                        for rank, result
                        in enumerate(results[:TOP_K], start=1)
                    ],
                }
            )

    summary = {}

    for method, data in grouped_metrics.items():
        summary[method] = aggregate(
            data["metrics"]
        )

    output = {
        "config": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "ground_truth": "document_id + expected_page",
        },
        "summary": summary,
        "results": per_question,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(summary)

    print()
    print(f"Detailed results written to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
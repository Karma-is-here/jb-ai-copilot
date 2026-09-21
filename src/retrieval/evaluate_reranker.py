import json
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker


BENCHMARK_DIR = Path("data/benchmarks")
OUTPUT_FILE = BENCHMARK_DIR / "reranker_evaluation.json"

CANDIDATE_K = 20
RERANK_TOP_K = 20


def load_benchmarks():
    benchmarks = []

    for path in BENCHMARK_DIR.rglob("*_questions.json"):
        relative = path.relative_to(BENCHMARK_DIR)

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
    if result["document_id"] != expected_document_id:
        return False

    page_start = result.get("page_start")
    page_end = result.get("page_end")

    if page_start is None or page_end is None:
        return False

    return page_start <= expected_page <= page_end


def first_hit_rank(results, benchmark):
    for rank, result in enumerate(results, start=1):
        if is_hit(
            result,
            benchmark["document_id"],
            benchmark["expected_page"],
        ):
            return rank

    return None


def compute_metrics(results, benchmark):
    hit_rank = first_hit_rank(results, benchmark)

    return {
        "recall@1": int(hit_rank is not None and hit_rank <= 1),
        "recall@3": int(hit_rank is not None and hit_rank <= 3),
        "recall@5": int(hit_rank is not None and hit_rank <= 5),
        "reciprocal_rank": (
            1.0 / hit_rank
            if hit_rank is not None
            else 0.0
        ),
        "hit_rank": hit_rank,
    }


def aggregate(results):
    n = len(results)

    if n == 0:
        return {
            "questions": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "MRR@20": 0.0,
        }

    return {
        "questions": n,
        "recall@1": sum(r["recall@1"] for r in results) / n,
        "recall@3": sum(r["recall@3"] for r in results) / n,
        "recall@5": sum(r["recall@5"] for r in results) / n,
        "MRR@20": sum(r["reciprocal_rank"] for r in results) / n,
    }


def main():
    benchmarks = load_benchmarks()

    print(f"Loaded {len(benchmarks)} benchmark questions.")

    print("\nLoading hybrid retriever...")
    hybrid = HybridRetriever()

    print("\nLoading cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    results = []

    for i, benchmark in enumerate(benchmarks, start=1):

        print(
            f"[{i}/{len(benchmarks)}] "
            f"{benchmark['question']}"
        )

        candidates = hybrid.search(
            benchmark["question"],
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked = reranker.rerank(
            benchmark["question"],
            candidates,
            top_k=RERANK_TOP_K,
        )

        metrics = compute_metrics(
            reranked,
            benchmark,
        )

        results.append(
            {
                **benchmark,
                **metrics,
                "top_results": [
                    {
                        "rank": rank,
                        "chunk_id": result["chunk_id"],
                        "document_id": result["document_id"],
                        "page_start": result.get("page_start"),
                        "page_end": result.get("page_end"),
                        "reranker_score": result.get(
                            "reranker_score"
                        ),
                    }
                    for rank, result
                    in enumerate(reranked, start=1)
                ],
            }
        )

    summary = aggregate(results)

    output = {
        "config": {
            "candidate_k": CANDIDATE_K,
            "rerank_top_k": RERANK_TOP_K,
            "ground_truth": "document_id + expected_page",
        },
        "summary": summary,
        "results": results,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 72)

    print(f"Questions : {summary['questions']}")
    print(f"Recall@1  : {summary['recall@1']:.3f}")
    print(f"Recall@3  : {summary['recall@3']:.3f}")
    print(f"Recall@5  : {summary['recall@5']:.3f}")
    print(f"MRR@20    : {summary['MRR@20']:.3f}")

    print("=" * 72)

    print(f"\nDetailed results written to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
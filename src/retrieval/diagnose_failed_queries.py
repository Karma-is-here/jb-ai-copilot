import json
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker


FILE = Path("data/benchmarks/generation_evaluation.json")


def normalize_document_id(benchmark_file):
    path = Path(benchmark_file)
    parts = list(path.parts)

    try:
        idx = parts.index("benchmarks")
        parts = parts[idx + 1:]
    except ValueError:
        pass

    filename = parts[-1]

    if filename.endswith("_questions.json"):
        filename = filename[:-len("_questions.json")]

    parts[-1] = filename

    return "/".join(parts)


def is_expected(chunk, expected_document, expected_page):
    return (
        chunk.get("document_id") == expected_document
        and chunk.get("page_start") is not None
        and chunk.get("page_end") is not None
        and chunk["page_start"] <= expected_page <= chunk["page_end"]
    )


def main():

    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Identify the 17 current failures.
    # ---------------------------------------------------------

    failures = []

    for r in data["results"]:

        expected_document = normalize_document_id(
            r["benchmark_file"]
        )

        expected_page = r["expected_page"]

        existing_top5 = r.get("retrieved_chunks", [])

        if not any(
            is_expected(
                c,
                expected_document,
                expected_page,
            )
            for c in existing_top5
        ):
            failures.append(r)

    print("=" * 100)
    print("FAILED-QUERY RETRIEVAL PATH DIAGNOSTIC")
    print("=" * 100)
    print(f"Queries to diagnose: {len(failures)}")
    print()

    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    hybrid_contains = 0
    reranker_keeps_top5 = 0
    reranker_drops = 0
    completely_missing = 0

    for i, r in enumerate(failures, 1):

        question = r["question"]

        expected_document = normalize_document_id(
            r["benchmark_file"]
        )

        expected_page = r["expected_page"]

        print("=" * 100)
        print(f"FAILURE {i}")
        print("=" * 100)

        print(f"QUESTION: {question}")
        print(
            f"EXPECTED: {expected_document} "
            f"page {expected_page}"
        )

        # -----------------------------------------------------
        # Hybrid top 20
        # -----------------------------------------------------

        hybrid_results = retriever.search(
            question,
            top_k=20,
            candidate_k=20,
        )

        hybrid_match = None

        for rank, chunk in enumerate(
            hybrid_results,
            1,
        ):
            if is_expected(
                chunk,
                expected_document,
                expected_page,
            ):
                hybrid_match = (rank, chunk)
                break

        if hybrid_match:

            hybrid_contains += 1

            rank, chunk = hybrid_match

            print()
            print(
                f"HYBRID: FOUND at rank {rank}"
            )

            print(
                f"  chunk={chunk.get('chunk_id')}"
            )

            print(
                f"  document={chunk.get('document_id')}"
            )

            print(
                f"  page={chunk.get('page_start')}-"
                f"{chunk.get('page_end')}"
            )

            print(
                f"  vector_rank={chunk.get('vector_rank')}"
            )

            print(
                f"  lexical_rank={chunk.get('lexical_rank')}"
            )

            # -------------------------------------------------
            # Rerank
            # -------------------------------------------------

            reranked = reranker.rerank(
                question,
                hybrid_results,
                top_k=20,
            )

            rerank_match = None

            for rank, chunk in enumerate(
                reranked,
                1,
            ):
                if is_expected(
                    chunk,
                    expected_document,
                    expected_page,
                ):
                    rerank_match = (rank, chunk)
                    break

            if rerank_match:

                rerank_rank, rerank_chunk = rerank_match

                print(
                    f"RERANKER: rank {rerank_rank}"
                )

                print(
                    f"  reranker_score="
                    f"{rerank_chunk.get('reranker_score')}"
                )

                if rerank_rank <= 5:
                    reranker_keeps_top5 += 1
                else:
                    reranker_drops += 1

            else:
                print(
                    "RERANKER: expected chunk disappeared"
                )

        else:

            completely_missing += 1

            print()
            print(
                "HYBRID: expected document/page NOT "
                "found in top-20"
            )

            # Show closest same-document results.
            same_document = [
                (rank, c)
                for rank, c in enumerate(
                    hybrid_results,
                    1,
                )
                if c.get("document_id")
                == expected_document
            ]

            if same_document:

                print()
                print("Same-document candidates:")

                for rank, c in same_document[:5]:

                    print(
                        f"  rank={rank} "
                        f"page={c.get('page_start')}-"
                        f"{c.get('page_end')} "
                        f"vector={c.get('vector_rank')} "
                        f"lexical={c.get('lexical_rank')}"
                    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Original failed queries       : "
        f"{len(failures)}"
    )

    print(
        f"Expected chunk in Hybrid@20   : "
        f"{hybrid_contains}"
    )

    print(
        f"Expected chunk missing Hybrid : "
        f"{completely_missing}"
    )

    print(
        f"Hybrid → Reranker Top-5       : "
        f"{reranker_keeps_top5}"
    )

    print(
        f"Reranker pushed below Top-5   : "
        f"{reranker_drops}"
    )


if __name__ == "__main__":
    main()
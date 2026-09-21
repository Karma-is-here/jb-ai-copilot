import json
from pathlib import Path


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

    results = data["results"]

    failures = []

    for r in results:

        expected_document = normalize_document_id(
            r["benchmark_file"]
        )

        expected_page = r["expected_page"]

        chunks = r.get("retrieved_chunks", [])

        if any(
            is_expected(
                c,
                expected_document,
                expected_page,
            )
            for c in chunks
        ):
            continue

        failures.append({
            "question": r["question"],
            "expected_document": expected_document,
            "expected_page": expected_page,
            "chunks": chunks,
        })

    print("=" * 100)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 100)
    print(f"Failures: {len(failures)}")
    print()

    for i, failure in enumerate(failures, 1):

        print("=" * 100)
        print(f"FAILURE {i}")
        print("=" * 100)

        print(f"QUESTION:")
        print(failure["question"])

        print()
        print(
            f"EXPECTED: "
            f"{failure['expected_document']} "
            f"page {failure['expected_page']}"
        )

        print()
        print("TOP RETRIEVED CHUNKS:")
        print("-" * 100)

        for rank, chunk in enumerate(
            failure["chunks"],
            1,
        ):

            print(
                f"{rank}. "
                f"{chunk.get('document_id')} "
                f"p{chunk.get('page_start')}-"
                f"{chunk.get('page_end')} "
                f"| "
                f"reranker={chunk.get('reranker_score')} "
                f"| "
                f"vector_rank={chunk.get('vector_rank')} "
                f"| "
                f"lexical_rank={chunk.get('lexical_rank')}"
            )

            text = chunk.get("text", "")

            text = text.replace("\n", " ")

            print(
                f"   {text[:350]}"
            )

            print()


if __name__ == "__main__":
    main()
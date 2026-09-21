import json
from pathlib import Path


FILE = Path("data/benchmarks/generation_evaluation.json")


def normalize_document_id(benchmark_file):
    """
    Example:

    data\\benchmarks\\investment\\advisory\\advisory-mandates_questions.json

    ->

    investment/advisory/advisory-mandates
    """

    path = Path(benchmark_file)

    parts = list(path.parts)

    # Find the benchmarks directory.
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


def source_matches_expected(
    source,
    expected_document,
    expected_page,
):
    if not source:
        return False

    if source.get("document_id") != expected_document:
        return False

    page_start = source.get("page_start")
    page_end = source.get("page_end")

    if page_start is None or page_end is None:
        return False

    return page_start <= expected_page <= page_end


def main():

    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    total = len(results)

    retrieval_failure = []
    retrieval_success = []

    citation_failure = []
    citation_success = []

    no_citation = []

    for r in results:

        benchmark_file = r.get("benchmark_file", "")
        expected_page = r.get("expected_page")

        expected_document = normalize_document_id(
            benchmark_file
        )

        retrieved = r.get("retrieved_chunks", [])

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        retrieval_ok = any(
            chunk.get("document_id") == expected_document
            and chunk.get("page_start") is not None
            and chunk.get("page_end") is not None
            and chunk["page_start"] <= expected_page <= chunk["page_end"]
            for chunk in retrieved
        )

        if retrieval_ok:
            retrieval_success.append(r)
        else:
            retrieval_failure.append(r)

        # -----------------------------------------------------
        # Citations
        # -----------------------------------------------------

        citation_validation = r.get(
            "citation_validation",
            {}
        )

        citations = citation_validation.get(
            "referenced_sources",
            []
        )

        sources = {
            str(k): v
            for k, v in r.get("sources", {}).items()
        }

        if not citations:
            no_citation.append(r)
            continue

        correct_citation = any(
            source_matches_expected(
                sources.get(str(citation_id)),
                expected_document,
                expected_page,
            )
            for citation_id in citations
        )

        if correct_citation:
            citation_success.append(r)
        else:
            citation_failure.append(r)

    # ---------------------------------------------------------
    # Cross classification
    # ---------------------------------------------------------

    retrieval_ok_citation_ok = 0
    retrieval_ok_citation_bad = 0
    retrieval_bad = 0

    for r in results:

        benchmark_file = r.get("benchmark_file", "")
        expected_page = r.get("expected_page")

        expected_document = normalize_document_id(
            benchmark_file
        )

        retrieved = r.get("retrieved_chunks", [])

        retrieval_ok = any(
            chunk.get("document_id") == expected_document
            and chunk.get("page_start") is not None
            and chunk.get("page_end") is not None
            and chunk["page_start"] <= expected_page <= chunk["page_end"]
            for chunk in retrieved
        )

        citations = r.get(
            "citation_validation",
            {}
        ).get(
            "referenced_sources",
            []
        )

        sources = {
            str(k): v
            for k, v in r.get("sources", {}).items()
        }

        citation_ok = any(
            source_matches_expected(
                sources.get(str(citation_id)),
                expected_document,
                expected_page,
            )
            for citation_id in citations
        )

        if not retrieval_ok:
            retrieval_bad += 1

        elif citation_ok:
            retrieval_ok_citation_ok += 1

        else:
            retrieval_ok_citation_bad += 1

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("=" * 80)
    print("FAILURE MODE ANALYSIS")
    print("=" * 80)

    print(f"Total questions                 : {total}")
    print()

    print(
        f"Retrieval success              : "
        f"{len(retrieval_success)} "
        f"({len(retrieval_success) / total:.3f})"
    )

    print(
        f"Retrieval failure              : "
        f"{len(retrieval_failure)} "
        f"({len(retrieval_failure) / total:.3f})"
    )

    print()

    print(
        f"Correct citation present       : "
        f"{len(citation_success)} "
        f"({len(citation_success) / total:.3f})"
    )

    print(
        f"No correct citation            : "
        f"{len(citation_failure)} "
        f"({len(citation_failure) / total:.3f})"
    )

    print(
        f"No citations                   : "
        f"{len(no_citation)} "
        f"({len(no_citation) / total:.3f})"
    )

    print()
    print("=" * 80)
    print("PIPELINE DIAGNOSIS")
    print("=" * 80)

    print(
        f"Retrieval failure               : "
        f"{retrieval_bad}"
    )

    print(
        f"Retrieval OK + citation OK      : "
        f"{retrieval_ok_citation_ok}"
    )

    print(
        f"Retrieval OK + citation failure : "
        f"{retrieval_ok_citation_bad}"
    )

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if retrieval_bad:
        print(
            "Some questions fail because the expected "
            "document/page did not reach the final context."
        )
    else:
        print(
            "All benchmark questions have the expected "
            "document/page in the final retrieved context."
        )

    if retrieval_ok_citation_bad:
        print(
            "Some questions have correct evidence available "
            "but the answer does not cite the expected source."
        )
    else:
        print(
            "Every question with retrieved evidence also has "
            "at least one citation targeting the expected source."
        )


if __name__ == "__main__":
    main()
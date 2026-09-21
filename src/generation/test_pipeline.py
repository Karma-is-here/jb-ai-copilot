from src.generation.pipeline import RAGPipeline


def main():

    question = (
        "What does an advisory mandate mean, "
        "and who retains the final investment decision?"
    )

    pipeline = RAGPipeline(
        candidate_k=20,
        context_k=5,
    )

    result = pipeline.answer(question)

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print()
    print("=" * 80)
    print("CITATION VALIDATION")
    print("=" * 80)

    validation = result["citation_validation"]

    print(
        "Citations found:",
        validation["citations_found"],
    )

    print(
        "Referenced sources:",
        validation["referenced_sources"],
    )

    print(
        "Invalid sources:",
        validation["invalid_sources"],
    )

    print(
        "Valid:",
        validation["valid"],
    )

    print()
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for citation_id, source in result["sources"].items():
        print(
            f"[{citation_id}] "
            f"{source['document_id']} "
            f"pages "
            f"{source['page_start']}-"
            f"{source['page_end']}"
        )


if __name__ == "__main__":
    main()
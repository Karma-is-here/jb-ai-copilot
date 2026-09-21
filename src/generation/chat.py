from src.generation.pipeline import RAGPipeline


def main():
    print("=" * 80)
    print("JB COPILOT — INTERACTIVE RAG")
    print("=" * 80)
    print("Type 'exit' to quit.\n")

    pipeline = RAGPipeline()

    while True:
        question = input("\nAsk: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        print("\nRetrieving and generating...\n")

        try:
            result = pipeline.answer(question)

            print("=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(result["answer"])

            print("\n" + "=" * 80)
            print("SOURCES")
            print("=" * 80)

            for citation_id, source in result["sources"].items():
                print(
                    f"[{citation_id}] "
                    f"{source['document_id']} "
                    f"pages {source['page_start']}-{source['page_end']}"
                )

            print("\n" + "=" * 80)
            print("CITATION VALIDATION")
            print("=" * 80)
            print(result["citation_validation"])

        except Exception as e:
            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
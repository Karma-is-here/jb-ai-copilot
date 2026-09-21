from lexical import LexicalRetriever


def main():

    retriever = LexicalRetriever()

    query = "What does an advisory mandate mean?"

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print(f"Query: {query}")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\n#{i}")
        print(f"Chunk:    {result['chunk_id']}")
        print(f"Document: {result['document_id']}")
        print(
            f"Pages:    "
            f"{result['page_start']}-{result['page_end']}"
        )
        print(
            f"Score:    "
            f"{result['lexical_score']:.4f}"
        )
        print(
            f"Text:     "
            f"{result['text'][:500]}..."
        )


if __name__ == "__main__":
    main()
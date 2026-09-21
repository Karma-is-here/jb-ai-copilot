from vector import VectorRetriever


def main():
    retriever = VectorRetriever()

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
        print(f"Chunk:      {result['chunk_id']}")
        print(f"Document:   {result['document_id']}")
        print(f"Pages:      {result['page_start']}-{result['page_end']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Strategy:   {result['chunking_strategy']}")
        print(f"Text:       {result['text'][:500]}...")


if __name__ == "__main__":
    main()
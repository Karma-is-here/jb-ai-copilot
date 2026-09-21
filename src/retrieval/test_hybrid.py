from src.retrieval.hybrid import HybridRetriever


def main():

    retriever = HybridRetriever()

    query = "What does an advisory mandate mean?"

    results = retriever.search(
        query=query,
        top_k=10,
        candidate_k=20,
    )

    print()
    print(f"Query: {query}")
    print("=" * 100)

    for i, result in enumerate(results, start=1):

        print(f"\n#{i}")
        print(f"Chunk:        {result['chunk_id']}")
        print(f"Document:     {result['document_id']}")
        print(
            f"Pages:        "
            f"{result['page_start']}-{result['page_end']}"
        )
        print(
            f"RRF score:    "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Vector rank:  "
            f"{result['vector_rank']}"
        )
        print(
            f"Lexical rank: "
            f"{result['lexical_rank']}"
        )
        print(f"Text:         {result['text'][:500]}...")


if __name__ == "__main__":
    main()
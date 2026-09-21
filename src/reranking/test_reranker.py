from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker


def main():
    query = "What does an advisory mandate mean?"

    print("Loading hybrid retriever...")
    hybrid = HybridRetriever()

    print("Retrieving candidates...")
    candidates = hybrid.search(
        query,
        top_k=20,
        candidate_k=20,
    )

    print(f"Retrieved {len(candidates)} candidates.")

    print("\nLoading reranker...")
    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for result in results:
        print(
            f"\nRank: {result['reranker_rank']}"
            f"\nReranker score: {result['reranker_score']:.4f}"
            f"\nDocument: {result['document_id']}"
            f"\nPages: {result['page_start']}-{result['page_end']}"
            f"\nChunk: {result['chunk_id']}"
        )

        print(f"Text: {result['text'][:400]}...")


if __name__ == "__main__":
    main()
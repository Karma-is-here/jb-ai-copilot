from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.generation.context import ContextAssembler


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

    print("Loading reranker...")
    reranker = CrossEncoderReranker()

    reranked = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print("Assembling context...")
    assembler = ContextAssembler(max_sources=5)

    sources = assembler.assemble(reranked)

    context = assembler.format_for_llm(sources)

    citation_map = assembler.citation_map(sources)

    print()
    print("=" * 80)
    print("LLM CONTEXT")
    print("=" * 80)
    print(context)

    print()
    print("=" * 80)
    print("CITATION MAP")
    print("=" * 80)

    for citation_id, metadata in citation_map.items():
        print(
            f"[{citation_id}] "
            f"{metadata['document_id']} "
            f"pages "
            f"{metadata['page_start']}-"
            f"{metadata['page_end']}"
        )


if __name__ == "__main__":
    main()
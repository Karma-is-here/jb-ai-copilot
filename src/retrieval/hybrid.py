from .lexical import LexicalRetriever
from .vector import VectorRetriever


class HybridRetriever:

    def __init__(self):
        self.vector = VectorRetriever()
        self.lexical = LexicalRetriever()

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
    ):
        vector_results = self.vector.search(
            query=query,
            top_k=candidate_k,
        )

        lexical_results = self.lexical.search(
            query=query,
            top_k=candidate_k,
        )

        fused = {}

        # ---------------------------------------------
        # Vector contribution
        # ---------------------------------------------

        for rank, result in enumerate(vector_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result["document_id"],
                    "text": result["text"],
                    "source_file": result["source_file"],
                    "page_start": result["page_start"],
                    "page_end": result["page_end"],
                    "section": result["section"],
                    "chunking_strategy": result[
                        "chunking_strategy"
                    ],
                    "chunk_index": result["chunk_index"],
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "lexical_rank": None,
                }

            fused[chunk_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

            fused[chunk_id]["vector_rank"] = rank

        # ---------------------------------------------
        # Lexical contribution
        # ---------------------------------------------

        for rank, result in enumerate(lexical_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": result["document_id"],
                    "text": result["text"],
                    "source_file": result["source_file"],
                    "page_start": result["page_start"],
                    "page_end": result["page_end"],
                    "section": result["section"],
                    "chunking_strategy": result[
                        "chunking_strategy"
                    ],
                    "chunk_index": result["chunk_index"],
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "lexical_rank": None,
                }

            fused[chunk_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

            fused[chunk_id]["lexical_rank"] = rank

        # ---------------------------------------------
        # Final ranking
        # ---------------------------------------------

        results = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return results[:top_k]
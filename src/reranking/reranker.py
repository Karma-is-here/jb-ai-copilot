from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder.

    The model jointly reads the query and candidate chunk,
    producing a relevance score for each pair.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5):
        """
        Rerank candidate chunks for a query.

        Parameters
        ----------
        query:
            User's search query.

        candidates:
            Candidate chunks returned by hybrid retrieval.

        top_k:
            Number of reranked chunks to return.

        Returns
        -------
        list[dict]
            Candidates sorted by reranker score.
        """

        if not candidates:
            return []

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["reranker_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        for rank, result in enumerate(reranked[:top_k], start=1):
            result["reranker_rank"] = rank

        return reranked[:top_k]
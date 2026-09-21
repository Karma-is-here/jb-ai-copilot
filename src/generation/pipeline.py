from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker

from src.generation.prompt import SYSTEM_PROMPT, build_prompt
from src.generation.llm import LLMClient
from src.generation.context import (
    ContextAssembler,
    validate_and_normalize_citations,
)


class RAGPipeline:
    """
    End-to-end retrieval-augmented generation pipeline.

    Flow:
        Query
          ↓
        Hybrid retrieval
          ↓
        RRF
          ↓
        Cross-encoder reranking
          ↓
        Context assembly
          ↓
        LLM generation
          ↓
        Answer + citations
    """

    def __init__(
        self,
        candidate_k: int = 20,
        context_k: int = 5,
    ):
        print("Initializing RAG pipeline...")

        self.candidate_k = candidate_k
        self.context_k = context_k

        self.retriever = HybridRetriever()

        self.reranker = CrossEncoderReranker()

        self.context_assembler = ContextAssembler(
            max_sources=context_k
        )

        self.llm = LLMClient()

    def retrieve(self, question: str):
        """
        Retrieve and rerank candidate chunks.
        """

        candidates = self.retriever.search(
            question,
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        reranked = self.reranker.rerank(
            question,
            candidates,
            top_k=self.context_k,
        )

        return reranked

    def answer(self, question: str):
        """
        Run the complete RAG pipeline.
        """

        # 1. Retrieval + reranking
        reranked = self.retrieve(question)

        # 2. Context assembly
        sources = self.context_assembler.assemble(
            reranked
        )

        context = self.context_assembler.format_for_llm(
            sources
        )

        citation_map = self.context_assembler.citation_map(
            sources
        )

        # 3. Prompt construction
        user_prompt = build_prompt(
            question,
            context,
        )

        # 4. LLM generation
        raw_answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        citation_validation = validate_and_normalize_citations(
            raw_answer,
            citation_map,
        )

        return {
            "question": question,
            "answer": citation_validation["answer"],
            "raw_answer": raw_answer,
            "sources": citation_map,
            "retrieved_chunks": reranked,
            "citation_validation": citation_validation,
        }
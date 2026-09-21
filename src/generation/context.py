import re

from dataclasses import dataclass


@dataclass
class ContextSource:
    """
    A single source that can be provided to the LLM.

    Keeps the original retrieval provenance intact.
    """

    citation_id: int
    chunk_id: str
    document_id: str
    source_file: str | None
    page_start: int | None
    page_end: int | None
    section: str | None
    text: str
    reranker_score: float | None = None


class ContextAssembler:
    """
    Converts reranked retrieval results into a structured
    LLM-ready context while preserving provenance.
    """

    def __init__(self, max_sources: int = 5):
        self.max_sources = max_sources

    def assemble(
        self,
        results: list[dict],
    ) -> list[ContextSource]:
        """
        Convert reranked results into citation-aware sources.
        """

        selected = results[:self.max_sources]

        sources = []

        for citation_id, result in enumerate(
            selected,
            start=1,
        ):
            sources.append(
                ContextSource(
                    citation_id=citation_id,
                    chunk_id=result["chunk_id"],
                    document_id=result["document_id"],
                    source_file=result.get("source_file"),
                    page_start=result.get("page_start"),
                    page_end=result.get("page_end"),
                    section=result.get("section"),
                    text=result["text"],
                    reranker_score=result.get(
                        "reranker_score"
                    ),
                )
            )

        return sources

    def format_for_llm(
        self,
        sources: list[ContextSource],
    ) -> str:
        """
        Produce a citation-aware text context for the LLM.
        """

        blocks = []

        for source in sources:

            if source.page_start == source.page_end:
                page = str(source.page_start)

            elif (
                source.page_start is not None
                and source.page_end is not None
            ):
                page = (
                    f"{source.page_start}-"
                    f"{source.page_end}"
                )

            else:
                page = "unknown"

            blocks.append(
                f"[SOURCE {source.citation_id}]\n"
                f"Document: {source.document_id}\n"
                f"Page: {page}\n"
                f"Chunk ID: {source.chunk_id}\n"
                f"Content:\n"
                f"{source.text}"
            )

        return "\n\n".join(blocks)

    def citation_map(
        self,
        sources: list[ContextSource],
    ) -> dict:
        """
        Create a machine-readable mapping between citation IDs
        and their original document provenance.
        """

        return {
            str(source.citation_id): {
                "chunk_id": source.chunk_id,
                "document_id": source.document_id,
                "source_file": source.source_file,
                "page_start": source.page_start,
                "page_end": source.page_end,
                "section": source.section,
            }
            for source in sources
        }


def validate_and_normalize_citations(
    answer: str,
    citation_map: dict,
) -> dict:
    """
    Validate and normalize LLM citations.

    Supported formats:

        [1]
        [SOURCE 1]
        [Source 1]
        【1】
        【SOURCE 1】

    All valid citations are normalized to:

        [1]

    Status values:

        GROUNDED
        ABSTAINED
        INVALID_CITATION
        NO_CITATION
    """

    pattern = re.compile(
        r"(?:"
        r"\[\s*(?:SOURCE\s*)?(\d+)\s*\]"
        r"|"
        r"【\s*(?:SOURCE\s*)?(\d+)\s*】"
        r")",
        re.IGNORECASE,
    )

    matches = pattern.findall(answer)

    referenced_ids = []
    invalid_ids = []

    for match in matches:
        value = match[0] or match[1]
        citation_id = int(value)

        if str(citation_id) in citation_map:

            if citation_id not in referenced_ids:
                referenced_ids.append(citation_id)

        else:

            if citation_id not in invalid_ids:
                invalid_ids.append(citation_id)

    def normalize(match):
        value = match.group(1) or match.group(2)
        return f"[{int(value)}]"

    normalized_answer = pattern.sub(
        normalize,
        answer,
    )

    # Detect whether the model explicitly abstained.
    abstention_patterns = [
        r"\bdo not contain\b",
        r"\bdoes not contain\b",
        r"\bnot contain\b",
        r"\bno information\b",
        r"\binsufficient information\b",
        r"\bnot enough information\b",
        r"\bcannot answer\b",
        r"\bunable to answer\b",
        r"\bnot available in the (?:provided|available) sources\b",
    ]

    is_abstention = any(
        re.search(
            pattern,
            normalized_answer,
            re.IGNORECASE,
        )
        for pattern in abstention_patterns
    )

    if is_abstention and len(invalid_ids) == 0:
        status = "ABSTAINED"

    elif len(invalid_ids) > 0:
        status = "INVALID_CITATION"

    elif len(referenced_ids) > 0:
        status = "GROUNDED"

    else:
        status = "NO_CITATION"

    return {
        "answer": normalized_answer,
        "citations_found": len(matches),
        "referenced_sources": referenced_ids,
        "invalid_sources": invalid_ids,
        "valid": status in {
            "GROUNDED",
            "ABSTAINED",
        },
        "status": status,
    }
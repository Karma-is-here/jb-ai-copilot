from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from src.api import conversations_repo, documents_repo
from src.api.auth import Identity, get_current_identity
from src.api.ids import derive_conversation_title
from src.api.schemas import ChatRequest, ChatResponse, Diagnostics, Source

router = APIRouter(tags=["chat"])


def _citation_map_to_sources(citation_map: dict) -> list[Source]:
    sources: list[Source] = []
    for citation_id_str, entry in citation_map.items():
        metadata = documents_repo.get_document_metadata(entry["document_id"])
        sources.append(
            Source(
                citation_id=int(citation_id_str),
                chunk_id=entry["chunk_id"],
                document_id=entry["document_id"],
                source_file=entry.get("source_file"),
                page_start=entry.get("page_start"),
                page_end=entry.get("page_end"),
                section=entry.get("section"),
                **metadata,
            )
        )
    sources.sort(key=lambda s: s.citation_id)
    return sources


def _handle_chat(pipeline, identity, body: ChatRequest) -> ChatResponse:
    """
    All the synchronous work for one /api/chat call (DB reads/writes via
    psycopg_pool, plus RAGPipeline.answer()) — run as a single unit on a
    threadpool worker so it never blocks the event loop.
    """
    conversation_id = body.conversation_id
    if not conversation_id:
        title = derive_conversation_title(body.question)
        conversation = conversations_repo.create_conversation(identity.user_id, title)
        conversation_id = conversation["id"]

    conversations_repo.insert_message(
        conversation_id=conversation_id,
        role="user",
        content=body.question,
        status="complete",
    )

    result = pipeline.answer(body.question)

    validation = result["citation_validation"]
    sources = _citation_map_to_sources(result["sources"])
    diagnostics = Diagnostics(
        retrieval_candidate_count=len(result.get("retrieved_chunks", [])),
        reranked_count=len(sources),
        citation_status=validation["status"],
    )

    assistant_message = conversations_repo.insert_message(
        conversation_id=conversation_id,
        role="assistant",
        content=validation["answer"],
        status="abstained" if validation["status"] == "ABSTAINED" else "complete",
        answer_status=validation["status"],
        sources=[s.model_dump(by_alias=True) for s in sources],
        diagnostics=diagnostics.model_dump(by_alias=True),
        related_question=body.question,
    )

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=assistant_message["id"],
        answer=validation["answer"],
        status=validation["status"],
        sources=sources,
        created_at=assistant_message["created_at"],
        diagnostics=diagnostics,
    )


@router.post("/api/chat", response_model=ChatResponse)
async def ask_question(
    body: ChatRequest,
    request: Request,
    identity: Identity = Depends(get_current_identity),
) -> ChatResponse:
    pipeline = request.app.state.rag_pipeline
    return await run_in_threadpool(_handle_chat, pipeline, identity, body)

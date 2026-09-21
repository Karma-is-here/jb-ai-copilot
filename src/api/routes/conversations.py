from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from src.api import conversations_repo
from src.api.auth import Identity, get_current_identity
from src.api.ids import derive_conversation_title
from src.api.schemas import CamelModel, ChatMessage, Conversation

router = APIRouter(tags=["conversations"])


class CreateConversationRequest(CamelModel):
    # The frontend's conversationService.createConversation sends
    # { firstQuestion }; CamelModel's populate_by_name accepts either
    # first_question or firstQuestion as the incoming JSON key.
    first_question: str = ""


def _to_message(row: dict) -> ChatMessage:
    return ChatMessage(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=str(row["created_at"]),
        status=row["status"],
        answer_status=row["answer_status"],
        sources=row["sources"],
        diagnostics=row["diagnostics"],
        related_question=row["related_question"],
    )


def _to_conversation(row: dict, messages: list[dict]) -> Conversation:
    return Conversation(
        id=row["id"],
        title=row["title"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        messages=[_to_message(m) for m in messages],
    )


def _list_conversations(identity: Identity) -> list[Conversation]:
    rows = conversations_repo.list_conversations(identity.user_id)
    return [_to_conversation(row, conversations_repo.list_messages(row["id"])) for row in rows]


def _get_conversation(conversation_id: str) -> Conversation | None:
    row = conversations_repo.get_conversation(conversation_id)
    if not row:
        return None
    return _to_conversation(row, conversations_repo.list_messages(conversation_id))


def _create_conversation(identity: Identity, first_question: str) -> Conversation:
    title = derive_conversation_title(first_question) if first_question else "New conversation"
    row = conversations_repo.create_conversation(identity.user_id, title)
    return _to_conversation(row, [])


@router.get("/api/conversations", response_model=list[Conversation])
async def list_conversations(identity: Identity = Depends(get_current_identity)):
    return await run_in_threadpool(_list_conversations, identity)


@router.post("/api/conversations", response_model=Conversation)
async def create_conversation(
    body: CreateConversationRequest, identity: Identity = Depends(get_current_identity)
):
    return await run_in_threadpool(_create_conversation, identity, body.first_question)


@router.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str, identity: Identity = Depends(get_current_identity)
):
    conversation = await run_in_threadpool(_get_conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Conversation not found."}},
        )
    return conversation


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str, identity: Identity = Depends(get_current_identity)
):
    await run_in_threadpool(conversations_repo.delete_conversation, conversation_id)

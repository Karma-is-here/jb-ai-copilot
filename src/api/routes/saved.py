from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from src.api import saved_repo
from src.api.auth import Identity, get_current_identity
from src.api.schemas import SaveAnswerRequest, SaveDocumentRequest, SavedAnswer, SavedDocument

router = APIRouter(tags=["saved"])


def _to_saved_document(row: dict) -> SavedDocument:
    # saved_items stores the pointer generically as target_id (shared with
    # SavedAnswer, which uses its own dedicated message_id/conversation_id
    # columns instead) — map it to the documentId shape here.
    return SavedDocument(id=row["id"], document_id=row["target_id"], saved_at=row["saved_at"])


@router.get("/api/saved/answers", response_model=list[SavedAnswer])
async def list_saved_answers(identity: Identity = Depends(get_current_identity)):
    rows = await run_in_threadpool(saved_repo.list_saved, identity.user_id, "answer")
    return [SavedAnswer(**row) for row in rows]


@router.post("/api/saved/answers", response_model=SavedAnswer)
async def save_answer(
    body: SaveAnswerRequest, identity: Identity = Depends(get_current_identity)
):
    row = await run_in_threadpool(
        saved_repo.save_answer,
        identity.user_id,
        body.conversation_id,
        body.message_id,
        body.question,
        body.answer_excerpt,
    )
    return SavedAnswer(**row)


@router.delete("/api/saved/answers/{item_id}", status_code=204)
async def unsave_answer(item_id: str, identity: Identity = Depends(get_current_identity)):
    await run_in_threadpool(saved_repo.delete_saved, identity.user_id, item_id)


@router.get("/api/saved/documents", response_model=list[SavedDocument])
async def list_saved_documents(identity: Identity = Depends(get_current_identity)):
    rows = await run_in_threadpool(saved_repo.list_saved, identity.user_id, "document")
    return [_to_saved_document(row) for row in rows]


@router.post("/api/saved/documents", response_model=SavedDocument)
async def save_document(
    body: SaveDocumentRequest, identity: Identity = Depends(get_current_identity)
):
    row = await run_in_threadpool(saved_repo.save_document, identity.user_id, body.document_id)
    return _to_saved_document(row)


@router.delete("/api/saved/documents/{item_id}", status_code=204)
async def unsave_document(item_id: str, identity: Identity = Depends(get_current_identity)):
    await run_in_threadpool(saved_repo.delete_saved, identity.user_id, item_id)

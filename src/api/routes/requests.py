import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api import requests_repo
from src.api.auth import Identity, get_current_identity, require_admin
from src.api.config import UPLOADS_DIR
from src.api.ids import generate_id
from src.api.schemas import DocumentRequest, RejectRequestBody

router = APIRouter(tags=["requests"])


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"error": {"code": "NOT_FOUND", "message": "Request not found."}},
    )


def _save_upload(request_id: str, filename: str, content: bytes) -> str:
    safe_name = Path(filename).name or "upload.bin"
    request_dir = UPLOADS_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    dest = request_dir / safe_name
    dest.write_bytes(content)
    return str(dest)


def _create_request(
    file_name: str,
    content: bytes,
    title: str,
    description: str,
    reason: str,
    category: str,
    tags_json: str,
    request_knowledge_base_addition: str,
    related_question: str | None,
    submitted_by: str,
    submitted_by_email: str,
) -> DocumentRequest:
    request_id = generate_id("REQ").upper().replace("_", "-")
    stored_path = _save_upload(request_id, file_name, content)

    try:
        tags = json.loads(tags_json) if tags_json else []
    except json.JSONDecodeError:
        tags = []

    row = requests_repo.create_request(
        request_id=request_id,
        file_name=file_name,
        file_size_bytes=len(content),
        stored_path=stored_path,
        submitted_by=submitted_by,
        submitted_by_email=submitted_by_email,
        category=category,
        description=description or title,
        reason=reason,
        tags=tags,
        related_question=related_question,
        request_knowledge_base_addition=request_knowledge_base_addition.lower() == "true",
    )
    return DocumentRequest(**row)


@router.post("/api/documents/upload", response_model=DocumentRequest)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    reason: str = Form(""),
    category: str = Form(...),
    tags: str = Form("[]"),
    request_knowledge_base_addition: str = Form("true", alias="requestKnowledgeBaseAddition"),
    related_question: str | None = Form(None, alias="relatedQuestion"),
    submitted_by: str = Form(..., alias="submittedBy"),
    submitted_by_email: str = Form(..., alias="submittedByEmail"),
    _identity: Identity = Depends(get_current_identity),
):
    content = await file.read()
    return await run_in_threadpool(
        _create_request,
        file.filename or "upload.bin",
        content,
        title,
        description,
        reason,
        category,
        tags,
        request_knowledge_base_addition,
        related_question,
        submitted_by,
        submitted_by_email,
    )


@router.get("/api/requests", response_model=list[DocumentRequest])
async def list_my_requests(identity: Identity = Depends(get_current_identity)):
    rows = await run_in_threadpool(requests_repo.list_requests, identity.email)
    return [DocumentRequest(**row) for row in rows]


@router.get("/api/requests/{request_id}", response_model=DocumentRequest)
async def get_request(request_id: str, _identity: Identity = Depends(get_current_identity)):
    row = await run_in_threadpool(requests_repo.get_request, request_id)
    if not row:
        raise _not_found()
    return DocumentRequest(**row)


@router.get("/api/admin/requests", response_model=list[DocumentRequest])
async def list_all_requests(_identity: Identity = Depends(require_admin)):
    rows = await run_in_threadpool(requests_repo.list_requests, None)
    return [DocumentRequest(**row) for row in rows]


@router.post("/api/admin/requests/{request_id}/approve", response_model=DocumentRequest)
async def approve_request(request_id: str, identity: Identity = Depends(require_admin)):
    row = await run_in_threadpool(requests_repo.approve_request, request_id, identity.email)
    if not row:
        raise _not_found()
    return DocumentRequest(**row)


@router.post("/api/admin/requests/{request_id}/reject", response_model=DocumentRequest)
async def reject_request(
    request_id: str, body: RejectRequestBody, identity: Identity = Depends(require_admin)
):
    row = await run_in_threadpool(
        requests_repo.reject_request, request_id, identity.email, body.rejection_reason
    )
    if not row:
        raise _not_found()
    return DocumentRequest(**row)


@router.post("/api/admin/requests/{request_id}/mark-available", response_model=DocumentRequest)
async def mark_available(request_id: str, _identity: Identity = Depends(require_admin)):
    row = await run_in_threadpool(requests_repo.mark_available, request_id)
    if not row:
        raise _not_found()
    return DocumentRequest(**row)

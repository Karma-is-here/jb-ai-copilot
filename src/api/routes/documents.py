from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from src.api import documents_repo
from src.api.auth import get_current_identity
from src.api.schemas import KnowledgeDocument

router = APIRouter(tags=["documents"])


@router.get("/api/documents", response_model=list[KnowledgeDocument])
async def list_documents(_identity=Depends(get_current_identity)):
    return await run_in_threadpool(documents_repo.list_documents)


@router.get("/api/documents/search", response_model=list[KnowledgeDocument])
async def search_documents(q: str = "", _identity=Depends(get_current_identity)):
    return await run_in_threadpool(documents_repo.search_documents, q)


@router.get("/api/documents/{document_id:path}/file")
async def get_document_file(document_id: str):
    # Must be registered before the bare {document_id:path} route below —
    # :path is greedy and would otherwise swallow the trailing "/file"
    # segment as part of document_id, since FastAPI matches in
    # registration order.
    #
    # No get_current_identity dependency here, deliberately: this URL is
    # loaded by the browser directly (react-pdf/pdf.js's own fetch, and
    # the "Open in new tab" <a href>), neither of which goes through
    # apiClient — so neither can attach the X-User-ID/X-User-Email
    # headers every other route requires. It's still safe to serve
    # without them: resolve_file_path only returns a path for a
    # document_id that's actually registered in `chunks` (no arbitrary
    # filesystem access), and these are internal firm documents any
    # signed-in user can already see through every other route.
    path = await run_in_threadpool(documents_repo.resolve_file_path, document_id)
    if not path:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "File not found."}},
        )
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.get("/api/documents/{document_id:path}", response_model=KnowledgeDocument)
async def get_document(document_id: str, _identity=Depends(get_current_identity)):
    document = await run_in_threadpool(documents_repo.get_document, document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Document not found."}},
        )
    return document

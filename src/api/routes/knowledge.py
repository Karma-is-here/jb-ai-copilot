from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from src.api import documents_repo
from src.api.auth import get_current_identity
from src.api.schemas import KnowledgeArea, KnowledgeDocument

router = APIRouter(tags=["knowledge"])

_AREA_NAMES = {
    "investment": "Investment",
    "overview": "Overview",
    "technology": "Technology",
}
_AREA_DESCRIPTIONS = {
    "investment": "Advisory and discretionary mandate documentation",
    "overview": "Firm-wide overview and market reports",
    "technology": "Technology and engineering documentation",
}


def _list_areas() -> list[KnowledgeArea]:
    documents = documents_repo.list_documents()
    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.knowledge_area] = counts.get(doc.knowledge_area, 0) + 1

    return [
        KnowledgeArea(
            id=area_id,
            name=_AREA_NAMES.get(area_id, area_id.capitalize()),
            description=_AREA_DESCRIPTIONS.get(area_id, ""),
            document_count=count,
        )
        for area_id, count in sorted(counts.items())
    ]


@router.get("/api/knowledge", response_model=list[KnowledgeArea])
async def list_areas(_identity=Depends(get_current_identity)):
    return await run_in_threadpool(_list_areas)


@router.get("/api/knowledge/{area_id}", response_model=KnowledgeArea)
async def get_area(area_id: str, _identity=Depends(get_current_identity)):
    areas = await run_in_threadpool(_list_areas)
    for area in areas:
        if area.id == area_id:
            return area
    raise HTTPException(
        status_code=404,
        detail={"error": {"code": "NOT_FOUND", "message": "Knowledge area not found."}},
    )


@router.get("/api/knowledge/{area_id}/documents", response_model=list[KnowledgeDocument])
async def list_area_documents(area_id: str, _identity=Depends(get_current_identity)):
    return await run_in_threadpool(documents_repo.list_documents, area_id)

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from src.api.db import pool
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _check_rag_dependencies() -> dict:
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"database": "ok"}


@router.get("/api/health/rag")
async def health_rag(request: Request):
    pipeline_ready = getattr(request.app.state, "rag_pipeline", None) is not None
    db_status = await run_in_threadpool(_check_rag_dependencies)
    return {"pipeline": "ok" if pipeline_ready else "not_initialized", **db_status}

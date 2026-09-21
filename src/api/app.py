import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.config import FRONTEND_ORIGIN
from src.api.db import close_pool, ensure_schema, open_pool
from src.api.routes import chat, conversations, documents, health, knowledge, requests, saved
from src.generation.pipeline import RAGPipeline

logger = logging.getLogger("juls_copilot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAGPipeline loads real models (embedding + reranker) — build it once,
    # exactly like src/generation/chat.py already does, and reuse it for
    # every request via app.state.
    logger.info("Starting JULS Copilot API — initializing RAGPipeline...")
    app.state.rag_pipeline = RAGPipeline()

    open_pool()
    ensure_schema()

    yield

    close_pool()


app = FastAPI(title="JULS Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    # Routes already raise HTTPException(detail={"error": {...}}) in the
    # shape the frontend expects — pass it straight through. Anything that
    # raised a plain string detail (e.g. FastAPI's own internals) gets
    # wrapped into the same envelope.
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return _error_response(exc.status_code, "ERROR", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error_response(422, "VALIDATION_ERROR", "The request could not be processed.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Never serialize tracebacks/internal details to the client.
    logger.exception("Unhandled error while processing request")
    return _error_response(500, "INTERNAL_ERROR", "Something went wrong. Please try again.")


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(knowledge.router)
app.include_router(requests.router)
app.include_router(saved.router)

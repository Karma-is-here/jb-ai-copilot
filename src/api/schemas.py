from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """
    Base for every response model: write fields in snake_case in Python,
    serialize as camelCase JSON — matching the frontend's TypeScript
    types field-for-field without the frontend needing any changes.
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)


AnswerStatus = Literal["GROUNDED", "ABSTAINED", "INVALID_CITATION", "NO_CITATION"]
DocumentRequestStatus = Literal[
    "PENDING_REVIEW", "APPROVED", "PROCESSING", "AVAILABLE", "REJECTED"
]


class Source(CamelModel):
    citation_id: int
    chunk_id: str
    document_id: str
    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    text: str | None = None
    reranker_score: float | None = None
    document_title: str | None = None
    document_type: str | None = None
    document_date: str | None = None


class Diagnostics(CamelModel):
    retrieval_candidate_count: int | None = None
    reranked_count: int | None = None
    citation_status: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    question: str


class ChatResponse(CamelModel):
    conversation_id: str
    message_id: str
    answer: str
    status: AnswerStatus
    sources: list[Source]
    created_at: datetime
    diagnostics: Diagnostics | None = None


class ChatMessage(CamelModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime
    sources: list[Source] | None = None
    status: str | None = None
    answer_status: AnswerStatus | None = None
    diagnostics: Diagnostics | None = None
    related_question: str | None = None


class Conversation(CamelModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage]


class KnowledgeArea(CamelModel):
    id: str
    name: str
    description: str
    document_count: int


class KnowledgeDocument(CamelModel):
    id: str
    title: str
    type: str
    knowledge_area: str
    date: str
    version: str | None = None
    pages: int
    excerpt: str
    topics: list[str]
    file_url: str | None = None
    most_relevant: bool | None = None


class DocumentRequest(CamelModel):
    request_id: str
    file_name: str
    file_size_bytes: int
    submitted_by: str
    submitted_by_email: str
    category: str
    description: str
    reason: str
    tags: list[str]
    related_question: str | None = None
    request_knowledge_base_addition: bool
    status: DocumentRequestStatus
    submitted_at: datetime
    updated_at: datetime
    reviewed_by: str | None = None
    rejection_reason: str | None = None


class SavedAnswer(CamelModel):
    id: str
    conversation_id: str
    message_id: str
    question: str
    answer_excerpt: str
    saved_at: datetime


class SavedDocument(CamelModel):
    id: str
    document_id: str
    saved_at: datetime


class SaveAnswerRequest(CamelModel):
    conversation_id: str
    message_id: str
    question: str
    answer_excerpt: str


class SaveDocumentRequest(CamelModel):
    document_id: str


class RejectRequestBody(CamelModel):
    rejection_reason: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str


ChunkRow = dict[str, Any]

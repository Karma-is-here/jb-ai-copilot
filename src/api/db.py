from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

from src.api.config import postgres_dsn_kwargs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_id_idx ON conversations (user_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT,
    answer_status TEXT,
    sources JSONB,
    diagnostics JSONB,
    related_question TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_id_idx ON messages (conversation_id);

CREATE TABLE IF NOT EXISTS document_requests (
    request_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    stored_path TEXT NOT NULL,
    submitted_by TEXT NOT NULL,
    submitted_by_email TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    reason TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]',
    related_question TEXT,
    request_knowledge_base_addition BOOLEAN NOT NULL DEFAULT true,
    status TEXT NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by TEXT,
    rejection_reason TEXT
);

CREATE INDEX IF NOT EXISTS document_requests_submitted_by_email_idx
    ON document_requests (submitted_by_email);

CREATE TABLE IF NOT EXISTS saved_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    conversation_id TEXT,
    message_id TEXT,
    question TEXT,
    answer_excerpt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS saved_items_user_id_type_idx ON saved_items (user_id, type);
"""

pool = ConnectionPool(
    conninfo=make_conninfo(**postgres_dsn_kwargs()),
    min_size=1,
    max_size=10,
    open=False,
)


def open_pool() -> None:
    pool.open(wait=True, timeout=30)


def close_pool() -> None:
    pool.close()


def ensure_schema() -> None:
    """
    Idempotently creates the application tables this API layer owns.
    Never touches the existing `chunks` table/schema.
    """
    with pool.connection() as conn:
        conn.execute(_SCHEMA)

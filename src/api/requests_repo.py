import json

from src.api.db import pool
from src.api.ids import now_iso

_COLUMNS = (
    "request_id, file_name, file_size_bytes, submitted_by, submitted_by_email, "
    "category, description, reason, tags, related_question, "
    "request_knowledge_base_addition, status, submitted_at, updated_at, "
    "reviewed_by, rejection_reason"
)


def create_request(
    request_id: str,
    file_name: str,
    file_size_bytes: int,
    stored_path: str,
    submitted_by: str,
    submitted_by_email: str,
    category: str,
    description: str,
    reason: str,
    tags: list[str],
    related_question: str | None,
    request_knowledge_base_addition: bool,
) -> dict:
    now = now_iso()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO document_requests (
                request_id, file_name, file_size_bytes, stored_path,
                submitted_by, submitted_by_email, category, description, reason,
                tags, related_question, request_knowledge_base_addition,
                status, submitted_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                request_id,
                file_name,
                file_size_bytes,
                stored_path,
                submitted_by,
                submitted_by_email,
                category,
                description,
                reason,
                json.dumps(tags),
                related_question,
                request_knowledge_base_addition,
                "PENDING_REVIEW",
                now,
                now,
            ),
        )
    return get_request(request_id)


def get_request(request_id: str) -> dict | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS} FROM document_requests WHERE request_id = %s",
            (request_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [c.name for c in cur.description]
        return dict(zip(cols, row))


def list_requests(submitted_by_email: str | None = None) -> list[dict]:
    where = "WHERE submitted_by_email = %s" if submitted_by_email else ""
    params = (submitted_by_email,) if submitted_by_email else ()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLUMNS} FROM document_requests {where} ORDER BY submitted_at DESC",
            params,
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _update_status(request_id: str, **fields) -> dict | None:
    if not fields:
        return get_request(request_id)
    fields["updated_at"] = now_iso()
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    with pool.connection() as conn:
        conn.execute(
            f"UPDATE document_requests SET {set_clause} WHERE request_id = %s",
            (*fields.values(), request_id),
        )
    return get_request(request_id)


def approve_request(request_id: str, reviewed_by: str) -> dict | None:
    return _update_status(request_id, status="PROCESSING", reviewed_by=reviewed_by)


def reject_request(request_id: str, reviewed_by: str, rejection_reason: str) -> dict | None:
    return _update_status(
        request_id,
        status="REJECTED",
        reviewed_by=reviewed_by,
        rejection_reason=rejection_reason,
    )


def mark_available(request_id: str) -> dict | None:
    return _update_status(request_id, status="AVAILABLE")

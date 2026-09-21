from src.api.db import pool
from src.api.ids import generate_id, now_iso


_SELECT_COLUMNS = (
    "id, user_id, type, target_id, conversation_id, message_id, "
    "question, answer_excerpt, created_at AS saved_at"
)


def list_saved(user_id: str, item_type: str) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM saved_items
            WHERE user_id = %s AND type = %s
            ORDER BY created_at DESC
            """,
            (user_id, item_type),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def find_existing(user_id: str, item_type: str, target_id: str) -> dict | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM saved_items
            WHERE user_id = %s AND type = %s AND target_id = %s
            """,
            (user_id, item_type, target_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [c.name for c in cur.description]
        return dict(zip(cols, row))


def save_answer(user_id: str, conversation_id: str, message_id: str, question: str, answer_excerpt: str) -> dict:
    existing = find_existing(user_id, "answer", message_id)
    if existing:
        return existing

    item_id = generate_id("saved")
    now = now_iso()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO saved_items (
                id, user_id, type, target_id, conversation_id, message_id,
                question, answer_excerpt, created_at
            )
            VALUES (%s, %s, 'answer', %s, %s, %s, %s, %s, %s)
            """,
            (item_id, user_id, message_id, conversation_id, message_id, question, answer_excerpt, now),
        )
    return {
        "id": item_id,
        "user_id": user_id,
        "type": "answer",
        "target_id": message_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "question": question,
        "answer_excerpt": answer_excerpt,
        "saved_at": now,
    }


def save_document(user_id: str, document_id: str) -> dict:
    existing = find_existing(user_id, "document", document_id)
    if existing:
        return existing

    item_id = generate_id("saveddoc")
    now = now_iso()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO saved_items (id, user_id, type, target_id, created_at)
            VALUES (%s, %s, 'document', %s, %s)
            """,
            (item_id, user_id, document_id, now),
        )
    return {
        "id": item_id,
        "user_id": user_id,
        "type": "document",
        "target_id": document_id,
        "conversation_id": None,
        "message_id": None,
        "question": None,
        "answer_excerpt": None,
        "saved_at": now,
    }


def delete_saved(user_id: str, item_id: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "DELETE FROM saved_items WHERE id = %s AND user_id = %s",
            (item_id, user_id),
        )

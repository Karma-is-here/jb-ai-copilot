"""
Storage for conversations/messages — the new tables owned by this API
layer (src/api/db.py). Shared by routes/chat.py (which persists as a
side effect of answering) and routes/conversations.py (CRUD/history).
"""

import json

from src.api.db import pool
from src.api.ids import generate_id, now_iso


def create_conversation(user_id: str, title: str) -> dict:
    conversation_id = generate_id("conv")
    now = now_iso()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (conversation_id, user_id, title, now, now),
        )
    return {"id": conversation_id, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}


def get_conversation(conversation_id: str) -> dict | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [c.name for c in cur.description]
        return dict(zip(cols, row))


def list_conversations(user_id: str) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def touch_conversation(conversation_id: str, when: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = %s WHERE id = %s",
            (when, conversation_id),
        )


def insert_message(
    conversation_id: str,
    role: str,
    content: str,
    status: str | None = None,
    answer_status: str | None = None,
    sources: list[dict] | None = None,
    diagnostics: dict | None = None,
    related_question: str | None = None,
) -> dict:
    message_id = generate_id("msg")
    now = now_iso()

    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, status, answer_status,
                sources, diagnostics, related_question, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                message_id,
                conversation_id,
                role,
                content,
                status,
                answer_status,
                json.dumps(sources) if sources is not None else None,
                json.dumps(diagnostics) if diagnostics is not None else None,
                related_question,
                now,
            ),
        )

    touch_conversation(conversation_id, now)

    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "status": status,
        "answer_status": answer_status,
        "sources": sources,
        "diagnostics": diagnostics,
        "related_question": related_question,
        "created_at": now,
    }


def list_messages(conversation_id: str) -> list[dict]:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, conversation_id, role, content, status, answer_status,
                   sources, diagnostics, related_question, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def delete_conversation(conversation_id: str) -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))

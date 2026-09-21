"""
Read-only adapter over the existing `chunks` table (there is no
`documents` table in Postgres — no title, date, or page-count metadata
exists anywhere in the DB). This module is the single place that derives
document-level information, so a citation card in chat and the source
viewer page always agree, and it is shared by routes/chat.py,
routes/documents.py and routes/knowledge.py.

Best-effort by design: titles are derived from filenames, "type" is a
small fixed lookup by category — there is no real metadata to draw from.
"""

import json
from functools import lru_cache
from pathlib import Path

from src.api.config import MANIFEST_PATH, RAW_DATA_DIR
from src.api.db import pool
from src.api.schemas import KnowledgeDocument

_TYPE_BY_CATEGORY = {
    "investment": "Product Document",
    "overview": "Internal Guide",
    "technology": "Internal Guide",
}
_DEFAULT_TYPE = "Internal Guide"


@lru_cache(maxsize=1)
def _manifest_by_document_id() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {doc["document_id"]: doc for doc in data.get("documents", [])}


def _derive_title(filename: str) -> str:
    stem = Path(filename).stem
    words = stem.replace("_", " ").replace("-", " ").split()
    return " ".join(word.capitalize() for word in words) or stem


def _derive_type(category: str) -> str:
    return _TYPE_BY_CATEGORY.get(category, _DEFAULT_TYPE)


def _derive_date(manifest_entry: dict | None) -> str:
    if manifest_entry and manifest_entry.get("created_at"):
        # "2026-09-07T15:01:16.376910+00:00" -> "Sep 2026"
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(manifest_entry["created_at"])
            return dt.strftime("%b %Y")
        except ValueError:
            pass
    return "—"


def _derive_topics(category: str, manifest_entry: dict | None) -> list[str]:
    topics = [category]
    subcategory = (manifest_entry or {}).get("subcategory")
    # A real subfolder (e.g. "advisory"), not just the filename repeated
    # (categories with no subfolder have subcategory == filename).
    if subcategory and not subcategory.endswith(".pdf"):
        topics.append(subcategory)
    return topics


def _row_to_document(row: dict) -> KnowledgeDocument:
    document_id: str = row["document_id"]
    source_file: str = row["source_file"] or ""
    manifest_entry = _manifest_by_document_id().get(document_id)
    category = (manifest_entry or {}).get("category") or document_id.split("/")[0]

    return KnowledgeDocument(
        id=document_id,
        title=_derive_title(source_file or document_id),
        type=_derive_type(category),
        knowledge_area=category,
        date=_derive_date(manifest_entry),
        pages=row["max_page"] or 1,
        excerpt=(row["excerpt"] or "").strip()[:280],
        topics=_derive_topics(category, manifest_entry),
        file_url=f"/api/documents/{document_id}/file",
    )


_BASE_SELECT = """
    SELECT
        document_id,
        MAX(source_file) AS source_file,
        MIN(page_start) AS min_page,
        MAX(page_end) AS max_page,
        COUNT(*) AS chunk_count,
        (array_agg(text ORDER BY chunk_index ASC))[1] AS excerpt
    FROM chunks
"""


def list_documents(area_id: str | None = None) -> list[KnowledgeDocument]:
    where = ""
    params: tuple = ()
    if area_id:
        where = "WHERE document_id = %s OR document_id LIKE %s"
        params = (area_id, f"{area_id}/%")

    sql = f"{_BASE_SELECT} {where} GROUP BY document_id ORDER BY document_id"

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return [_row_to_document(row) for row in rows]


def get_document(document_id: str) -> KnowledgeDocument | None:
    sql = f"{_BASE_SELECT} WHERE document_id = %s GROUP BY document_id"

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (document_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [c.name for c in cur.description]
        return _row_to_document(dict(zip(cols, row)))


def get_document_metadata(document_id: str) -> dict:
    """Lightweight title/type/date lookup used to enrich chat citations."""
    doc = get_document(document_id)
    if not doc:
        return {"document_title": None, "document_type": None, "document_date": None}
    return {
        "document_title": doc.title,
        "document_type": doc.type,
        "document_date": doc.date,
    }


def search_documents(query: str) -> list[KnowledgeDocument]:
    """
    Reuses the existing lexical search infrastructure (the same
    search_vector/GIN index src/retrieval/lexical.py already relies on)
    — no second search system.
    """
    if not query.strip():
        return list_documents()

    sql = """
        SELECT
            document_id,
            MAX(source_file) AS source_file,
            MIN(page_start) AS min_page,
            MAX(page_end) AS max_page,
            COUNT(*) AS chunk_count,
            (array_agg(text ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) DESC))[1] AS excerpt,
            MAX(ts_rank_cd(search_vector, websearch_to_tsquery('english', %s))) AS relevance
        FROM chunks
        WHERE search_vector @@ websearch_to_tsquery('english', %s)
        GROUP BY document_id
        ORDER BY relevance DESC
        LIMIT 30
    """

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (query, query, query))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    return [_row_to_document(row) for row in rows]


def document_exists(document_id: str) -> bool:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM chunks WHERE document_id = %s LIMIT 1", (document_id,))
        return cur.fetchone() is not None


def resolve_file_path(document_id: str) -> Path | None:
    """
    Only returns a path for a document_id that is actually registered in
    `chunks` — never accepts arbitrary filesystem paths from the caller.
    """
    if not document_exists(document_id):
        return None

    candidate = (RAW_DATA_DIR / document_id).with_suffix(".pdf").resolve()
    raw_root = RAW_DATA_DIR.resolve()

    if raw_root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate

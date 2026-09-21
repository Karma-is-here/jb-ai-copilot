import uuid
from datetime import datetime, timezone


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_conversation_title(question: str) -> str:
    """LLM-free title heuristic — mirrors lib/deriveTitle.ts on the frontend."""
    title = question.strip().rstrip("?").strip()
    if not title:
        return "New conversation"
    title = title[0].upper() + title[1:]
    if len(title) > 60:
        title = title[:57].rstrip() + "..."
    return title

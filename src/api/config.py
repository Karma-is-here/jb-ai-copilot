import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Existing env vars (already required by src/retrieval and src/generation —
# read here only, never redefined or given different meaning).
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "jb_copilot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "jb_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# New, API-layer-only env vars.
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

# Repo root is three levels up from this file (src/api/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
UPLOADS_DIR = REPO_ROOT / "data" / "uploads"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def postgres_dsn_kwargs() -> dict:
    return {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "dbname": POSTGRES_DB,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
    }

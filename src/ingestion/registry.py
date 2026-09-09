from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_metadata(file_path: Path) -> dict:
    """Build immutable source metadata for a document."""

    relative_path = file_path.relative_to(RAW_DIR)

    stat = file_path.stat()

    return {
        "document_id": relative_path.with_suffix("").as_posix(),
        "source_path": relative_path.as_posix(),
        "filename": file_path.name,

        "category": (
            relative_path.parts[0]
            if len(relative_path.parts) >= 1
            else None
        ),

        "subcategory": (
            relative_path.parts[1]
            if len(relative_path.parts) >= 2
            else None
        ),

        "created_at": datetime.fromtimestamp(
            stat.st_ctime,
            tz=timezone.utc,
        ).isoformat(),

        "modified_at": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),

        "sha256": calculate_sha256(file_path),
    }


def load_manifest() -> dict:
    """Load the document manifest."""

    if not MANIFEST_PATH.exists():
        return {
            "version": 1,
            "documents": [],
        }

    if MANIFEST_PATH.stat().st_size == 0:
        return {
            "version": 1,
            "documents": [],
        }

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(manifest: dict) -> None:
    """Save the document manifest."""

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


def register_documents() -> None:
    """
    Discover all PDFs under data/raw/ and register
    their source metadata in data/manifest.json.

    This function does NOT perform extraction,
    normalization, chunking, embedding, etc.
    """

    manifest = load_manifest()

    documents = manifest["documents"]

    existing_documents = {
        document["document_id"]: document
        for document in documents
    }

    pdf_files = sorted(
        RAW_DIR.rglob("*.pdf")
    )

    for pdf_path in pdf_files:

        metadata = get_file_metadata(pdf_path)

        document_id = metadata["document_id"]

        if document_id not in existing_documents:

            metadata["extraction"] = {
                "status": "pending",
                "processed_at": None,
                "output": None,
                "error": None,
            }

            metadata["normalization"] = {
                "status": "pending",
                "processed_at": None,
                "output": None,
                "error": None,
            }

            documents.append(metadata)

            print(
                f"[NEW] {metadata['source_path']}"
            )

        else:

            existing = existing_documents[document_id]

            if existing["sha256"] != metadata["sha256"]:

                existing.update(
                    {
                        "modified_at": metadata["modified_at"],
                        "sha256": metadata["sha256"],
                    }
                )

                print(
                    f"[CHANGED] {metadata['source_path']}"
                )

            else:

                print(
                    f"[UNCHANGED] {metadata['source_path']}"
                )

    manifest["documents"] = sorted(
        documents,
        key=lambda document: document["modified_at"],
        reverse=True,
    )

    save_manifest(manifest)

    print(
        f"\nRegistered {len(manifest['documents'])} documents."
    )


if __name__ == "__main__":
    register_documents()
from pathlib import Path
import json
from datetime import datetime, timezone

import pymupdf
import pytesseract
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/manifest.json")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(manifest: dict) -> None:
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


def update_extraction_status(
    document: dict,
    status: str,
    processed_at: str | None = None,
    output: str | None = None,
    error: str | None = None,
) -> None:
    """
    Update ONLY the extraction section of a document.

    This function must not modify document identity
    or source metadata.
    """

    document["extraction"] = {
        "status": status,
        "processed_at": processed_at,
        "output": output,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_page_text(page) -> tuple[str, str]:
    """
    Try native PDF text extraction first.

    Fall back to OCR if insufficient text is found.

    Returns:
        (text, extraction_method)
    """

    text = page.get_text("text").strip()

    if len(text) >= 100:
        return text, "pymupdf"

    pix = page.get_pixmap(
        dpi=300,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    ocr_text = pytesseract.image_to_string(
        image,
        lang="eng",
    ).strip()

    return ocr_text, "ocr"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> Path:
    """
    Extract one PDF page-by-page into JSONL.

    The directory structure under data/raw/
    is preserved under data/processed/.
    """

    pdf_path = pdf_path.resolve()
    raw_dir = RAW_DIR.resolve()

    try:
        relative_path = pdf_path.relative_to(raw_dir)
    except ValueError as exc:
        raise ValueError(
            f"PDF must be located inside {RAW_DIR}: {pdf_path}"
        ) from exc

    doc = pymupdf.open(pdf_path)

    page_count = len(doc)

    output_path = (
        PROCESSED_DIR
        / relative_path.parent
        / f"{relative_path.stem}.jsonl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nFile: {pdf_path}")
    print(f"Pages: {page_count}")
    print(f"Output: {output_path}")
    print("=" * 80)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for page_number, page in enumerate(
            doc,
            start=1,
        ):

            text, method = extract_page_text(page)

            record = {
                "document_id": relative_path.with_suffix("").as_posix(),
                "source_file": pdf_path.name,
                "source_path": relative_path.as_posix(),

                "page_number": page_number,
                "page_count": page_count,

                "extraction_method": method,

                "text": text,
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(
                f"Page {page_number}: "
                f"{method} | "
                f"{len(text)} characters"
            )

    doc.close()

    print(
        f"\nExtraction complete: {output_path}"
    )

    return output_path


# ---------------------------------------------------------------------------
# Process documents from manifest
# ---------------------------------------------------------------------------

def process_pending_documents() -> None:
    """
    Process documents whose extraction is pending or failed.

    This function modifies ONLY extraction state.
    """

    manifest = load_manifest()

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for document in manifest["documents"]:

        extraction = document.get(
            "extraction",
            {},
        )

        status = extraction.get(
            "status"
        )

        # Only process documents requiring extraction.
        if status not in {
            "pending",
            "failed",
        }:
            skipped_count += 1

            print(
                f"[SKIP] {document['source_path']}"
            )

            continue

        pdf_path = (
            RAW_DIR
            / document["source_path"]
        )

        if not pdf_path.exists():

            update_extraction_status(
                document=document,
                status="failed",
                processed_at=None,
                output=None,
                error=(
                    f"Source file not found: "
                    f"{pdf_path}"
                ),
            )

            failed_count += 1

            print(
                f"[ERROR] {document['source_path']} "
                f"source file not found"
            )

            continue

        try:

            output_path = extract_pdf(
                pdf_path
            )

            processed_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            relative_output = (
                output_path
                .relative_to(
                    PROCESSED_DIR
                )
                .as_posix()
            )

            update_extraction_status(
                document=document,
                status="success",
                processed_at=processed_at,
                output=relative_output,
                error=None,
            )

            processed_count += 1

            # Save after every successful document.
            save_manifest(manifest)

        except Exception as exc:

            update_extraction_status(
                document=document,
                status="failed",
                processed_at=None,
                output=None,
                error=str(exc),
            )

            failed_count += 1

            save_manifest(manifest)

            print(
                f"[ERROR] {document['source_path']}: "
                f"{exc}"
            )

    save_manifest(manifest)

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Processed : {processed_count}")
    print(f"Skipped   : {skipped_count}")
    print(f"Failed    : {failed_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_pending_documents()
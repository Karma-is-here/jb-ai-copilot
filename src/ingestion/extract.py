from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_page_text(page) -> tuple[str, str]:
    """
    Try native PDF text extraction first.
    Fall back to OCR if insufficient text is found.

    Returns:
        (text, extraction_method)
    """

    # Step 1: Try native PDF text extraction
    text = page.get_text("text").strip()

    # Step 2: Decide whether the extracted text is sufficient
    if len(text) >= 100:
        return text, "pymupdf"

    # Step 3: Fall back to OCR
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


def extract_pdf(pdf_path: Path) -> None:
    doc = pymupdf.open(pdf_path)

    print(f"File: {pdf_path}")
    print(f"Pages: {len(doc)}")
    print("=" * 80)

    for page_number, page in enumerate(doc, start=1):

        text, method = extract_page_text(page)

        print(f"\n--- PAGE {page_number} ---")
        print(f"Method: {method}")
        print(f"Characters: {len(text)}")
        print(text[:2000])

    doc.close()


if __name__ == "__main__":
    PDF_PATH = Path(
        "data/raw/investment/discretionary/discretionary-mandates.pdf"
    )

    extract_pdf(PDF_PATH)
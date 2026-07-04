"""
PDF Extractor
Extracts text from PDFs. Uses native text extraction where possible,
and falls back to OCR for scanned/image-only pages.
"""

import os
import io
import json
import re
from pathlib import Path
from collections import Counter

import fitz  # this is PyMuPDF
import pytesseract
from PIL import Image
from langdetect import detect, LangDetectException
from tqdm import tqdm

# --- CONFIG ---
# Point pytesseract to the Tesseract executable we installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RAW_PDF_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "extracted_pages.json"

MIN_NATIVE_TEXT_LENGTH = 20  # below this, we assume the page is scanned and needs OCR
OCR_DPI = 300  # higher = better OCR accuracy but slower


def extract_native_text(page):
    """Try to pull text directly from the PDF (fast, works for normal text pages)."""
    return page.get_text("text").strip()


def extract_text_via_ocr(page):
    """Render the page as an image and run OCR on it (for scanned pages)."""
    pix = page.get_pixmap(dpi=OCR_DPI)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()


def detect_language(text):
    """Return a language code like 'en', or 'unknown' if it can't be detected."""
    if not text or len(text.strip()) < 10:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def remove_repeated_headers_footers(pages):
    """
    Detects lines that repeat across most pages of the SAME pdf (typical of
    headers/footers, like a document title on every page) and strips them out.
    """
    line_counts = Counter()
    for page in pages:
        lines = [line.strip() for line in page["text"].split("\n") if line.strip()]
        # only look at first 2 and last 2 lines of each page (headers/footers live there)
        candidate_lines = lines[:2] + lines[-2:]
        line_counts.update(set(candidate_lines))

    total_pages = len(pages)
    if total_pages == 0:
        return pages

    # a line appearing on more than 50% of pages is very likely a header/footer
    repeated_lines = {
        line for line, count in line_counts.items()
        if count / total_pages > 0.5 and len(line) < 120
    }

    for page in pages:
        cleaned_lines = [
            line for line in page["text"].split("\n")
            if line.strip() not in repeated_lines
        ]
        page["text"] = "\n".join(cleaned_lines).strip()

    return pages


def process_pdf(pdf_path: Path):
    """Extract all pages from one PDF, using OCR as a fallback where needed."""
    doc = fitz.open(pdf_path)
    pages = []

    for page_number in tqdm(range(len(doc)), desc=f"  Pages in {pdf_path.name}", leave=False):
        page = doc.load_page(page_number)
        native_text = extract_native_text(page)

        if len(native_text) >= MIN_NATIVE_TEXT_LENGTH:
            text = native_text
            source = "native"
        else:
            text = extract_text_via_ocr(page)
            source = "ocr"

        pages.append({
            "pdf_id": pdf_path.stem,       # filename without extension, used as an ID
            "filename": pdf_path.name,
            "page_number": page_number + 1,  # 1-indexed for humans
            "text": text,
            "source": source,
        })

    doc.close()

    # clean repeated headers/footers across this PDF's pages
    pages = remove_repeated_headers_footers(pages)

    # detect language now that text is cleaned
    for page in tqdm(pages, desc=f"  Detecting language", leave=False):
        page["language"] = detect_language(page["text"])

    return pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_PDF_DIR}. Add some PDFs there first.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    all_pages = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        pages = process_pdf(pdf_path)
        all_pages.extend(pages)
        native_count = sum(1 for p in pages if p["source"] == "native")
        ocr_count = sum(1 for p in pages if p["source"] == "ocr")
        print(f"  -> {pdf_path.name}: {len(pages)} pages ({native_count} native, {ocr_count} OCR)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Extracted {len(all_pages)} total pages -> saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
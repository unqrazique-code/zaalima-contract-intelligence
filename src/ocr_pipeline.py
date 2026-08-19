"""
Week 1, Day 3-5: OCR Pipeline
Handles raw PDF contract ingestion. CUAD ships pre-extracted text, but
real-world contracts arrive as PDFs -- sometimes with a text layer,
sometimes scanned images with none. This pipeline handles both:

1. First tries direct text extraction (fast, exact) via pdfplumber.
2. Falls back to OCR (Tesseract via pdf2image + pytesseract) for pages
   with no extractable text (scanned/image-only pages).

Usage:
    python src/ocr_pipeline.py path/to/contract.pdf
    python src/ocr_pipeline.py path/to/contract.pdf --out data/processed/ocr_output.json
"""

import argparse
import json
import os

import pdfplumber
from pdf2image import convert_from_path
import pytesseract


def extract_text_native(pdf_path: str) -> list[dict]:
    """Try direct text extraction per page. Returns text found + which pages
    had none (candidates for OCR)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text.strip(),
                "source": "native" if text.strip() else None,
            })
    return pages


def ocr_page(pdf_path: str, page_number: int) -> str:
    """Rasterize a single PDF page to an image and run Tesseract OCR on it."""
    images = convert_from_path(
        pdf_path, first_page=page_number, last_page=page_number, dpi=300
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0]).strip()


def process_pdf(pdf_path: str) -> dict:
    """Full pipeline: native extraction first, OCR fallback per page."""
    pages = extract_text_native(pdf_path)

    ocr_page_count = 0
    for page in pages:
        if not page["text"]:
            page["text"] = ocr_page(pdf_path, page["page_number"])
            page["source"] = "ocr"
            ocr_page_count += 1

    full_text = "\n\n".join(p["text"] for p in pages if p["text"])

    return {
        "file": os.path.basename(pdf_path),
        "total_pages": len(pages),
        "pages_requiring_ocr": ocr_page_count,
        "pages": pages,
        "full_text": full_text,
    }


def main():
    parser = argparse.ArgumentParser(description="OCR pipeline for contract PDFs")
    parser.add_argument("pdf_path", help="Path to the input PDF contract")
    parser.add_argument("--out", default=None, help="Optional path to save JSON output")
    args = parser.parse_args()

    result = process_pdf(args.pdf_path)

    print(f"File: {result['file']}")
    print(f"Total pages: {result['total_pages']}")
    print(f"Pages that needed OCR: {result['pages_requiring_ocr']}")
    print(f"Extracted text length: {len(result['full_text'])} chars")
    print("\n--- Preview (first 500 chars) ---")
    print(result["full_text"][:500])

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved full output to: {args.out}")


if __name__ == "__main__":
    main()

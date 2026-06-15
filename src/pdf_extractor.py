"""
src/pdf_extractor.py — PDF Text Extraction Module
===================================================
Responsibilities:
  - Scan the `data/` folder for all PDF files
  - Extract text from every page using pdfplumber
  - Skip blank or near-empty pages (configurable threshold)
  - Attach metadata (source filename, page number) to every page
  - Save raw page-level output as JSON inside `extracted/`

Why pdfplumber?
  pdfplumber gives fine-grained control over text extraction and
  handles tables, columns, and complex layouts better than PyPDF2.
"""

import os
import json
import logging
import pdfplumber

# Import project-level configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, EXTRACTED_DIR, SUPPORTED_EXTENSIONS, MIN_PAGE_CHARS

# ── Logging Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_pdf_files(data_dir: str = DATA_DIR) -> list[str]:
    """
    Scan the data directory and return a list of absolute paths
    for all supported PDF files.

    Args:
        data_dir: Path to the folder containing PDFs.

    Returns:
        Sorted list of absolute file paths.
    """
    pdf_files = []
    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            pdf_files.append(os.path.join(data_dir, filename))

    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{data_dir}'")
    return pdf_files


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a single PDF file.

    Each page is represented as a dictionary:
        {
            "source":   "_Company_Policy_Part_A.pdf",  # filename only
            "page":     1,                                   # 1-indexed
            "content":  "Extracted text of the page..."
        }

    Pages with fewer than MIN_PAGE_CHARS characters are skipped
    (e.g., cover pages, image-only pages, dividers).

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of page dictionaries with content and metadata.
    """
    pages = []
    filename = os.path.basename(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  Processing '{filename}' — {total_pages} page(s)")

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    # extract_text() returns None for image-only pages
                    raw_text = page.extract_text()

                    if raw_text is None:
                        logger.debug(f"    Page {page_num}: no text layer — skipping")
                        continue

                    # Skip pages that are too short to be meaningful
                    if len(raw_text.strip()) < MIN_PAGE_CHARS:
                        logger.debug(f"    Page {page_num}: too short ({len(raw_text.strip())} chars) — skipping")
                        continue

                    pages.append({
                        "source":  filename,
                        "page":    page_num,
                        "content": raw_text
                    })

                except Exception as page_err:
                    # A single bad page should not stop the whole document
                    logger.warning(f"    Page {page_num} in '{filename}' caused an error: {page_err}")

    except Exception as doc_err:
        logger.error(f"  Failed to open '{filename}': {doc_err}")

    logger.info(f"  ✔ Extracted {len(pages)} valid page(s) from '{filename}'")
    return pages


def save_extracted_pages(pages: list[dict], filename: str) -> str:
    """
    Save the extracted pages list as a JSON file inside `extracted/`.

    The output filename mirrors the source PDF name:
        e.g., `_Policy_Part_A.pdf` → `extracted/_Policy_Part_A.json`

    Args:
        pages:    List of page dicts returned by extract_text_from_pdf().
        filename: Original PDF filename (used to derive the output name).

    Returns:
        Absolute path to the saved JSON file.
    """
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # Replace .pdf extension with .json
    base_name  = os.path.splitext(filename)[0]
    output_path = os.path.join(EXTRACTED_DIR, f"{base_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    logger.info(f"  💾 Saved → {output_path}")
    return output_path


def extract_all_pdfs(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Main entry point for the extraction stage.

    Processes every PDF in `data_dir`, saves per-file JSON to `extracted/`,
    and returns all pages as a flat list for the next pipeline stage.

    Args:
        data_dir: Folder containing HR policy PDFs.

    Returns:
        Flat list of all extracted page dicts across all documents.
    """
    pdf_files  = get_pdf_files(data_dir)
    all_pages  = []

    if not pdf_files:
        logger.warning("No PDF files found. Check your DATA_DIR in config.py.")
        return all_pages

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        pages    = extract_text_from_pdf(pdf_path)

        if pages:
            save_extracted_pages(pages, filename)
            all_pages.extend(pages)

    logger.info(f"\n📄 Extraction complete — {len(all_pages)} total pages across {len(pdf_files)} file(s)")
    return all_pages

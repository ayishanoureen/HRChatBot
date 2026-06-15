"""
src/text_cleaner.py — Text Cleaning & Normalization Module
============================================================
Responsibilities:
  - Receive raw page text extracted by pdf_extractor.py
  - Remove noise: extra whitespace, control characters, repeated symbols
  - Normalize unicode and punctuation
  - Preserve meaningful structure (headings, bullet points, numbered lists)
  - Return clean text that is RAG-friendly

Why clean before chunking?
  Noisy text (extra spaces, garbled characters) degrades embedding quality
  and makes retrieved passages harder for an LLM to interpret. Cleaning
  first produces more semantically coherent chunks.
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)


# ── Individual Cleaning Steps ──────────────────────────────────────────────────

def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters (e.g., form feeds \\f, null bytes).
    Keeps standard whitespace: spaces, tabs, newlines.
    """
    # Replace form-feed (\\f) and carriage-return (\\r) with newline
    text = text.replace("\f", "\n").replace("\r", "\n")

    # Remove any remaining non-printable characters except \\n and \\t
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )
    return cleaned


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode to NFC form and replace common look-alike characters.
    e.g., curly quotes → straight quotes, em-dashes → hyphens
    """
    text = unicodedata.normalize("NFC", text)

    replacements = {
        "\u2018": "'",   # left single quotation mark
        "\u2019": "'",   # right single quotation mark
        "\u201c": '"',   # left double quotation mark
        "\u201d": '"',   # right double quotation mark
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2022": "-",   # bullet •
        "\u00a0": " ",   # non-breaking space
        "\u2026": "...", # ellipsis
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    return text


def collapse_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive spaces/tabs into a single space.
    Preserve intentional paragraph breaks (double newlines).
    """
    # Collapse multiple spaces/tabs on the same line
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines into exactly 2 (one paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces at the start/end of each line
    lines = [line.strip() for line in text.split("\n")]
    text  = "\n".join(lines)

    return text.strip()


def remove_repeated_symbols(text: str) -> str:
    """
    Remove lines that consist entirely of repeated symbols:
    e.g., '==========', '----------', '...........'
    These are often decorative dividers in PDFs that add no semantic value.
    """
    # Match lines with 4+ repeated non-alphanumeric characters
    text = re.sub(r"^([^\w\s])\1{3,}$", "", text, flags=re.MULTILINE)
    return text


def fix_broken_words(text: str) -> str:
    """
    PDF extraction sometimes introduces hyphens at line breaks where a word
    wraps across lines (e.g., 'organ-\\nization' → 'organization').
    This step joins those broken words.
    """
    # e.g., "organ-\nization" → "organization"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    return text


# ── Master Cleaning Function ───────────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Apply the full cleaning pipeline to a single page's raw text.

    Pipeline order:
        1. Remove control characters
        2. Normalize unicode
        3. Fix broken hyphenated words
        4. Remove repeated symbol lines
        5. Collapse whitespace

    Args:
        raw_text: Raw string extracted from a PDF page.

    Returns:
        Cleaned, normalized string ready for chunking.
    """
    text = raw_text

    text = remove_control_characters(text)
    text = normalize_unicode(text)
    text = fix_broken_words(text)
    text = remove_repeated_symbols(text)
    text = collapse_whitespace(text)

    return text


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Apply clean_text() to every page in the extracted pages list.

    Input format (from pdf_extractor):
        [
            {"source": "file.pdf", "page": 1, "content": "raw text..."},
            ...
        ]

    Output format (same structure, content replaced with cleaned text):
        [
            {"source": "file.pdf", "page": 1, "content": "clean text..."},
            ...
        ]

    Pages whose content is empty after cleaning are dropped.

    Args:
        pages: List of raw page dicts from the extraction stage.

    Returns:
        List of cleaned page dicts (empty-content pages removed).
    """
    cleaned_pages = []
    skipped       = 0

    for page in pages:
        cleaned_content = clean_text(page["content"])

        if not cleaned_content:
            logger.debug(f"  Dropped empty page after cleaning: {page['source']} p.{page['page']}")
            skipped += 1
            continue

        cleaned_pages.append({
            "source":  page["source"],
            "page":    page["page"],
            "content": cleaned_content
        })

    logger.info(f"🧹 Cleaning complete — {len(cleaned_pages)} pages kept, {skipped} dropped")
    return cleaned_pages

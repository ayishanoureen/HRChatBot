"""
src/text_chunker.py — Text Chunking Module
===========================================
Responsibilities:
  - Receive cleaned page-level text from text_cleaner.py
  - Split text into smaller, overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter
  - Preserve rich metadata on every chunk:
      • source filename
      • original page number
      • chunk index (unique per document)
      • chunk_id (globally unique string key)
  - Save final chunks as JSON to `processed/all_chunks.json`

Why RecursiveCharacterTextSplitter?
  It tries to split on natural boundaries (paragraphs → sentences → words)
  before resorting to hard character splits. This keeps chunks semantically
  coherent — crucial for accurate RAG retrieval.
"""

import os
import json
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR,
    ALL_CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)

logger = logging.getLogger(__name__)


# ── Splitter Initialization ────────────────────────────────────────────────────

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create and return a configured RecursiveCharacterTextSplitter.

    Settings come from config.py so they can be tuned without touching
    this module:
        CHUNK_SIZE       — max characters per chunk
        CHUNK_OVERLAP    — characters shared between adjacent chunks
        CHUNK_SEPARATORS — ordered list of split boundaries to try
    """
    return RecursiveCharacterTextSplitter(
        chunk_size        = CHUNK_SIZE,
        chunk_overlap     = CHUNK_OVERLAP,
        separators        = CHUNK_SEPARATORS,
        length_function   = len,
        is_separator_regex = False,
    )


# ── Core Chunking Logic ────────────────────────────────────────────────────────

def chunk_pages(cleaned_pages: list[dict]) -> list[dict]:
    """
    Split every cleaned page into overlapping text chunks.

    For each page, the splitter may produce 1-N chunks. Each chunk gets:
        {
            "chunk_id":    "_Policy_Part_A__p3_c0",  # globally unique
            "source":      "_Company_Policy_Part_A.pdf",
            "page":        3,
            "chunk_index": 0,           # 0-based index within this document
            "content":     "chunk text..."
        }

    Args:
        cleaned_pages: List of cleaned page dicts from text_cleaner.py.

    Returns:
        Flat list of all chunk dicts across all documents.
    """
    splitter   = build_splitter()
    all_chunks = []

    # Track per-document chunk index for unique IDs
    doc_chunk_counters: dict[str, int] = {}

    for page in cleaned_pages:
        source  = page["source"]
        page_no = page["page"]
        content = page["content"]

        # Split this page's text into chunks
        raw_chunks = splitter.split_text(content)

        for raw_chunk in raw_chunks:
            # Skip chunks that are just whitespace
            stripped = raw_chunk.strip()
            if not stripped:
                continue

            # Increment per-document counter
            doc_chunk_counters[source] = doc_chunk_counters.get(source, 0)
            chunk_index = doc_chunk_counters[source]
            doc_chunk_counters[source] += 1

            # Build a globally unique chunk ID
            base_name = os.path.splitext(source)[0]
            chunk_id  = f"{base_name}__p{page_no}_c{chunk_index}"

            all_chunks.append({
                "chunk_id":    chunk_id,
                "source":      source,
                "page":        page_no,
                "chunk_index": chunk_index,
                "content":     stripped,
            })

    logger.info(f"✂️  Chunking complete — {len(all_chunks)} chunks from {len(cleaned_pages)} pages")
    return all_chunks


# ── Save Output ────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: str = ALL_CHUNKS_FILE) -> str:
    """
    Persist the final chunk list as a JSON file.

    The file is saved to `processed/all_chunks.json` by default (see config.py).
    This JSON is the direct input for Phase 2: embedding + vector store ingestion.

    Args:
        chunks:      List of chunk dicts produced by chunk_pages().
        output_path: Target file path (defaults to ALL_CHUNKS_FILE in config).

    Returns:
        Absolute path to the saved JSON file.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"💾 Saved {len(chunks)} chunks → {output_path} ({size_kb:.1f} KB)")
    return output_path


# ── Convenience Wrapper ────────────────────────────────────────────────────────

def chunk_and_save(cleaned_pages: list[dict]) -> list[dict]:
    """
    One-call wrapper: chunk all pages then save to disk.

    Args:
        cleaned_pages: Output of text_cleaner.clean_pages().

    Returns:
        The list of chunk dicts (also saved to disk as a side-effect).
    """
    chunks = chunk_pages(cleaned_pages)
    save_chunks(chunks)
    return chunks

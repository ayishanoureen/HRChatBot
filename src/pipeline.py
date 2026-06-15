"""
src/pipeline.py — Phase 1 Orchestration
=========================================
This module wires together the three stages of preprocessing:

    Stage 1 — Extract:  pdf_extractor  → raw page dicts
    Stage 2 — Clean:    text_cleaner   → clean page dicts
    Stage 3 — Chunk:    text_chunker   → final chunk dicts + saved JSON

It also provides structured summary statistics so you can verify
the pipeline ran correctly without opening any output files manually.
"""

import logging
import time
from collections import defaultdict

from src.pdf_extractor import extract_all_pdfs
from src.text_cleaner  import clean_pages
from src.text_chunker  import chunk_and_save

logger = logging.getLogger(__name__)


# ── Stats Helper ───────────────────────────────────────────────────────────────

def compute_stats(raw_pages: list[dict], clean_pages_list: list[dict], chunks: list[dict]) -> dict:
    """
    Build a summary dictionary from all three pipeline stages.

    Returns a dict with counts per file and totals, useful for
    verifying correctness and debugging.
    """
    # Pages per source file (after cleaning)
    pages_per_file: dict[str, int] = defaultdict(int)
    for page in clean_pages_list:
        pages_per_file[page["source"]] += 1

    # Chunks per source file
    chunks_per_file: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        chunks_per_file[chunk["source"]] += 1

    return {
        "total_pdfs":         len(set(p["source"] for p in raw_pages)),
        "total_raw_pages":    len(raw_pages),
        "total_clean_pages":  len(clean_pages_list),
        "pages_dropped":      len(raw_pages) - len(clean_pages_list),
        "total_chunks":       len(chunks),
        "pages_per_file":     dict(pages_per_file),
        "chunks_per_file":    dict(chunks_per_file),
    }


def print_summary(stats: dict, elapsed: float) -> None:
    """
    Print a nicely formatted pipeline summary to the terminal.
    """
    sep = "─" * 60

    print(f"\n{sep}")
    print("  📊  PIPELINE SUMMARY")
    print(sep)
    print(f"  PDFs processed       : {stats['total_pdfs']}")
    print(f"  Raw pages extracted  : {stats['total_raw_pages']}")
    print(f"  Pages after cleaning : {stats['total_clean_pages']}  (dropped {stats['pages_dropped']})")
    print(f"  Total chunks created : {stats['total_chunks']}")
    print(f"  Time elapsed         : {elapsed:.1f}s")
    print(sep)
    print("  Chunks per document:")

    for source, count in sorted(stats["chunks_per_file"].items()):
        pages = stats["pages_per_file"].get(source, 0)
        print(f"    • {source}")
        print(f"        pages={pages:<4}  chunks={count}")

    print(sep)
    print("  ✅ Phase 1 complete!  Output → processed/all_chunks.json")
    print(f"{sep}\n")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline() -> list[dict]:
    """
    Execute the full Phase 1 preprocessing pipeline.

    Returns:
        The final list of chunk dicts (already saved to disk).
    """
    start = time.time()

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STAGE 1 — PDF Extraction")
    logger.info("=" * 55)
    raw_pages = extract_all_pdfs()

    if not raw_pages:
        logger.error("No pages extracted. Check that PDFs are in the data/ folder.")
        return []

    # ── Stage 2: Clean ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("STAGE 2 — Text Cleaning")
    logger.info("=" * 55)
    cleaned = clean_pages(raw_pages)

    if not cleaned:
        logger.error("All pages were dropped during cleaning. Review MIN_PAGE_CHARS in config.py.")
        return []

    # ── Stage 3: Chunk ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("STAGE 3 — Chunking & Saving")
    logger.info("=" * 55)
    chunks = chunk_and_save(cleaned)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    stats   = compute_stats(raw_pages, cleaned, chunks)
    print_summary(stats, elapsed)

    return chunks

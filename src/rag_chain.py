"""
src/rag_chain.py — RAG Orchestration Layer
===========================================
Responsibilities:
  - Tie together retrieval + LLM generation into a single pipeline call
  - Accept a user query
  - Retrieve top-k relevant chunks from ChromaDB (via retriever.py)
  - Pass chunks + query to the LLM (via llm.py)
  - Return a structured response with the final answer AND source citations

This is the single module that Streamlit (Phase 4) and main.py (CLI)
both call — neither needs to know about retrieval or LLM internals.

RAG Flow:
    query → retriever.retrieve() → context_chunks
          → llm.generate_answer() → answer string
          → RAGResponse (answer + sources)
"""

import os
import sys
import logging
from typing import Optional
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RETRIEVAL_TOP_K, RETRIEVAL_MIN_SIMILARITY

from src.retriever import retrieve
from src.llm       import generate_answer

logger = logging.getLogger(__name__)


# ── Response Data Model ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Structured response returned by the RAG chain for every query.

    Attributes:
        query:   The original user question (unchanged).
        answer:  The LLM-generated natural language answer.
        sources: List of source citation dicts, each containing:
                     source (filename), page, similarity score.
        chunks:  The raw retrieved chunk dicts (for debugging / Streamlit expanders).
    """
    query:   str
    answer:  str
    sources: list[dict]         = field(default_factory=list)
    chunks:  list[dict]         = field(default_factory=list)

    def has_answer(self) -> bool:
        """Return True if a non-empty answer was generated."""
        return bool(self.answer and self.answer.strip())

    def format_sources(self) -> str:
        """Return a human-readable source citation string."""
        if not self.sources:
            return "No sources found."
        lines = []
        for i, src in enumerate(self.sources, 1):
            lines.append(
                f"  [{i}] {src['source']}  —  page {src['page']}  "
                f"(relevance: {src['similarity']:.0%})"
            )
        return "\n".join(lines)


# ── Source Extraction ──────────────────────────────────────────────────────────

def _extract_sources(chunks: list[dict]) -> list[dict]:
    """
    Deduplicate and extract source citations from retrieved chunks.

    Multiple chunks from the same page are collapsed into a single citation.
    Sources are sorted by similarity (highest first).

    Args:
        chunks: List of retriever result dicts.

    Returns:
        Deduplicated list of source citation dicts.
    """
    seen    = set()
    sources = []

    for chunk in chunks:
        key = (chunk.get("source"), chunk.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source":     chunk.get("source", "Unknown"),
                "page":       chunk.get("page", "?"),
                "similarity": chunk.get("similarity", 0.0),
            })

    # Sort by similarity descending
    return sorted(sources, key=lambda x: x["similarity"], reverse=True)


# ── Core RAG Chain ─────────────────────────────────────────────────────────────

def ask(
    query:      str,
    top_k:      int = RETRIEVAL_TOP_K,
    collection  = None,
) -> RAGResponse:
    """
    Execute the full RAG pipeline for a single user query.

    Steps:
        1. Retrieve top-k semantically similar chunks from ChromaDB
        2. Build a structured prompt with the retrieved context
        3. Call the LLM to generate a natural-language answer
        4. Return a RAGResponse with the answer + source citations

    Args:
        query:      The user's natural-language HR policy question.
        top_k:      Number of chunks to retrieve (default from config).
        collection: Pre-loaded ChromaDB collection (optional, for efficiency).

    Returns:
        RAGResponse dataclass with answer, sources, and raw chunks.
    """
    query = query.strip()
    if not query:
        return RAGResponse(
            query  = query,
            answer = "Please enter a question.",
        )

    logger.info(f"\n{'─'*60}")
    logger.info(f"💬 Processing query: \"{query}\"")

    # Step 1: Semantic retrieval
    logger.info("[Step 1] Retrieving relevant chunks from vector store...")
    chunks = retrieve(query, top_k=top_k, collection=collection)

    if not chunks:
        logger.warning("No chunks retrieved — vector store may be empty.")
        return RAGResponse(
            query  = query,
            answer = (
                "I wasn't able to find relevant information on that topic "
                "in the HR policy documents. Please contact HR directly."
            ),
        )

    # Compute confidence from the best semantic hit (neighbors have sim=0.0,
    # so restrict to ranked hits only)
    top_sim = max(
        (c["similarity"] for c in chunks if not c.get("is_neighbor")),
        default=0.0,
    )
    low_confidence = top_sim < RETRIEVAL_MIN_SIMILARITY

    logger.info(
        f"  ↳ {len(chunks)} chunk(s) retrieved | top score: {top_sim:.4f} "
        f"| {'LOW confidence' if low_confidence else 'OK'}"
    )

    # Step 2: LLM generation
    logger.info("[Step 2] Generating answer with LLM...")
    answer = generate_answer(query, chunks, low_confidence=low_confidence)

    # Step 3: Build source list
    sources = _extract_sources(chunks)

    response = RAGResponse(
        query   = query,
        answer  = answer,
        sources = sources,
        chunks  = chunks,
    )

    logger.info("[Step 3] RAG response ready.")
    return response


# ── Convenience: Run Multiple Questions ───────────────────────────────────────

def ask_many(queries: list[str], top_k: int = RETRIEVAL_TOP_K) -> list[RAGResponse]:
    """
    Run the RAG pipeline for a batch of questions.

    Reuses the same ChromaDB collection across all queries for efficiency.

    Args:
        queries: List of question strings.
        top_k:   Chunks to retrieve per query.

    Returns:
        List of RAGResponse objects (same order as input).
    """
    from src.vector_store import get_or_create_collection
    collection = get_or_create_collection()

    responses = []
    for query in queries:
        response = ask(query, top_k=top_k, collection=collection)
        responses.append(response)

    return responses

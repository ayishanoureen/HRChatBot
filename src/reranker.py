"""
src/reranker.py — Cross-Encoder Reranking Module
=================================================
Responsibilities:
  - Lazy-load BAAI/bge-reranker-base as a singleton (loaded once per process)
  - Accept a query + list of candidate chunk dicts
  - Score every (query, chunk_text) pair with the cross-encoder
  - Return the top-k chunks sorted by rerank score, each annotated with
    a 'rerank_score' field for logging / debugging

Why a cross-encoder for reranking?
  - The bi-encoder (MiniLM) embeds query and chunk independently — fast but
    less precise.
  - The cross-encoder sees both together and produces a finer-grained
    relevance score.
  - Running it over only 15 candidates keeps latency acceptable on CPU.
"""

import os
import sys
import logging
from typing import Optional

from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANKER_MODEL, RERANKER_TOP_K

logger = logging.getLogger(__name__)


# ── Singleton ──────────────────────────────────────────────────────────────────

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """
    Return the cross-encoder model, loading it on first call.

    The model is kept in memory for the lifetime of the process so
    subsequent queries pay no reload cost.
    """
    global _reranker
    if _reranker is None:
        logger.info(f"⚙️  Loading cross-encoder reranker: '{RERANKER_MODEL}'")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("✔ Reranker ready")
    return _reranker


# ── Core Rerank Function ───────────────────────────────────────────────────────

def rerank(
    query:  str,
    chunks: list[dict],
    top_k:  int = RERANKER_TOP_K,
) -> list[dict]:
    """
    Rerank a pool of candidate chunks using the cross-encoder and return
    the top-k most relevant ones.

    Steps:
      1. Build (query, chunk_text) pairs for every candidate chunk
      2. Run cross-encoder.predict() to get raw relevance logits
      3. Sort descending by score, slice top-k
      4. Annotate each returned chunk with 'rerank_score' (float)

    Args:
        query:  The original user question.
        chunks: Candidate chunk dicts from retriever.retrieve()
                (includes both semantic hits and neighbor chunks).
        top_k:  Number of chunks to keep after reranking.

    Returns:
        List of at most top_k chunk dicts, sorted by rerank_score descending.
        Each dict gains a 'rerank_score' key (raw cross-encoder logit).
    """
    if not chunks:
        return chunks

    top_k = min(top_k, len(chunks))

    model = get_reranker()

    # Build input pairs — cross-encoder expects [query, passage] per row
    pairs = [[query, c["content"]] for c in chunks]

    # predict() returns a numpy array of float32 logits
    scores = model.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(scores, chunks),
        key     = lambda x: float(x[0]),
        reverse = True,
    )

    result = []
    for score, chunk in ranked[:top_k]:
        chunk = dict(chunk)                          # shallow copy — don't mutate original
        chunk["rerank_score"] = round(float(score), 4)
        result.append(chunk)

    logger.info(
        f"  ↳ Reranked {len(chunks)} → top {len(result)} chunks "
        f"| best score: {result[0]['rerank_score']:.4f}"
    )
    return result

"""
src/retriever.py — Semantic Similarity Search Module
======================================================
Responsibilities:
  - Accept a natural-language user query
  - Convert the query to a dense embedding (same model as ingestion)
  - Query ChromaDB for the top-k most similar chunks
  - Expand each matched chunk with its immediate neighbors (prev/next)
    from the same source document using chunk_index metadata
  - Return structured results with text, metadata, and similarity score
  - Provide a formatted console display for testing/debugging

How RAG retrieval works:
  1. User query  →  embed_single()  →  query vector
  2. query vector  →  ChromaDB.query()  →  top-k nearest chunk vectors
  3. For each hit, fetch chunk_index±1 from the same source (neighbor expansion)
  4. Merged, de-duplicated chunks are passed to an LLM in Phase 3 as context
"""

import os
import sys
import logging
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RETRIEVAL_TOP_K,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_MIN_SIMILARITY,
    RETRIEVAL_NEIGHBOR_SIMILARITY,
    RETRIEVAL_MAX_CHUNKS,
)

from src.embeddings   import embed_single
from src.vector_store import get_or_create_collection

logger = logging.getLogger(__name__)


# ── HR Domain Synonym Map ───────────────────────────────────────────────────────
# Maps short/informal terms employees use to the formal HR vocabulary
# actually present in the policy documents.
# Entries are lowercase; matching is case-insensitive.
_HR_SYNONYMS: dict[str, list[str]] = {
    # Attendance
    "late":          ["late attendance", "late coming", "tardiness"],
    "absent":        ["absence", "absenteeism", "leave without notice"],
    "wfh":           ["work from home", "remote work", "telecommuting"],
    # Compensation
    "salary":        ["payroll", "compensation", "remuneration", "pay"],
    "bonus":         ["incentive", "variable pay", "performance bonus"],
    "increment":     ["salary revision", "pay raise", "annual increment"],
    "reimbursement": ["expense claim", "expense reimbursement", "claim"],
    # Travel / Stay
    "accommodation": ["hotel stay", "travel accommodation", "lodging"],
    "travel":        ["business travel", "official travel", "tour"],
    "conveyance":    ["travel allowance", "transport allowance"],
    # Leave
    "leave":         ["leave policy", "annual leave", "paid leave"],
    "maternity":     ["maternity leave", "maternity benefit"],
    "sick":          ["sick leave", "medical leave", "illness leave"],
    "casual":        ["casual leave", "personal leave"],
    # Performance
    "appraisal":     ["performance appraisal", "performance review", "evaluation"],
    "kpi":           ["key performance indicator", "performance metrics"],
    # Conduct
    "termination":   ["termination", "dismissal", "separation", "resignation"],
    "misconduct":    ["misconduct", "disciplinary action", "violation"],
    "notice":        ["notice period", "resignation notice"],
    # Benefits
    "insurance":     ["health insurance", "medical insurance", "group insurance"],
    "pf":            ["provident fund", "PF", "EPF"],
    "gratuity":      ["gratuity", "end of service benefit"],
    # Training
    "training":      ["training and development", "learning", "skill development"],
    "induction":     ["induction program", "onboarding", "orientation"],
}


def _expand_query(query: str) -> str:
    """
    Enrich a short or informal query with HR-domain synonyms.

    Strategy: for each word in the query that has a known synonym mapping,
    append those synonyms to the query text.  The enriched string is embedded
    so the resulting vector sits closer to formal policy vocabulary.

    The original query is preserved at the front — it has the highest weight
    in the final embedding.

    Args:
        query: Raw user query.

    Returns:
        Enriched query string (original + appended synonyms).
    """
    q_lower  = query.lower()
    extras: list[str] = []

    for keyword, synonyms in _HR_SYNONYMS.items():
        # Match whole-word occurrence of the keyword
        if keyword in q_lower.split() or keyword in q_lower:
            for syn in synonyms:
                if syn.lower() not in q_lower:   # don't duplicate what's already there
                    extras.append(syn)

    if not extras:
        return query

    enriched = query + " " + " ".join(extras)
    logger.debug(f"  ↳ Query expanded: \"{enriched[:120]}{'...' if len(enriched)>120 else ''}\"")
    return enriched

# ── Module-level caches ────────────────────────────────────────────────────────
# Keep a reference to the default collection so callers that don't pass one
# (e.g. --ask, --query) don't re-open the database on every call.
_default_collection = None


# ── Result Data Structure ──────────────────────────────────────────────────────

def _build_result(rank: int, doc: str, meta: dict, distance: float) -> dict:
    """
    Construct a clean, standardised result dict from ChromaDB response fields.

    ChromaDB returns 'distance' for cosine space — for L2-normalized vectors
    with cosine metric:   similarity = 1 - distance

    Args:
        rank:     1-based rank of this result (1 = most similar)
        doc:      The chunk's text content
        meta:     Metadata dict from ChromaDB (source, page, chunk_index)
        distance: Raw ChromaDB distance value

    Returns:
        Dict with keys: rank, content, source, page, chunk_index, similarity
    """
    similarity = round(1.0 - distance, 4)   # convert distance → similarity score
    return {
        "rank":        rank,
        "content":     doc,
        "source":      meta.get("source", "unknown"),
        "page":        meta.get("page", -1),
        "chunk_index": meta.get("chunk_index", -1),
        "similarity":  similarity,
    }


# ── Neighbor Chunk Expansion ───────────────────────────────────────────────────

def _fetch_neighbors(
    collection,
    source:      str,
    chunk_index: int,
) -> list[dict]:
    """
    Fetch the immediately adjacent chunks (prev and next) for a given chunk.

    Uses ChromaDB's metadata filter to find chunks with:
        source == source  AND  chunk_index in {chunk_index-1, chunk_index+1}

    Args:
        collection:  The ChromaDB collection to query.
        source:      Source filename of the seed chunk.
        chunk_index: chunk_index of the seed chunk.

    Returns:
        List of 0–2 neighbor result dicts (marked with is_neighbor=True,
        similarity=None, rank=None).
    """
    neighbors = []
    for neighbor_idx in (chunk_index - 1, chunk_index + 1):
        if neighbor_idx < 0:
            continue
        try:
            result = collection.get(
                where   = {"$and": [{"source": {"$eq": source}},
                                    {"chunk_index": {"$eq": neighbor_idx}}]},
                include = ["documents", "metadatas"],
            )
        except Exception as exc:
            logger.debug(f"  Neighbor fetch skipped ({source} idx={neighbor_idx}): {exc}")
            continue

        if not result["ids"]:
            continue

        doc  = result["documents"][0]
        meta = result["metadatas"][0]
        neighbors.append({
            "rank":        None,          # not a ranked hit
            "content":     doc,
            "source":      meta.get("source", source),
            "page":        meta.get("page", -1),
            "chunk_index": meta.get("chunk_index", neighbor_idx),
            "similarity":  0.0,           # no semantic score; 0.0 keeps sorted() safe
            "is_neighbor": True,
        })

    return neighbors


# ── Core Retrieval Function ────────────────────────────────────────────────────

def retrieve(
    query:      str,
    top_k:      int = RETRIEVAL_TOP_K,
    collection  = None,
) -> list[dict]:
    """
    Perform semantic similarity search for a given query.

    Steps:
      1. Embed the query with the same model used during ingestion
      2. Query ChromaDB for the `top_k` most similar chunks
      3. Return structured results sorted by similarity (highest first)

    Args:
        query:      Natural-language question or keyword string.
        top_k:      Number of results to return (default from config).
        collection: ChromaDB collection (loaded if None).

    Returns:
        List of result dicts, sorted by descending similarity score.
    """
    if not query or not query.strip():
        logger.warning("Empty query received — returning no results")
        return []

    logger.info(f"🔍 Query: \"{query}\"")

    # Detect broad queries (very short or matching major policy headers)
    words = query.strip().split()
    broad_keywords = ["policy", "policies", "rules", "hr", "overview", "facilities", "general", "benefits"]
    is_broad = (len(words) <= 2) or any(w.lower() in broad_keywords for w in words)
    
    if is_broad:
        top_k = max(top_k, 15)  # Boost retrieval for high-level questions
        logger.info("  ↳ Broad query detected: boosting retrieval depth")

    # Step 1: Embed the user query (with synonym expansion for better recall)
    enriched_query = _expand_query(query.strip())
    query_vector   = embed_single(enriched_query)

    # Step 2: Load collection if not provided (cache for subsequent calls)
    if collection is None:
        global _default_collection
        if _default_collection is None:
            _default_collection = get_or_create_collection()
        collection = _default_collection

    if collection.count() == 0:
        logger.error("Vector store is empty. Run Phase 2 ingestion first.")
        return []

    # Step 3: Query ChromaDB
    response = collection.query(
        query_embeddings = [query_vector],
        n_results        = min(top_k, collection.count()),  # guard: can't exceed total docs
        include          = ["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one per query) — we sent one query
    docs      = response["documents"][0]
    metas     = response["metadatas"][0]
    distances = response["distances"][0]

    results = [
        _build_result(rank + 1, doc, meta, dist)
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances))
    ]

    logger.info(f"  ↳ Retrieved {len(results)} chunk(s) — top similarity: {results[0]['similarity'] if results else 'N/A'}")

    # Debug: log full content of each retrieved chunk so you can inspect quality
    for r in results:
        logger.debug(
            f"\n  [Chunk #{r['rank']} | sim={r['similarity']:.4f} | "
            f"{r['source']} p{r['page']}]\n  {r['content'][:300]}..."
        )

    # ── Similarity filtering ───────────────────────────────────────────────
    # Drop chunks below the minimum threshold.
    filtered = [r for r in results if r["similarity"] >= RETRIEVAL_MIN_SIMILARITY]
    dropped  = len(results) - len(filtered)
    if dropped:
        logger.info(f"  ↳ {dropped} chunk(s) dropped below similarity threshold ({RETRIEVAL_MIN_SIMILARITY})")

    # SAFE fallback: if nothing passes the threshold, keep the two best
    # chunks rather than just one, so short/informal queries still get
    # a meaningful answer.
    if not filtered:
        hits = results[:2]
        logger.info("  ↳ No chunks above threshold — using top-2 as fallback")
    else:
        hits = filtered

    # ── Neighbor Expansion ────────────────────────────────────────────────────
    # For each high-confidence seed, add ONLY the next chunk (chunk_index + 1)
    # from the same source.  Fetching both prev and next can double context
    # size; next-only gives continuity without explosion.
    expansion_seeds = [
        h for h in hits if h["similarity"] >= RETRIEVAL_NEIGHBOR_SIMILARITY
    ]

    seen_keys: set[tuple] = {(r["source"], r["chunk_index"]) for r in hits}
    neighbor_chunks: list[dict] = []

    for hit in expansion_seeds:
        for offset in [-1, 1]:
            neighbor_idx = hit["chunk_index"] + offset
            if neighbor_idx < 0:
                continue

            try:
                result = collection.get(
                    where   = {"$and": [{"source": {"$eq": hit["source"]}},
                                        {"chunk_index": {"$eq": neighbor_idx}}]},
                    include = ["documents", "metadatas"],
                )
            except Exception as exc:
                logger.debug(f"  Neighbor fetch skipped ({hit['source']} idx={neighbor_idx}): {exc}")
                continue

            if not result["ids"]:
                continue

            key = (hit["source"], neighbor_idx)
            if key not in seen_keys:
                seen_keys.add(key)
                doc  = result["documents"][0]
                meta = result["metadatas"][0]
                neighbor_chunks.append({
                    "rank":        None,
                    "content":     doc,
                    "source":      meta.get("source", hit["source"]),
                    "page":        meta.get("page", -1),
                    "chunk_index": neighbor_idx,
                    "similarity":  0.0,
                    "is_neighbor": True,
                })

    if neighbor_chunks:
        logger.info(f"  ↳ +{len(neighbor_chunks)} neighbor chunk(s) added for context")

    # ── Chunk Grouping & Content Deduplication ─────────────────────────────────
    # Combine contiguous chunks and skip identical content to reduce redundancy.
    all_chunks = hits + neighbor_chunks
    all_chunks.sort(key=lambda x: (x["source"], x["chunk_index"]))
    
    grouped_chunks = []
    seen_content = set()  # Track unique text to avoid repeating boilerplate
    
    if all_chunks:
        current = all_chunks[0].copy()
        seen_content.add(current["content"].strip().lower()[:200]) # hash first 200 chars
        
        for i in range(1, len(all_chunks)):
            next_chunk = all_chunks[i]
            next_text_hash = next_chunk["content"].strip().lower()[:200]
            
            # Skip if we've seen this exact content (boilerplate/repeated headers)
            if next_text_hash in seen_content:
                continue
            seen_content.add(next_text_hash)
            
            # If same source and indices are contiguous
            if (next_chunk["source"] == current["source"] and 
                next_chunk["chunk_index"] <= current["chunk_index"] + 1):
                
                current["content"] += "\n" + next_chunk["content"]
                current["chunk_index"] = next_chunk["chunk_index"]
                current["similarity"] = max(current["similarity"], next_chunk["similarity"])
            else:
                grouped_chunks.append(current)
                current = next_chunk.copy()
        grouped_chunks.append(current)

    # ── Hard cap ──────────────────────────────────────────────────────────────
    # Regardless of how many chunks passed all the above steps, never send
    # more than RETRIEVAL_MAX_CHUNKS to the LLM to keep prompts tight.
    if len(grouped_chunks) > RETRIEVAL_MAX_CHUNKS:
        logger.info(
            f"  ↳ Capped from {len(grouped_chunks)} → {RETRIEVAL_MAX_CHUNKS} groups "
            f"(RETRIEVAL_MAX_CHUNKS)"
        )
        grouped_chunks = grouped_chunks[:RETRIEVAL_MAX_CHUNKS]

    logger.info(f"  ↳ Final context: {len(grouped_chunks)} grouped chunk(s) sent to LLM")
    return grouped_chunks


# ── Formatted Console Output ───────────────────────────────────────────────────

def print_results(query: str, results: list[dict]) -> None:
    """
    Print retrieval results in a clean, readable format for terminal testing.

    Args:
        query:   The original user query string.
        results: List of result dicts from retrieve().
    """
    SEP  = "─" * 70
    SEP2 = "═" * 70

    print(f"\n{SEP2}")
    print(f"  🔎  QUERY:  {query}")
    print(SEP2)

    if not results:
        print("  No results found.")
        print(f"{SEP2}\n")
        return

    for r in results:
        is_neighbor = r.get("is_neighbor", False)
        if is_neighbor:
            label = "  [neighbor]"
            score_str = ""
        else:
            bar       = "█" * int(r["similarity"] * 20)
            label     = f"  Rank #{r['rank']}"
            score_str = f"  |  Similarity: {r['similarity']:.4f}  {bar}"
        print(f"\n{label}{score_str}")
        print(f"  📄 Source : {r['source']}  (page {r['page']}, chunk {r['chunk_index']})")
        print(SEP)
        # Print a preview (first 400 chars to keep output manageable)
        preview = r["content"][:400].replace("\n", " ")
        if len(r["content"]) > 400:
            preview += "..."
        print(f"  {preview}")

    print(f"\n{SEP2}\n")


# ── Batch Test Retrieval ───────────────────────────────────────────────────────

def run_test_queries(queries: Optional[list[str]] = None) -> None:
    """
    Run a set of example queries and print results to the console.

    Useful for quickly verifying that the vector store is working correctly
    after ingestion. Called automatically during Phase 2 pipeline execution.

    Args:
        queries: List of query strings (defaults to built-in test set).
    """
    if queries is None:
        queries = [
            "What is the leave policy?",
            "How does the induction process work?",
            "What is the training and development process?",
            "What are the rules for employee conduct?",
            "How is performance appraisal done?",
            "What is the suggestion scheme?",
            "Explain the 5S workplace organization methodology",
        ]

    collection = get_or_create_collection()
    logger.info(f"\n🧪 Running {len(queries)} test retrieval queries...\n")

    for query in queries:
        results = retrieve(query, collection=collection)
        print_results(query, results)

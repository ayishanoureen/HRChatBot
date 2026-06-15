"""
src/vector_store.py — ChromaDB Vector Database Module
======================================================
Responsibilities:
  - Create and persist a ChromaDB collection on local disk
  - Insert chunks (text + embeddings + metadata) into ChromaDB
  - Avoid inserting duplicate chunks (idempotent ingestion)
  - Provide helper functions for collection info and reset
  - Keep ChromaDB client reusable across the app (singleton)

Why ChromaDB?
  - Fully local, no server setup required
  - Persists to disk automatically
  - Native support for cosine similarity search
  - Python-native API, easy to integrate with LangChain
"""

import os
import sys
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DIR, CHROMA_COLLECTION_NAME, CHROMA_DISTANCE_METRIC

logger = logging.getLogger(__name__)


# ── Client Singleton ───────────────────────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a persistent ChromaDB client (singleton).

    The database is stored at VECTOR_DIR (vectorstore/) so it
    survives process restarts without re-ingesting everything.

    Returns:
        chromadb.PersistentClient connected to the local vectorstore.
    """
    global _client
    if _client is None:
        os.makedirs(VECTOR_DIR, exist_ok=True)
        logger.info(f"🗄️  Connecting to ChromaDB at '{VECTOR_DIR}'")
        _client = chromadb.PersistentClient(
            path     = VECTOR_DIR,
            settings = Settings(anonymized_telemetry=False),
        )
        logger.info("✔ ChromaDB client ready")
    return _client


def get_or_create_collection(
    name: str = CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Get an existing ChromaDB collection or create a new one.

    The collection uses cosine distance (configured in config.py), which
    is the correct metric when embeddings are L2-normalized.

    Args:
        name: Name of the ChromaDB collection.

    Returns:
        A ChromaDB Collection object.
    """
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name      = name,
        metadata  = {"hnsw:space": CHROMA_DISTANCE_METRIC},
    )

    count = collection.count()
    logger.info(f"📚 Collection '{name}' ready — {count} existing document(s)")
    return collection


# ── Insertion ──────────────────────────────────────────────────────────────────

def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    """
    Retrieve the set of all chunk_ids already stored in the collection.

    Used to filter out duplicates before inserting, making ingestion
    idempotent — safe to re-run without creating duplicate entries.

    Args:
        collection: The ChromaDB collection to inspect.

    Returns:
        Set of string IDs already present in the collection.
    """
    if collection.count() == 0:
        return set()

    result = collection.get(include=[])   # fetch only IDs, no documents
    return set(result["ids"])


def insert_chunks(
    chunks:     list[dict],
    collection: Optional[chromadb.Collection] = None,
    batch_size: int = 100,
) -> int:
    """
    Insert embedded chunks into ChromaDB, skipping any already present.

    Each chunk is stored with:
        - id:        chunk_id (unique string key)
        - embedding: pre-computed vector (list of floats)
        - document:  the chunk's text content
        - metadata:  source filename, page number, chunk_index

    Args:
        chunks:     List of chunk dicts (must include 'embedding' field).
        collection: Target collection (created if None).
        batch_size: Chunks inserted per ChromaDB upsert call.

    Returns:
        Number of new chunks actually inserted.
    """
    if collection is None:
        collection = get_or_create_collection()

    existing_ids  = get_existing_ids(collection)
    new_chunks    = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("⚠️  All chunks already exist in the collection — skipping insertion")
        return 0

    logger.info(
        f"➕ Inserting {len(new_chunks)} new chunks "
        f"({len(chunks) - len(new_chunks)} duplicates skipped)"
    )

    inserted = 0
    for start in range(0, len(new_chunks), batch_size):
        batch = new_chunks[start : start + batch_size]

        collection.add(
            ids        = [c["chunk_id"]  for c in batch],
            embeddings = [c["embedding"] for c in batch],
            documents  = [c["content"]   for c in batch],
            metadatas  = [
                {
                    "source":      c["source"],
                    "page":        c["page"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        inserted += len(batch)
        logger.info(f"  Inserted batch {start // batch_size + 1} — {inserted}/{len(new_chunks)} chunks")

    logger.info(f"✅ Insertion complete — {inserted} new chunks stored in '{CHROMA_COLLECTION_NAME}'")
    return inserted


# ── Info & Utilities ───────────────────────────────────────────────────────────

def collection_info(collection: Optional[chromadb.Collection] = None) -> dict:
    """
    Return basic stats about the current collection.

    Returns:
        Dict with 'name' and 'total_documents'.
    """
    if collection is None:
        collection = get_or_create_collection()

    return {
        "name":            collection.name,
        "total_documents": collection.count(),
    }


def reset_collection(name: str = CHROMA_COLLECTION_NAME) -> None:
    """
    ⚠️  Delete and recreate the collection (destructive — use with caution).

    Useful during development when you want to re-ingest with different
    chunk settings.

    Args:
        name: Name of the collection to reset.
    """
    client = get_chroma_client()
    client.delete_collection(name)
    logger.warning(f"🗑️  Collection '{name}' deleted and reset")
    get_or_create_collection(name)

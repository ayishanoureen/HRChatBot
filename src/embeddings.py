"""
src/embeddings.py — Sentence Transformer Embedding Module
==========================================================
Responsibilities:
  - Load chunked JSON data from processed/all_chunks.json
  - Generate dense vector embeddings using Sentence Transformers
  - Use model: all-MiniLM-L6-v2 (384-dim, fast, high quality)
  - Support efficient batch processing
  - Return chunks augmented with their embedding vectors

Why all-MiniLM-L6-v2?
  - Lightweight (80MB), runs on CPU without issues
  - 384-dimensional embeddings — efficient for ChromaDB
  - Strong semantic similarity performance on short passages
  - No API key needed (fully local)
"""

import os
import json
import logging
import sys

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_CHUNKS_FILE, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


# ── Model Singleton ────────────────────────────────────────────────────────
# The model is loaded once and reused across the app to avoid repeated
# disk reads on every function call.
#
# WHY this design:  Every src/*.py file uses sys.path.append() to import
# config.py.  This can cause Python to load this file under TWO module
# identities ("src.embeddings" and "embeddings"), each with its OWN set
# of module-level globals.  A plain `_model = None` global would therefore
# exist in two copies, and the model would be loaded twice.
#
# FIX: We store the cached model as a class-level attribute on the
# SentenceTransformer class itself.  Because `sentence_transformers` is
# installed as a proper package, its classes are always the same object
# regardless of which copy of *this* module references them.  This gives
# us a true process-wide singleton.
_CACHE_ATTR = "_hr_chatbot_cached_model"


def get_embedding_model() -> SentenceTransformer:
    """Return the embedding model, loading it only on the very first call."""

    # Check the class-level cache (survives dual-module-identity)
    cached = getattr(SentenceTransformer, _CACHE_ATTR, None)
    if cached is not None:
        logger.debug("✔ Reusing cached embedding model (singleton)")
        return cached

    logger.info(f"📦 Loading embedding model: '{EMBEDDING_MODEL}' (CPU mode)")

    # 🔥 FORCE CPU + reduce overhead
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    model.max_seq_length = 256  # prevents long embedding slowdown

    # Store on the class so ALL module copies share the same instance
    setattr(SentenceTransformer, _CACHE_ATTR, model)

    logger.info(
        f"✔ Model loaded — embedding dim: "
        f"{model.get_embedding_dimension()}"
    )
    return model


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_chunks(chunks_file: str = ALL_CHUNKS_FILE) -> list[dict]:
    """
    Load the chunked JSON produced by Phase 1.

    Args:
        chunks_file: Path to all_chunks.json (defaults to config value).

    Returns:
        List of chunk dicts, each with keys:
            chunk_id, source, page, chunk_index, content

    Raises:
        FileNotFoundError: If the chunks file does not exist.
    """
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_file}\n"
            "Run Phase 1 first:  python main.py --phase 1"
        )

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"📂 Loaded {len(chunks)} chunks from '{chunks_file}'")
    return chunks


# ── Embedding Generation ───────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Uses batch processing to avoid loading everything into memory at once.
    Each embedding is a list of floats (384 dimensions for all-MiniLM-L6-v2).

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (same length as input).
    """
    model = get_embedding_model()

    logger.info(f"🔢 Embedding {len(texts)} texts in batches of {EMBEDDING_BATCH_SIZE}...")

    # encode() handles batching internally; show_progress_bar gives a tqdm bar
    embeddings = model.encode(
        texts,
        batch_size        = EMBEDDING_BATCH_SIZE,
        show_progress_bar = True,
        convert_to_numpy  = True,   # numpy → list conversion below
        normalize_embeddings = True, # L2 normalize for cosine similarity
    )

    # ChromaDB expects plain Python lists, not numpy arrays
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """
    Generate an embedding for a single query string.

    Used at retrieval time to embed the user's question.

    Args:
        text: Query string.

    Returns:
        Single embedding vector as a list of floats.
    """
    model = get_embedding_model()
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0].tolist()


# ── Augment Chunks with Embeddings ─────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to every chunk dict.

    Input chunk format:
        {"chunk_id": "...", "source": "...", "page": 1, "chunk_index": 0, "content": "..."}

    Output chunk format (same + embedding):
        {"chunk_id": "...", ..., "content": "...", "embedding": [0.12, -0.03, ...]}

    Args:
        chunks: List of chunk dicts from load_chunks().

    Returns:
        The same list of dicts, each augmented with an 'embedding' key.
    """
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    logger.info(f"✅ Embeddings generated for {len(chunks)} chunks")
    return chunks


# ── Convenience Loader ─────────────────────────────────────────────────────────

def load_and_embed_chunks() -> list[dict]:
    """
    One-call convenience: load chunks from disk and embed them all.

    Returns:
        List of chunk dicts, each containing an 'embedding' field.
    """
    chunks = load_chunks()
    return embed_chunks(chunks)

"""
config.py — Central Configuration for HR Policy Chatbot
========================================================
All paths, constants, and tunable parameters live here.
To change behaviour, edit this file — not the source modules.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")          # Input PDFs
EXTRACTED_DIR  = os.path.join(BASE_DIR, "extracted")     # Raw page JSON
PROCESSED_DIR  = os.path.join(BASE_DIR, "processed")     # Chunked JSON
VECTOR_DIR     = os.path.join(BASE_DIR, "vectorstore")   # Phase 2: ChromaDB

# ── PDF Extraction ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf"]

# Minimum number of characters a page must have to be kept (filters blank/image pages)
MIN_PAGE_CHARS = 50

# ── Text Chunking ───────────────────────────────────────────────────────────
# ~900 chars ≈ 225 tokens — tight enough to target a single policy rule/clause.
# Smaller chunks produce higher similarity scores for specific questions.
# NOTE: after changing these, re-run:  python main.py --phase 1 --phase 2 --reset
CHUNK_SIZE    = 900

# Overlap preserves cross-boundary sentences (~17% of chunk size).
CHUNK_OVERLAP = 150

# Separators tried in order by RecursiveCharacterTextSplitter
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ── Output Files ─────────────────────────────────────────────────────────────
ALL_CHUNKS_FILE = os.path.join(PROCESSED_DIR, "all_chunks.json")

# ── Phase 2: Embeddings ───────────────────────────────────────────────────────
# Sentence Transformer model for generating dense vector embeddings.
# all-MiniLM-L6-v2: 384-dim, ~80MB, runs on CPU, excellent semantic quality.
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"

# Number of chunks to embed in one SentenceTransformer.encode() call.
# Increase on GPU; 64 is safe for CPU with 8GB+ RAM.
EMBEDDING_BATCH_SIZE = 64

# ── Phase 2: ChromaDB ─────────────────────────────────────────────────────────
# Name of the collection inside the ChromaDB database.
CHROMA_COLLECTION_NAME = "hr_policy_chunks"

# Distance metric for nearest-neighbour search.
# Use "cosine" when embeddings are L2-normalised (we normalise in embeddings.py).
CHROMA_DISTANCE_METRIC = "cosine"

# ── Phase 2: Retrieval ────────────────────────────────────────────────────────
# How many candidate chunks to fetch from ChromaDB before filtering.
RETRIEVAL_TOP_K = 15

# Minimum similarity to include a chunk in context.
# 0.30 (slightly lowered) to catch more related sections for broad queries.
RETRIEVAL_MIN_SIMILARITY = 0.30

# Minimum similarity a hit must reach to trigger neighbor expansion.
# Set slightly above RETRIEVAL_MIN_SIMILARITY so only solid hits expand.
RETRIEVAL_NEIGHBOR_SIMILARITY = 0.45

# Hard cap on total chunks (hits + neighbors) sent to the LLM.
# 12 chunks provide enough context for complex multi-page policies.
RETRIEVAL_MAX_CHUNKS = 12

# ── Reranking ─────────────────────────────────────────────────────────────────
# Cross-encoder model used to rerank the initial retrieval pool.
RERANKER_MODEL      = "BAAI/bge-reranker-base"

# How many chunks to pull from ChromaDB before reranking.
RERANKER_RETRIEVE_K = 15

# How many top-reranked chunks to send to the LLM.
RERANKER_TOP_K      = 5

# ── Phase 3: LLM Configuration ────────────────────────────────────────────────
# Provider: Google Gemini API (fast, cloud-based, no local GPU needed)
import dotenv
dotenv.load_dotenv()

LLM_PROVIDER = "gemini"

# Gemini settings
# Get your API key: https://aistudio.google.com/app/apikey
# Add to .env file:  GEMINI_API_KEY=your-key-here
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"     # Reliable model with high free-tier limits

# Shared generation settings
LLM_TEMPERATURE = 0.1    # Very low = highly factual, no creative deviation
LLM_MAX_TOKENS  = 2048   # Enough for the longest policy answers without truncation


"""
main.py — HR Policy Chatbot: Complete Pipeline Entry Point
===========================================================
Run modes:

  python main.py --phase 1          # PDF extraction + chunking
  python main.py --phase 2          # embeddings + ChromaDB ingestion
  python main.py --phase all        # run Phase 1 then Phase 2

  python main.py --chat             # interactive RAG chatbot (CLI)
  python main.py --ask "question"   # single RAG query → answer + sources
  python main.py --query "question" # raw retrieval only (no LLM)

  python main.py --reset            # ⚠️  wipe & re-ingest ChromaDB
  python main.py --debug            # verbose logging

Quick start (first time):
  python main.py --phase all        # build the full pipeline
  python main.py --chat             # start chatting
"""
import sys
import logging
import argparse
import io

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError when printing emojis/special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# ── Logging ────────────────────────────────────────────────────────────────────

def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    logging.basicConfig(
        level    = level,
        format   = fmt,
        datefmt  = "%H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
    )


# ── Arguments ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "HR Policy Chatbot — RAG Pipeline & Chat Interface",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog          = __doc__,
    )
    parser.add_argument("--phase",  choices=["1", "2", "all"], default=None,
                        help="Build pipeline phase(s)")
    parser.add_argument("--chat",   action="store_true", default=False,
                        help="Start interactive RAG chatbot session")
    parser.add_argument("--ask",    type=str, default=None,
                        help="Single RAG question — returns full LLM answer")
    parser.add_argument("--query",  type=str, default=None,
                        help="Raw semantic retrieval only (no LLM, shows chunks)")
    parser.add_argument("--reset",  action="store_true", default=False,
                        help="⚠️  Reset ChromaDB collection before Phase 2")
    parser.add_argument("--debug",  action="store_true", default=False,
                        help="Enable verbose DEBUG logging")
    return parser.parse_args()


# ── Phase Runners ──────────────────────────────────────────────────────────────

def run_phase1(logger) -> None:
    from src.pipeline import run_pipeline
    logger.info("━" * 58)
    logger.info("  PHASE 1 — PDF Extraction & Preprocessing")
    logger.info("━" * 58)
    chunks = run_pipeline()
    if not chunks:
        logger.error("Phase 1 produced no chunks. Check data/ folder.")
        sys.exit(1)
    logger.info(f"✅ Phase 1 done — {len(chunks)} chunks saved to processed/")


def run_phase2(logger, reset: bool = False) -> None:
    from src.embeddings   import load_and_embed_chunks
    from src.vector_store import get_or_create_collection, insert_chunks, reset_collection

    logger.info("\n" + "━" * 58)
    logger.info("  PHASE 2 — Embeddings & Vector Store")
    logger.info("━" * 58)

    if reset:
        logger.warning("--reset flag: wiping existing ChromaDB collection...")
        reset_collection()

    logger.info("[2.1] Loading chunks and generating embeddings...")
    embedded_chunks = load_and_embed_chunks()

    logger.info("[2.2] Inserting into ChromaDB...")
    collection = get_or_create_collection()
    inserted   = insert_chunks(embedded_chunks, collection=collection)
    total      = collection.count()
    logger.info(f"✅ Phase 2 done — {inserted} new chunks inserted | {total} total in store")


# ── RAG Ask (single question) ──────────────────────────────────────────────────

def run_ask(question: str) -> None:
    """Run a single question through the full RAG pipeline and print the answer."""
    from src.rag_chain import ask

    response = ask(question)

    SEP = "═" * 65
    print(f"\n{SEP}")
    print(f"  💬  {response.query}")
    print(SEP)
    print()
    print(response.answer)
    print()
    if response.sources:
        print("─" * 65)
        print("  📚  Sources:")
        print(response.format_sources())
    print(f"{SEP}\n")


# ── Raw Retrieval (no LLM) ─────────────────────────────────────────────────────

def run_raw_query(question: str) -> None:
    """Retrieve and display raw chunks without calling the LLM."""
    from src.retriever import retrieve, print_results
    results = retrieve(question)
    print_results(question, results)


# ── Interactive Chat Mode ──────────────────────────────────────────────────────

CHAT_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🤖   HR Policy Chatbot  —  RAG Assistant       ║
║                                                              ║
║  Ask any question about HR policies and get instant answers. ║
║  Type  'quit' or 'exit' to end the session.                  ║
║  Type  'sources' after a question to see citations.          ║
╚══════════════════════════════════════════════════════════════╝
"""

def run_chat() -> None:
    """
    Interactive CLI chatbot loop.
    Maintains a ChromaDB collection across turns for efficiency.
    """
    from src.rag_chain    import ask
    from src.vector_store import get_or_create_collection
    from src.embeddings   import get_embedding_model

    print(CHAT_BANNER)

    # Pre-load the embedding model once to avoid reload on every query
    print("⏳ Loading embedding model (one-time)...")
    get_embedding_model()
    print("✅ Embedding model ready.\n")

    # Pre-load the collection once to avoid repeated disk reads
    try:
        collection = get_or_create_collection()
        if collection.count() == 0:
            print("⚠️  Vector store is empty. Run:  python main.py --phase 2\n")
            return
        print(f"✅ Vector store ready — {collection.count()} chunks loaded.\n")
    except Exception as e:
        print(f"❌ Could not connect to vector store: {e}")
        print("   Run:  python main.py --phase 2  first.\n")
        return

    last_response = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\n👋 Goodbye! Have a great day.")
            break

        # Show sources for the last response
        if user_input.lower() == "sources":
            if last_response and last_response.sources:
                print("\n📚 Sources from last answer:")
                print(last_response.format_sources())
                print()
            else:
                print("  No previous answer to show sources for.\n")
            continue

        # ── Full RAG pipeline ──────────────────────────────────────────────
        print("\n🔍 Searching HR policies...\n")
        try:
            response      = ask(user_input, collection=collection)
            last_response = response
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue

        # Print the answer
        print("─" * 65)
        print(f"🤖 HR Assistant:\n")
        print(response.answer)

        # Show brief source hints
        if response.sources:
            src_summary = ", ".join(
                f"{s['source'].split('_')[0]}…p{s['page']}"
                for s in response.sources[:3]
            )
            print(f"\n   📎 Sources: {src_summary}  (type 'sources' for full list)")
        print("─" * 65 + "\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # ── Pipeline build modes ───────────────────────────────────────────────────
    if args.phase:
        logger.info("🚀 HR Policy Chatbot — Pipeline Builder")

        if args.phase in ("1", "all"):
            run_phase1(logger)

        if args.phase in ("2", "all"):
            run_phase2(logger, reset=args.reset)

        logger.info("\n🎉 Pipeline ready! Start the chatbot with:  python main.py --chat")
        return

    # ── Single RAG question ────────────────────────────────────────────────────
    if args.ask:
        run_ask(args.ask)
        return

    # ── Raw retrieval (no LLM) ─────────────────────────────────────────────────
    if args.query:
        run_raw_query(args.query)
        return

    # ── Interactive chat ───────────────────────────────────────────────────────
    if args.chat:
        run_chat()
        return

    # ── Default: show help ─────────────────────────────────────────────────────
    print(__doc__)


if __name__ == "__main__":
    main()

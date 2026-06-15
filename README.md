# HR Policy Chatbot

An AI-powered RAG chatbot for answering employee HR policy questions, built with Python, LangChain, ChromaDB, and Streamlit.

---

## Project Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | PDF extraction, cleaning, chunking |
| Phase 2 | 🔜 Next | Embeddings + ChromaDB vector store |
| Phase 3 | 🔜 Planned | LLM integration + RAG retrieval |
| Phase 4 | 🔜 Planned | Streamlit chat UI |

---

## Project Structure

```
HRChatBot/
├── data/               ← HR policy PDFs (input)
├── extracted/          ← Per-PDF raw page JSON (auto-generated)
├── processed/          ← all_chunks.json (auto-generated, Phase 2 input)
├── vectorstore/        ← ChromaDB database (Phase 2, auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py   ← pdfplumber extraction
│   ├── text_cleaner.py    ← unicode normalization & noise removal
│   ├── text_chunker.py    ← LangChain RecursiveCharacterTextSplitter
│   └── pipeline.py        ← orchestrates all stages
│
├── config.py           ← all tunable settings
├── main.py             ← entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Phase 1 (PDF preprocessing)

```bash
python main.py
```

For verbose output:

```bash
python main.py --debug
```

---

## Output

After running Phase 1:

- `extracted/<filename>.json` — raw page text per PDF (for debugging)
- `processed/all_chunks.json` — all chunks with metadata, ready for embedding

Each chunk looks like:

```json
{
  "chunk_id":    "_Company_Policy_Part_A__p3_c0",
  "source":      "_Company_Policy_Part_A.pdf",
  "page":        3,
  "chunk_index": 0,
  "content":     "Leave policy text..."
}
```

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Max characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `MIN_PAGE_CHARS` | 50 | Min chars to keep a page |

---

## Tech Stack

- **pdfplumber** — PDF text extraction
- **LangChain** — text splitting + RAG framework
- **ChromaDB** — local vector database (Phase 2)
- **Sentence Transformers** — embeddings (Phase 2)
- **Streamlit** — chat UI (Phase 4)

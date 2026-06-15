# 🏢 HR Policy Chatbot

An AI-powered HR assistant that answers employee policy questions using Retrieval-Augmented Generation (RAG).

The system extracts information from HR policy documents, stores them in a ChromaDB vector database, retrieves relevant sections, and generates contextual answers using Gemini.

## Features

* 📄 PDF HR policy ingestion
* ✂️ Automatic text chunking
* 🧠 Sentence Transformer embeddings
* 🔍 ChromaDB semantic search
* 🤖 Gemini-powered responses
* 💬 Streamlit chat interface
* 📚 Source-based answers

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Sentence Transformers
* Gemini API
* pdfplumber

## Project Structure

```text
HRChatBot/
├── data/
├── processed/
├── vectorstore/
├── src/
├── app.py
├── main.py
└── config.py
```

## Installation

```bash
git clone <repository-url>
cd HRChatBot

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

## Build Vector Store

```bash
python main.py --phase all
```

## Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Example Questions

* What is the leave policy?
* What is the notice period?
* How does the appraisal process work?
* What are the employee benefits?



# Grounded RAG

A complete, modular Basic Retrieval-Augmented Generation application built with
Python, Streamlit, OpenAI embeddings and chat generation, and a local FAISS
vector index.

This is intentionally the foundation only. It does not implement a query
analyzer, CAR-RAG classification, relevance checker, evidence/claim checker,
adaptive re-retrieval, or evaluation metrics.

## Requirements

- Python 3.11 or 3.12
- An OpenAI API key with access to embeddings and chat completions

## Project structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── documents/
│       └── .gitkeep
├── indexes/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── chunker.py
    ├── document_loader.py
    ├── embeddings.py
    ├── generator.py
    ├── retriever.py
    └── vector_store.py
```

## Local setup

### 1. Create and activate a virtual environment

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the API key

Copy `.env.example` to `.env` and set the key:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
OPENAI_API_KEY=your_real_key
```

Never commit `.env` or place a real key in `.env.example`.

### 4. Run the application

```bash
streamlit run app.py
```

The browser will open to the Streamlit interface. In a headless environment,
Streamlit prints a local URL that can be opened manually.

## Using the app

1. Upload one or more PDF or TXT files.
2. Adjust chunking or the number of retrieved chunks in the sidebar if needed.
3. Click **Process documents**. Text is extracted, cleaned, chunked, embedded,
   and saved to the local FAISS index.
4. Enter a question and click **Ask**.
5. Review the concise answer and the retrieved source chunks, document names,
   character ranges, and cosine-similarity scores.

Processed source files are stored under `data/documents/`. The FAISS index and
its traceability metadata are stored under `indexes/`.

## Error handling

The app reports missing API configuration, unsupported or invalid PDFs, empty
documents, empty questions, missing indexes, mismatched index metadata, and
OpenAI request failures in the Streamlit interface.
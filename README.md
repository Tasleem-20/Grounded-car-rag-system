# 🔎 Grounded RAG System

A modular **Retrieval-Augmented Generation (RAG)** application built with **Python, Streamlit, OpenAI embeddings, OpenAI chat generation, and FAISS**.

The system allows users to upload documents, convert them into searchable vector representations, retrieve relevant content for a question, and generate answers grounded in the retrieved evidence.

> **Note:** This repository represents the foundational RAG implementation. Advanced CAR-RAG components such as query classification, relevance checking, evidence/claim verification, adaptive re-retrieval, and evaluation metrics are planned as future enhancements.

## 🚀 Live Demo

## 🚀 Live Demo

🔗 **[Try the Live Application](https://grounded-car-rag-system-lt3uxr7rpndyjmpwvxqm3g.streamlit.app/)**

The application is deployed using Streamlit and provides an interactive web interface for document processing and question answering.

## ✨ Key Features

* 📄 Upload PDF and TXT documents
* ✂️ Configurable document chunking
* 🧠 Generate embeddings using OpenAI
* 🔍 Semantic similarity search using FAISS
* 📚 Retrieve relevant document chunks
* 🤖 Generate answers using an OpenAI chat model
* 📌 Display retrieved evidence and source information
* 📊 Show cosine-similarity scores for retrieved chunks
* 🔗 Maintain traceability between answers and source documents
* 🖥️ Interactive Streamlit interface
* ⚠️ Error handling for invalid documents, missing configuration, empty queries, and retrieval issues

## 🏗️ System Workflow

```text
                Documents
                    │
                    ▼
            Document Loader
                    │
                    ▼
               Chunking
                    │
                    ▼
          OpenAI Embeddings
                    │
                    ▼
             FAISS Index
                    │
                    │
User Question ──────┤
                    ▼
               Retriever
                    │
                    ▼
          Relevant Evidence
                    │
                    ▼
          OpenAI Chat Model
                    │
                    ▼
            Grounded Answer
                    │
                    ▼
          Source Information
```

## 🛠️ Technologies Used

| Technology        | Purpose                                             |
| ----------------- | --------------------------------------------------- |
| Python            | Core application development                        |
| Streamlit         | Interactive web interface                           |
| OpenAI Embeddings | Convert document chunks into vector representations |
| OpenAI Chat Model | Generate answers from retrieved context             |
| FAISS             | Efficient vector similarity search                  |
| PyPDF             | PDF text extraction                                 |
| NumPy             | Numerical and vector operations                     |
| python-dotenv     | Environment variable management                     |

## 📂 Project Structure

```text
Grounded-car-rag-system/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── documents/
│       └── .gitkeep
│
├── indexes/
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── chunker.py
    ├── document_loader.py
    ├── embeddings.py
    ├── generator.py
    ├── retriever.py
    └── vector_store.py
```

## ⚙️ Requirements

* Python 3.11 or 3.12
* OpenAI API key
* Access to OpenAI embeddings and chat generation models

## 💻 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tasleem-20/Grounded-car-rag-system.git
cd Grounded-car-rag-system
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and add your OpenAI API key.

```dotenv
OPENAI_API_KEY=your_real_key
```

⚠️ **Never commit your actual `.env` file or expose API keys in the repository.**

### 5. Run the application

```bash
streamlit run app.py
```

The Streamlit interface will then be available in your browser.

## 📖 How to Use

### Step 1 — Upload Documents

Upload one or more **PDF or TXT files** through the Streamlit interface.

### Step 2 — Process Documents

Click **Process documents**.

The application:

1. Extracts text from the uploaded documents.
2. Cleans and prepares the extracted text.
3. Splits the text into manageable chunks.
4. Generates embeddings for the chunks.
5. Stores the embeddings in a local FAISS index.

### Step 3 — Ask a Question

Enter a question related to the uploaded documents and click **Ask**.

### Step 4 — Retrieve Evidence

The system searches the FAISS vector index and retrieves the most relevant document chunks.

### Step 5 — Generate a Grounded Answer

The retrieved evidence is provided to the language model so that the generated response is based on the available document context.

The application also displays source information and similarity scores to improve transparency and traceability.

## 🔍 Retrieval Process

The retrieval pipeline works as follows:

```text
Document
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Index
   ↓
User Query Embedding
   ↓
Similarity Search
   ↓
Top Relevant Chunks
   ↓
LLM Context
   ↓
Grounded Response
```

## 🧩 Current Implementation

The current version focuses on the **core RAG pipeline**:

* Document ingestion
* Text extraction
* Chunking
* Embedding generation
* FAISS vector storage
* Semantic retrieval
* Context-based answer generation
* Evidence/source display

The repository does **not yet implement** the complete CAR-RAG pipeline.

## 🔮 Future Enhancements

The following components can be added to extend the system toward a more advanced CAR-RAG implementation:

* 🧭 Query type classification
* 📊 Retrieval sufficiency/relevance checking
* 🔄 Adaptive re-retrieval
* 🧾 Evidence verification
* 🔗 Claim-level grounding
* 📈 RAG evaluation metrics
* 🎯 Retrieval quality scoring
* 🧠 More advanced query analysis
* 📚 Improved multi-document reasoning

## ⚠️ Error Handling

The application handles common errors including:

* Missing OpenAI API configuration
* Invalid or unsupported PDF files
* Empty documents
* Empty questions
* Missing FAISS indexes
* Incompatible index metadata
* OpenAI API request failures

Errors are reported directly through the Streamlit interface.

## 🔐 Security

* API keys are stored through environment variables.
* `.env` is excluded from version control.
* `.env.example` is provided as a configuration template.
* No real API credentials should be committed to GitHub.

## 🎯 Project Objective

The objective of this project is to build a transparent and modular **Retrieval-Augmented Generation system** that can answer questions using information retrieved from user-provided documents rather than relying solely on the language model's internal knowledge.

The project also provides a foundation for implementing more advanced **adaptive and evidence-grounded RAG techniques**.

## 👨‍💻 Author

**Shaik Tasleem**

GitHub: `Tasleem-20`

## ⭐ Acknowledgement

This project was developed as part of an exploration of **Retrieval-Augmented Generation, vector search, document retrieval, and grounded question answering**.

If you find the project useful, consider giving the repository a ⭐.

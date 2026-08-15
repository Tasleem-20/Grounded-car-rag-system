"""
Grounded CAR-RAG Builder — Modern Academic UI

A 4th-Year B.Tech Computer Science Project Demonstration Interface.
Features 5 interactive pages:
  1. Home / Dashboard
  2. Document Management
  3. RAG Chat & Live Workflow
  4. Evidence & Sources Inspector
  5. System Architecture & About
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.answer_checker import AnswerCheckError, AnswerChecker
from src.chunker import chunk_document
from src.document_loader import DocumentLoadError, load_document
from src.embeddings import EmbeddingError, EmbeddingService
from src.evidence_checker import EvidenceCheckError, EvidenceChecker
from src.generator import GenerationError, Generator
from src.query_analyzer import QueryAnalyzer
from src.retriever import Retriever
from src.vector_store import VectorStore, VectorStoreError

load_dotenv()

# Streamlit Community Cloud secret fallback
try:
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Directories & Constants
ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
INDEX_DIR = ROOT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


def ensure_directories() -> None:
    """Create local data folders used by the application."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_saved_store() -> VectorStore | None:
    """Load previously processed vector index if available."""
    if not VectorStore.exists(INDEX_PATH, METADATA_PATH):
        return None
    try:
        return VectorStore.load(INDEX_PATH, METADATA_PATH)
    except VectorStoreError:
        return None


def process_documents(
    uploaded_files: list,
    embedding_service: EmbeddingService,
    chunk_size: int,
    chunk_overlap: int,
) -> VectorStore:
    """Extract, chunk, embed, and persist all uploaded documents."""
    if not uploaded_files:
        raise ValueError("Upload at least one PDF or TXT document before processing.")

    documents = []
    chunks = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        if not content:
            continue

        document = load_document(uploaded_file.name, content)
        documents.append(document)
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))

        destination = DOCUMENTS_DIR / Path(uploaded_file.name).name
        destination.write_bytes(content)

    if not documents:
        raise ValueError("The uploaded files were empty.")
    if not chunks:
        raise ValueError("No readable text was found in the uploaded documents.")

    embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
    store = VectorStore.from_embeddings(chunks, embeddings)
    store.save(INDEX_PATH, METADATA_PATH)
    return store


# -----------------------------------------------------------------------------
# CSS Styling & Aesthetic Enhancements
# -----------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Container & Layout */
        .block-container {
            max-width: 1200px;
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        /* Top Header Styling */
        .academic-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .academic-title {
            color: #38bdf8;
            font-size: 1.85rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .academic-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
        }
        .academic-badge {
            background-color: #0369a1;
            color: #e0f2fe;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Metric Cards */
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .metric-value {
            color: #38bdf8;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Workflow Step Cards */
        .wf-card {
            background: #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 0 8px 8px 0;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .wf-card.success { border-left-color: #22c55e; }
        .wf-card.warning { border-left-color: #eab308; }
        .wf-card.info { border-left-color: #3b82f6; }
        .wf-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #f8fafc;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .wf-detail {
            color: #cbd5e1;
            font-size: 0.88rem;
            margin-top: 0.3rem;
        }

        /* Badges */
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .pill-blue { background-color: #1e40af; color: #dbeafe; }
        .pill-green { background-color: #166534; color: #dcfce7; }
        .pill-yellow { background-color: #854d0e; color: #fef9c3; }
        .pill-purple { background-color: #6b21a8; color: #f3e8ff; }

        /* Answer Card */
        .answer-box {
            background-color: #0f172a;
            border: 1px solid #3b82f6;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        .answer-header {
            color: #60a5fa;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .answer-text {
            color: #f1f5f9;
            font-size: 1rem;
            line-height: 1.6;
        }

        /* Evidence Expanders */
        .evidence-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Navigation & Sidebar
# -----------------------------------------------------------------------------
def render_sidebar(store: VectorStore | None) -> tuple[str, int, int, int]:
    with st.sidebar:
        st.markdown("### 🎓 RAG Navigation")
        options = [
            "📌 Dashboard / Home",
            "📂 Document Management",
            "💬 Ask Questions (RAG)",
            "🔍 Evidence & Sources",
            "ℹ️ System & Architecture",
        ]
        if "nav_page" not in st.session_state or st.session_state.nav_page not in options:
            st.session_state.nav_page = options[0]

        page = st.radio(
            "Select Page",
            options=options,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### ⚙️ RAG Hyperparameters")
        top_k = st.slider(
            "Retrieved chunks (top_k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of document chunks passed to evidence check & answer generator.",
        )
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=400,
            max_value=1600,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Maximum characters per searchable text chunk.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=400,
            value=DEFAULT_CHUNK_OVERLAP,
            step=25,
            help="Overlapping characters between adjacent chunks for context preservation.",
        )

        st.divider()
        st.markdown("### 📊 Vector Index Status")
        if store is None:
            st.warning("No active vector index.")
            st.caption("Go to 'Document Management' to upload & index documents.")
        else:
            st.success("FAISS Index Active")
            st.metric("Indexed Chunks", len(store.chunks))
            st.metric("Source Documents", len(store.document_names))

        st.divider()
        st.caption("4th-Year B.Tech CS Project — CAR-RAG")

    return page, top_k, chunk_size, chunk_overlap


def set_nav_page(page_name: str) -> None:
    st.session_state["nav_page"] = page_name


# -----------------------------------------------------------------------------
# PAGE 1: Home / Dashboard
# -----------------------------------------------------------------------------
def render_dashboard_page(store: VectorStore | None) -> None:
    st.markdown(
        """
        <div class="academic-header">
            <div class="academic-title">
                🤖 Grounded CAR-RAG System
                <span class="academic-badge">Academic Project</span>
            </div>
            <div class="academic-subtitle">
                Self-Checking Retrieval-Augmented Generation Architecture for Accurate Document QA
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        doc_count = len(store.document_names) if store else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{doc_count}</div>
                <div class="metric-label">Loaded Documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        chunk_count = len(store.chunks) if store else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{chunk_count}</div>
                <div class="metric-label">Searchable Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.1rem; padding-top:0.3rem;">MiniLM-L6</div>
                <div class="metric-label">Vector Embedding</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.1rem; padding-top:0.3rem;">Llama-3.3-70B</div>
                <div class="metric-label">LLM Engine (Groq)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Core System Objectives")

    st.markdown(
        """
        **Retrieval-Augmented Generation (RAG)** addresses large language model hallucinations by grounding answers 
        directly in retrieved domain-specific source evidence.

        This system implements an advanced **Self-Checking Adaptive RAG (CAR-RAG)** workflow featuring:
        - 📄 **Metadata-Preserving Document Chunking**: PDF & TXT parser with exact character-offset tracking.
        - 🔎 **FAISS Vector Store**: Fast L2-normalized cosine similarity search over local sentence embeddings.
        - 🧠 **Query Complexity Classifier**: Pre-retrieval routing to determine retrieval scope.
        - ⚖️ **Semantic Evidence Checker**: LLM-as-a-Judge verification of evidence sufficiency prior to generation.
        - 🔄 **Adaptive Re-Retrieval**: Automatic expansion of retrieval parameters if initial evidence is incomplete.
        - 🛡️ **Answer Grounding Verifier**: Self-correction phase checking generated claims against source evidence.
        """
    )

    st.divider()
    st.subheader("🚀 Quick Actions")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info("### 1️⃣ Upload & Index\nUpload your project documents (PDF or TXT) to construct the local vector index.")
        st.button(
            "📂 Open Document Management",
            type="primary",
            use_container_width=True,
            on_click=set_nav_page,
            args=("📂 Document Management",),
        )
    with col_b:
        st.success("### 2️⃣ Ask Questions\nQuery the vector store, inspect live RAG execution steps, and review verified evidence.")
        st.button(
            "💬 Open RAG Chat",
            type="primary",
            use_container_width=True,
            on_click=set_nav_page,
            args=("💬 Ask Questions (RAG)",),
        )


# -----------------------------------------------------------------------------
# PAGE 2: Document Management
# -----------------------------------------------------------------------------
def render_document_page(
    store: VectorStore | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    st.markdown("## 📂 Document Ingestion & Index Management")
    st.write(
        "Upload PDF or TXT reference files. Text is automatically cleaned, split into overlapping "
        "chunks with character range metadata, converted into 384-dimensional dense vectors, and stored in FAISS."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Upload Reference Files")
        uploaded_files = st.file_uploader(
            "Supported formats: PDF, TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Uploaded files are processed locally and indexed in indexes/faiss.index",
        )

        if st.button(
            "⚡ Process & Build Vector Index",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        ):
            try:
                with st.status("Building FAISS Vector Index...", expanded=True) as status:
                    st.write("📖 Extracting text and parsing character boundaries...")
                    embedding_service = EmbeddingService()

                    st.write("🧬 Generating embeddings with Sentence-Transformers...")
                    new_store = process_documents(
                        uploaded_files,
                        embedding_service,
                        chunk_size,
                        chunk_overlap,
                    )
                    st.session_state.store = new_store
                    st.session_state.last_results = []
                    st.session_state.last_answer = ""
                    st.session_state.last_workflow = []

                    status.update(
                        label="Indexing Completed Successfully!",
                        state="complete",
                        expanded=False,
                    )
                st.rerun()
            except (DocumentLoadError, EmbeddingError, VectorStoreError, ValueError) as error:
                st.error(f"Processing Error: {error}")
            except Exception as error:
                st.error(f"Unexpected document processing error: {error}")

    with col2:
        st.subheader("Active Document Store")
        if store is None:
            st.info("No documents indexed yet.")
        else:
            st.markdown(f"**Total Source Files:** `{len(store.document_names)}`")
            st.markdown(f"**Total Searchable Chunks:** `{len(store.chunks)}`")
            st.markdown("---")
            st.markdown("**Indexed Files List:**")
            for doc in store.document_names:
                st.markdown(f"- 📄 `{doc}`")

            if DOCUMENTS_DIR.exists():
                files = list(DOCUMENTS_DIR.glob("*"))
                if files:
                    st.markdown("---")
                    st.markdown("**Storage Information:**")
                    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
                    st.caption(f"Directory: `data/documents/` ({total_bytes / 1024:.1f} KB)")


# -----------------------------------------------------------------------------
# PAGE 3: Ask Questions / RAG Chat & Workflow
# -----------------------------------------------------------------------------
def render_chat_page(
    store: VectorStore | None,
    top_k: int,
) -> None:
    st.markdown("## 💬 Ask Questions & Live RAG Workflow")
    st.write(
        "Ask questions grounded in your uploaded documents. Observe real-time query classification, "
        "evidence evaluation, adaptive re-retrieval, and answer verification."
    )

    if store is None:
        st.warning("⚠️ No active vector index found. Please upload documents in Document Management first.")
        return

    # Question Input Form
    with st.form("rag_query_form", clear_on_submit=False):
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the core findings or requirements detailed in the document?",
            height=90,
        )
        submit_button = st.form_submit_button("🚀 Submit Question & Run CAR-RAG Pipeline", type="primary")

    if submit_button:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
            return

        workflow_log = []

        try:
            # 1. Query Analysis
            with st.spinner("Analyzing question complexity..."):
                query_analyzer = QueryAnalyzer()
                analysis = query_analyzer.analyze(question)
                st.session_state.last_analysis = analysis

            workflow_log.append({
                "stage": "Query Analysis",
                "status": "info",
                "detail": f"Type: '{analysis.query_type}' | Adaptive retrieval required: {analysis.needs_more_retrieval}"
            })

            # 2. Initial Retrieval
            with st.spinner("Embedding query & searching FAISS index..."):
                embedding_service = EmbeddingService()
                retriever = Retriever(store, embedding_service)
                results = retriever.retrieve(question, top_k=top_k)

            workflow_log.append({
                "stage": "Initial Retrieval",
                "status": "info",
                "detail": f"Retrieved top {len(results)} chunks (top_k={top_k})"
            })

            # 3. Adaptive retrieval if complex
            if analysis.needs_more_retrieval:
                workflow_log.append({
                    "stage": "Adaptive Retrieval (Query Complexity)",
                    "status": "warning",
                    "detail": f"Complex query detected. Expanding retrieval to top_k={top_k * 2}"
                })
                add_results = retriever.retrieve(question, top_k=top_k * 2)
                combined = results + add_results
                unique_results = {(r.document_name, r.chunk_index): r for r in combined}
                results = list(unique_results.values())
                results.sort(key=lambda r: r.score, reverse=True)
                results = results[:top_k * 2]

            # 4. Evidence Sufficiency Check
            evidence_checker = EvidenceChecker()
            evidence_check = evidence_checker.check(question, results)
            st.session_state.last_evidence_check = evidence_check

            if evidence_check.sufficient:
                workflow_log.append({
                    "stage": "Evidence Checker",
                    "status": "success",
                    "detail": f"Sufficient: True | Score: {evidence_check.score:.3f} | Reason: {evidence_check.reason}"
                })
            else:
                workflow_log.append({
                    "stage": "Evidence Checker",
                    "status": "warning",
                    "detail": f"Sufficient: False | Score: {evidence_check.score:.3f} | Reason: {evidence_check.reason}"
                })
                # Re-retrieval for insufficient evidence
                workflow_log.append({
                    "stage": "Adaptive Re-Retrieval (Insufficient Evidence)",
                    "status": "warning",
                    "detail": "Performing expanded vector search due to low evidence score..."
                })
                add_results = retriever.retrieve(question, top_k=top_k * 2)
                combined = results + add_results
                unique_results = {(r.document_name, r.chunk_index): r for r in combined}
                results = list(unique_results.values())
                results.sort(key=lambda r: r.score, reverse=True)
                results = results[:top_k * 2]
                evidence_check = evidence_checker.check(question, results)
                st.session_state.last_evidence_check = evidence_check

            # 5. Answer Generation
            generator = Generator()
            answer_checker = AnswerChecker()

            answer = generator.answer(question, results)
            answer_check = answer_checker.check(question, answer, results)
            st.session_state.last_answer_check = answer_check

            if not answer_check.supported:
                workflow_log.append({
                    "stage": "Answer Verification & Self-Correction",
                    "status": "warning",
                    "detail": f"First pass unsupported (score: {answer_check.score:.3f}). Regenerating..."
                })
                answer = generator.answer(question, results)
                answer_check = answer_checker.check(question, answer, results)
                st.session_state.last_answer_check = answer_check

            if answer_check.supported:
                workflow_log.append({
                    "stage": "Answer Verification",
                    "status": "success",
                    "detail": f"Supported: True | Score: {answer_check.score:.3f} | Reason: {answer_check.reason}"
                })
                st.session_state.last_answer = answer
            else:
                workflow_log.append({
                    "stage": "Answer Verification",
                    "status": "warning",
                    "detail": f"Supported: False | Score: {answer_check.score:.3f} | Fallback grounded refusal active."
                })
                st.session_state.last_answer = (
                    "The available documents do not contain enough supported information to provide a reliable answer."
                )

            st.session_state.last_question = question
            st.session_state.last_results = results
            st.session_state.last_workflow = workflow_log

        except (
            EmbeddingError,
            GenerationError,
            VectorStoreError,
            EvidenceCheckError,
            AnswerCheckError,
        ) as error:
            st.error(f"RAG Pipeline Failure: {error}")
        except Exception as error:
            st.error(f"Unexpected Pipeline Error: {error}")

    # Display Answer and Workflow if available
    if getattr(st.session_state, "last_answer", None):
        st.divider()

        # Display Pipeline Metric Badges
        analysis = getattr(st.session_state, "last_analysis", None)
        ev_check = getattr(st.session_state, "last_evidence_check", None)
        ans_check = getattr(st.session_state, "last_answer_check", None)

        badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)

        if analysis:
            with badge_col1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Query Type</div>
                        <div class="status-pill pill-blue">{analysis.query_type.upper()}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        if ev_check:
            with badge_col2:
                pill_cls = "pill-green" if ev_check.sufficient else "pill-yellow"
                status_txt = "SUFFICIENT" if ev_check.sufficient else "INSUFFICIENT"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Evidence Score ({ev_check.score:.2f})</div>
                        <div class="status-pill {pill_cls}">{status_txt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        if ans_check:
            with badge_col3:
                pill_cls = "pill-green" if ans_check.supported else "pill-yellow"
                status_txt = "VERIFIED" if ans_check.supported else "UNSUPPORTED"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Answer Grounding ({ans_check.score:.2f})</div>
                        <div class="status-pill {pill_cls}">{status_txt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        with badge_col4:
            res_len = len(st.session_state.last_results)
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Evidentiary Chunks</div>
                    <div class="status-pill pill-purple">{res_len} CHUNKS</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Grounded Answer Display
        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-header">🤖 Grounded Answer</div>
                <div class="answer-text">{st.session_state.last_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Workflow Progression
        with st.expander("🔄 View Live RAG Execution Steps & Decision Log", expanded=True):
            for step in getattr(st.session_state, "last_workflow", []):
                st.markdown(
                    f"""
                    <div class="wf-card {step['status']}">
                        <div class="wf-title">{step['stage']}</div>
                        <div class="wf-detail">{step['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# PAGE 4: Evidence & Sources Inspector
# -----------------------------------------------------------------------------
def render_evidence_page() -> None:
    st.markdown("## 🔍 Evidence & Retrieved Sources Inspector")
    st.write(
        "Examine the exact text chunks retrieved from the vector index, their document origins, "
        "character offsets, and cosine similarity scores."
    )

    results = getattr(st.session_state, "last_results", [])
    question = getattr(st.session_state, "last_question", "")

    if question:
        st.info(f"**Current Question:** *\"{question}\"*")

    if not results:
        st.warning("No evidence chunks available yet. Ask a question in the RAG Chat page first.")
        return

    st.subheader(f"Retrieved Chunks ({len(results)})")

    for idx, result in enumerate(results, start=1):
        score_pct = max(0, min(100, int(result.score * 100)))

        with st.expander(
            f"Chunk #{idx} | Document: {result.document_name} | Cosine Similarity: {result.score:.4f}",
            expanded=(idx == 1),
        ):
            st.progress(score_pct / 100, text=f"Similarity Score: {result.score:.4f}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**Document Name:** `{result.document_name}`")
            with col2:
                st.caption(f"**Chunk Index:** `{result.chunk_index + 1}`")
            with col3:
                st.caption(f"**Character Offset:** `{result.start_char:,} – {result.end_char:,}`")

            st.markdown("---")
            st.text_area(
                f"Chunk Content #{idx}",
                value=result.text,
                height=130,
                disabled=True,
                key=f"chunk_text_{idx}",
            )


# -----------------------------------------------------------------------------
# PAGE 5: System Architecture & About
# -----------------------------------------------------------------------------
def render_about_page() -> None:
    st.markdown("## ℹ️ System Architecture & Project Documentation")
    st.write(
        "Comprehensive architectural breakdown designed for B.Tech project presentation and viva evaluation."
    )

    tab1, tab2, tab3 = st.tabs(["🏗️ RAG Architecture Flow", "🛠️ Technology Stack", "🎓 Viva Q&A Guide"])

    with tab1:
        st.subheader("Self-Checking Adaptive RAG Workflow")
        st.markdown(
            """
            ```
            [ User Input Question ]
                       │
                       ▼
             [ Query Analyzer ] ──── (Classifies Query: Simple vs. Complex)
                       │
                       ▼
             [ FAISS Retriever ] ─── (Sentence-Transformers MiniLM-L6 Embeddings)
                       │
                       ▼
            [ Evidence Checker ] ─── (LLM-as-a-Judge: Verification Score & Sufficiency)
                       │
             ┌─────────┴─────────┐
             │ Insufficient?     │ Sufficient?
             ▼                   ▼
     [ Adaptive Re-Retrieval ]  [ Answer Generator ] ── (Groq Llama-3.3-70B)
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
            [ Answer Checker ] ──── (Verifies Grounding & Hallucinations)
                       │
             ┌─────────┴─────────┐
             │ Unsupported?      │ Supported?
             ▼                   ▼
       [ Regenerate / Refuse ]  [ Final Grounded Output ]
            ```
            """
        )

    with tab2:
        st.subheader("Component Technology Stack")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                - **Language**: Python 3.11
                - **UI Framework**: Streamlit
                - **Vector Database**: FAISS (Facebook AI Similarity Search)
                - **Embedding Model**: `all-MiniLM-L6-v2` via `sentence-transformers`
                - **Embedding Dimensions**: 384-dimensional dense vectors
                """
            )
        with col2:
            st.markdown(
                """
                - **LLM Engine**: Groq API (`llama-3.3-70b-versatile`)
                - **Document Extractors**: `pypdf` for PDFs, UTF-8 text parser for TXT
                - **Evaluation Logic**: Semantic LLM JSON-Structured Verification
                - **Distance Metric**: L2-Normalized Inner Product (Cosine Similarity)
                """
            )

    with tab3:
        st.subheader("Key Academic Viva Concepts")

        with st.expander("Q1: What is the main problem Naive RAG faces, and how does CAR-RAG solve it?"):
            st.write(
                "Naive RAG passes retrieved context directly to an LLM without evaluating whether the context is "
                "relevant or complete. CAR-RAG (Corrective/Adaptive RAG) introduces pre-generation evidence checking "
                "and post-generation answer verification to eliminate hallucinations and trigger additional retrieval "
                "when initial evidence is deficient."
            )

        with st.expander("Q2: How does FAISS calculate vector similarity in this system?"):
            st.write(
                "Vectors generated by Sentence-Transformers are L2-normalized first. In L2-normalized vector space, "
                "the inner product (`IndexFlatIP`) is mathematically equivalent to cosine similarity, ranging from -1.0 to 1.0."
            )

        with st.expander("Q3: How are document chunks traced back to source files?"):
            st.write(
                "Each chunk maintains strict dataclass attributes: `document_name`, `chunk_index`, `start_char`, and "
                "`end_char`. This metadata is stored alongside the FAISS index in `indexes/metadata.json`."
            )


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Grounded CAR-RAG Builder",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    if "store" not in st.session_state:
        st.session_state.store = load_saved_store()
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    store = st.session_state.store
    page, top_k, chunk_size, chunk_overlap = render_sidebar(store)

    if page == "📌 Dashboard / Home":
        render_dashboard_page(store)
    elif page == "📂 Document Management":
        render_document_page(store, chunk_size, chunk_overlap)
    elif page == "💬 Ask Questions (RAG)":
        render_chat_page(store, top_k)
    elif page == "🔍 Evidence & Sources":
        render_evidence_page()
    elif page == "ℹ️ System & Architecture":
        render_about_page()


if __name__ == "__main__":
    main()
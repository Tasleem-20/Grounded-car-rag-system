from __future__ import annotations

import io
import json
import os
import shutil
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.answer_checker import AnswerChecker
from src.car_rag import CARRAG
from src.chunker import TextChunk
from src.config import Settings
from src.embeddings import EmbeddingService
from src.evidence_checker import EvidenceChecker
from src.generator import Generator
from src.groq_models import GroqModelService, format_groq_error
from src.image_embeddings import ImageEmbeddingService
from src.image_store import ImageRecord, ImageStore
from src.ingestion import (
    chunk_text,
    clean_text,
    detect_file_type,
    extract_pdf_visuals,
    internal_loader_filename,
    load_document,
)
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.sources import (
    filter_evidence_to_source,
    is_ambiguous_certificate_question,
    match_source_from_question,
    normalize_source_name,
    retrieval_mode_for_selection,
    safe_filename,
    same_source_name,
)
from src.vector_store import VectorStore

load_dotenv()

st.set_page_config(
    page_title="CAR-RAG Builder",
    page_icon="📄",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] {
                background: #111827;
                border-right: 1px solid #1f2937;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
                letter-spacing: 0.02em;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }
            h1 { font-size: 1.85rem !important; font-weight: 650 !important; }
            h2, h3 { letter-spacing: 0.01em; }
            div[data-testid="stMetric"] {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 0.75rem 1rem;
            }
            .file-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                padding: 0.7rem 0.95rem;
                margin-bottom: 0.45rem;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            .file-name { font-weight: 600; color: #f8fafc; }
            .file-type {
                font-size: 0.75rem;
                letter-spacing: 0.06em;
                color: #cbd5e1;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
            }
            .muted { color: #94a3b8; font-size: 0.95rem; }
            .status-ok { color: #86efac; }
            .status-warn { color: #fcd34d; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


DEFAULT_DATASET = {
    "files": [],
    "dataset_name": "No active dataset",
}

if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": 5,
        "retrieval_mode": "both",
        "show_debug": False,
    }

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "selected_source" not in st.session_state:
    st.session_state.selected_source = None


def get_data_dir() -> Path:
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def active_dataset_path() -> Path:
    return get_data_dir() / "active_dataset.json"


def persist_active_dataset(dataset: dict) -> None:
    active_dataset_path().write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_persisted_dataset() -> dict:
    for path in (active_dataset_path(), Path("indexes") / "active_dataset.json"):
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "files" in payload:
                    payload.setdefault("dataset_name", "Active dataset")
                    return payload
            except (OSError, json.JSONDecodeError):
                continue
    return DEFAULT_DATASET.copy()


if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = read_persisted_dataset()


def detect_uploaded_file_type(uploaded_file, content: bytes) -> str:
    """Detect the actual file type instead of trusting only the filename."""

    return detect_file_type(
        safe_filename(getattr(uploaded_file, "name", "")),
        content,
        getattr(uploaded_file, "type", "") or "",
    )


def get_active_file(source_name: str) -> dict | None:
    for file_info in st.session_state.active_dataset.get("files", []):
        if same_source_name(file_info["name"], source_name):
            return file_info
    return None


def get_active_file_type(source_name: str) -> str | None:
    file_info = get_active_file(source_name)
    if not file_info:
        return None
    return str(file_info.get("type", "")).upper()


def get_api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return str(secret)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


def debug_enabled() -> bool:
    return bool(st.session_state.settings.get("show_debug", False))


def show_error(message: str, error: Exception | None = None) -> None:
    st.error(message)
    if error is not None and debug_enabled():
        st.caption(str(error))


def render_file_row(name: str, file_type: str, prefix: str = "") -> None:
    label = f"{prefix}{name}" if prefix else name
    st.markdown(
        f'<div class="file-row"><span class="file-name">{escape(label)}</span>'
        f'<span class="file-type">{escape(file_type)}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource
def get_image_embedding_service() -> ImageEmbeddingService:
    return ImageEmbeddingService()


@st.cache_resource
def get_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


@st.cache_resource
def get_groq_service() -> GroqModelService | None:
    key = get_api_key()
    if not key:
        return None
    return GroqModelService(api_key=key)


@st.cache_resource
def get_generator() -> Generator:
    return Generator(api_key=get_api_key())


@st.cache_resource
def get_answer_checker() -> AnswerChecker:
    return AnswerChecker(api_key=get_api_key())


@st.cache_resource
def get_evidence_checker() -> EvidenceChecker:
    return EvidenceChecker(api_key=get_api_key())


def optional_generator() -> Generator | None:
    if not get_api_key():
        return None
    return get_generator()


def optional_answer_checker() -> AnswerChecker | None:
    if not get_api_key():
        return None
    return get_answer_checker()


def optional_evidence_checker() -> EvidenceChecker | None:
    if not get_api_key():
        return None
    return get_evidence_checker()


def get_text_index_path() -> Path:
    return get_data_dir() / "text.index"


def get_text_metadata_path() -> Path:
    return get_data_dir() / "text_metadata.json"


def get_image_index_path() -> Path:
    return get_data_dir() / "image.index"


def get_image_metadata_path() -> Path:
    return get_data_dir() / "image_metadata.json"


def get_images_dir() -> Path:
    path = get_data_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_images_staging_dir() -> Path:
    path = get_data_dir() / "images_staging"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


@st.cache_resource
def load_text_store() -> VectorStore | None:
    candidates = [
        (get_text_index_path(), get_text_metadata_path()),
        (get_data_dir() / "faiss.index", get_data_dir() / "metadata.json"),
        (Path("indexes") / "faiss.index", Path("indexes") / "metadata.json"),
        (Path("indexes") / "text.index", Path("indexes") / "text_metadata.json"),
    ]
    for index_path, metadata_path in candidates:
        if VectorStore.exists(index_path, metadata_path):
            try:
                return VectorStore.load(index_path, metadata_path)
            except Exception:
                continue
    return None


@st.cache_resource
def load_image_store() -> ImageStore | None:
    candidates = [
        (get_image_index_path(), get_image_metadata_path()),
        (get_data_dir() / "image.faiss.index", get_data_dir() / "image_metadata.json"),
        (Path("indexes") / "image.faiss.index", Path("indexes") / "image_metadata.json"),
        (Path("indexes") / "image.index", Path("indexes") / "image_metadata.json"),
    ]
    for index_path, metadata_path in candidates:
        if ImageStore.exists(index_path, metadata_path):
            try:
                return ImageStore.load(index_path, metadata_path)
            except Exception:
                continue
    return None


def caption_visual(
    image_path: Path,
    source_name: str,
    page_number: int | None = None,
) -> tuple[str, str | None]:
    """Caption an image. Failures become warnings, not upload failures."""

    from src.vision import fallback_caption

    groq_service = get_groq_service()
    if groq_service is None:
        return (
            fallback_caption(source_name, page_number),
            "Groq is not configured. Images were indexed without captions.",
        )

    try:
        from src.vision import caption_image as vision_caption

        return vision_caption(image_path, source_name=source_name), None
    except Exception as error:
        return fallback_caption(source_name, page_number), format_groq_error(error)


def image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            return image.width, image.height
    except Exception:
        return 0, 0


def process_uploads(uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Please upload at least one PDF, TXT, JPG, PNG, or WEBP file.")
        return

    settings = get_settings()
    embedding_service = get_embedding_service()
    image_embedding_service = get_image_embedding_service()

    text_chunks: list[TextChunk] = []
    image_records: list[ImageRecord] = []
    image_embeddings: list[list[float]] = []
    active_files: list[dict] = []
    caption_warnings: list[str] = []
    staging_dir: Path | None = None

    progress = st.progress(0)
    status = st.empty()
    prepared_files = []

    for uploaded_file in uploaded_files:
        name = safe_filename(uploaded_file.name)
        content = uploaded_file.getvalue()
        detected_type = detect_uploaded_file_type(uploaded_file, content)

        if detected_type == "unknown":
            st.error(
                f"Unsupported file type: **{name}**. "
                "Please upload PDF, TXT, PNG, JPG/JPEG, or WEBP."
            )
            continue

        prepared_files.append((name, content, detected_type))

    if not prepared_files:
        st.error("No supported files were found in the upload.")
        return

    total_files = len(prepared_files)
    final_images_dir = get_data_dir() / "images"

    try:
        staging_dir = get_images_staging_dir()

        for file_number, (name, content, detected_type) in enumerate(
            prepared_files, start=1
        ):
            status.write(f"Processing {file_number}/{total_files}: **{name}**")
            active_files.append(
                {
                    "name": name,
                    "type": detected_type.upper(),
                    "modality": (
                        "image" if detected_type in {"png", "jpg", "webp"} else "text"
                    ),
                }
            )

            if detected_type in {"png", "jpg", "webp"}:
                stored_name = Path(
                    internal_loader_filename(name, detected_type)
                ).name
                image_path = staging_dir / f"{file_number}_{stored_name}"
                image_path.write_bytes(content)

                caption, warning = caption_visual(image_path, name)
                if warning:
                    caption_warnings.append(f"{name}: {warning}")

                width, height = image_dimensions(image_path)
                record = ImageRecord(
                    image_id=f"{normalize_source_name(name)}_{file_number}",
                    modality="image",
                    document_name=name,
                    source_filename=name,
                    image_path=str(final_images_dir / image_path.name),
                    page_number=None,
                    caption=caption,
                    source_type="standalone_image",
                    width=width,
                    height=height,
                )
                image_records.append(record)

                try:
                    image_embeddings.append(
                        image_embedding_service.embed_images([image_path])[0]
                    )
                except Exception as error:
                    image_records.pop()
                    st.warning(f"Image embedding failed for {name}.")
                    if debug_enabled():
                        st.caption(str(error))

                progress.progress(file_number / total_files)
                continue

            loader_name = internal_loader_filename(name, detected_type)

            try:
                text = load_document(loader_name, io.BytesIO(content))
            except TypeError:
                try:
                    text = load_document(loader_name, content)
                except Exception as error:
                    raise RuntimeError(f"Unable to read {name}.") from error
            except Exception as error:
                raise RuntimeError(f"Unable to read {name}.") from error

            cleaned = clean_text(text)
            if cleaned.strip():
                chunks = chunk_text(
                    cleaned,
                    chunk_size=getattr(settings, "chunk_size", 800),
                    chunk_overlap=getattr(settings, "chunk_overlap", 120),
                )
                for chunk_index, chunk in enumerate(chunks):
                    text_chunks.append(
                        TextChunk(
                            document_name=name,
                            chunk_index=chunk_index,
                            text=chunk,
                            start_char=0,
                            end_char=len(chunk),
                        )
                    )

            if detected_type == "pdf":
                unique_prefix = (
                    f"{normalize_source_name(name).replace(' ', '_')}_{file_number}"
                )
                try:
                    pdf_visuals = extract_pdf_visuals(
                        io.BytesIO(content),
                        output_dir=staging_dir,
                        source_filename=name,
                        unique_prefix=unique_prefix,
                    )
                except Exception as error:
                    st.warning(f"PDF visual extraction failed for {name}.")
                    if debug_enabled():
                        st.caption(str(error))
                    pdf_visuals = []

                for visual_number, visual in enumerate(pdf_visuals, start=1):
                    if isinstance(visual, dict):
                        image_path = Path(
                            visual.get("image_path", visual.get("path", ""))
                        )
                        page_number = visual.get("page_number")
                        width = int(visual.get("width") or 0)
                        height = int(visual.get("height") or 0)
                    else:
                        image_path = Path(str(visual))
                        page_number = None
                        width, height = 0, 0

                    if not image_path.exists():
                        continue

                    caption, warning = caption_visual(
                        image_path,
                        name,
                        int(page_number) if page_number is not None else None,
                    )
                    if warning:
                        page_label = page_number or visual_number
                        caption_warnings.append(
                            f"{name} (page {page_label}): {warning}"
                        )

                    if not width or not height:
                        width, height = image_dimensions(image_path)

                    record = ImageRecord(
                        image_id=(
                            f"{normalize_source_name(name)}_page_"
                            f"{page_number or visual_number}_{file_number}"
                        ),
                        modality="image",
                        document_name=name,
                        source_filename=name,
                        image_path=str(final_images_dir / image_path.name),
                        page_number=(
                            int(page_number) if page_number is not None else None
                        ),
                        caption=caption,
                        source_type="pdf_visual",
                        width=width,
                        height=height,
                    )
                    image_records.append(record)

                    try:
                        image_embeddings.append(
                            image_embedding_service.embed_images([image_path])[0]
                        )
                    except Exception as error:
                        image_records.pop()
                        st.warning(
                            f"Image embedding failed for {name}, "
                            f"page {page_number or visual_number}."
                        )
                        if debug_enabled():
                            st.caption(str(error))

            progress.progress(file_number / total_files)

        status.write("Building search indexes...")

        if text_chunks:
            embeddings = embedding_service.embed_texts(
                [item.text for item in text_chunks]
            )
            text_store = VectorStore.from_embeddings(
                chunks=text_chunks,
                embeddings=embeddings,
            )
        else:
            text_store = None

        if image_records and image_embeddings:
            image_store = ImageStore.from_embeddings(
                records=image_records,
                embeddings=image_embeddings,
            )
        else:
            image_store = None

        final_images_dir.mkdir(parents=True, exist_ok=True)
        if staging_dir is not None and staging_dir.exists():
            for staged_file in staging_dir.iterdir():
                if staged_file.is_file():
                    shutil.copy2(staged_file, final_images_dir / staged_file.name)
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir = None

        if text_store is not None:
            text_store.save(get_text_index_path(), get_text_metadata_path())
        else:
            for path in (get_text_index_path(), get_text_metadata_path()):
                if path.exists():
                    path.unlink()

        if image_store is not None:
            image_store.save(get_image_index_path(), get_image_metadata_path())
        else:
            for path in (get_image_index_path(), get_image_metadata_path()):
                if path.exists():
                    path.unlink()

        load_text_store.clear()
        load_image_store.clear()

        dataset_types = list(
            dict.fromkeys(file_info["type"] for file_info in active_files)
        )
        dataset_name = (
            active_files[0]["name"]
            if len(active_files) == 1
            else f"{len(active_files)} files"
        )
        dataset = {
            "files": active_files,
            "dataset_name": dataset_name,
            "types": dataset_types,
        }
        st.session_state.active_dataset = dataset
        persist_active_dataset(dataset)
        st.session_state.selected_source = None
        st.session_state.last_result = None
    except Exception:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    status.empty()
    progress.empty()

    st.success(f"Active dataset updated successfully with {len(active_files)} file(s).")

    unique_warnings = list(dict.fromkeys(caption_warnings))
    for warning in unique_warnings[:6]:
        st.warning(warning)
    if len(unique_warnings) > 6:
        st.caption(f"{len(unique_warnings) - 6} additional captioning warning(s) omitted.")


def retrieval_mode_for_source(selected_source: str | None, search_all: bool) -> str:
    text_store = load_text_store()
    image_store = load_image_store()
    return retrieval_mode_for_selection(
        file_type=get_active_file_type(selected_source) if selected_source else None,
        search_all=search_all,
        default_mode=st.session_state.settings.get("retrieval_mode", "both"),
        has_text_index=text_store is not None,
        has_image_index=image_store is not None,
    )


def run_car_rag(
    question: str,
    selected_source: str | None = None,
    search_all: bool = False,
):
    text_store = load_text_store()
    image_store = load_image_store()
    analyzer = get_query_analyzer()
    analysis = analyzer.analyze(question)

    retrieval_mode = retrieval_mode_for_source(selected_source, search_all)
    prefer_images = analysis.needs_image_retrieval or retrieval_mode == "image"
    source_filter = None if search_all else selected_source

    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=get_embedding_service(),
        image_embeddings=get_image_embedding_service(),
    )

    top_k = int(st.session_state.settings.get("top_k", 5))

    evidence = retriever.retrieve(
        question=question,
        top_k=top_k,
        mode=retrieval_mode,
        prefer_images=prefer_images,
        source_filter=source_filter,
    )

    rag = CARRAG(
        analyzer=analyzer,
        retriever=retriever,
        groq_service=get_groq_service(),
        answer_checker=optional_answer_checker(),
        evidence_checker=optional_evidence_checker(),
        generator=optional_generator(),
    )

    result = rag.run(
        question=question,
        initial_evidence=evidence,
        query_analysis=analysis,
        source_filter=source_filter,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        prefer_images=prefer_images,
    )

    if source_filter and hasattr(result, "evidence"):
        result.evidence = filter_evidence_to_source(result.evidence, source_filter)

    return result


PAGES = [
    "Dashboard",
    "Documents",
    "Ask RAG",
    "Evidence",
    "Architecture",
    "Status",
    "Settings",
]


with st.sidebar:
    st.markdown("### CAR-RAG Builder")
    st.caption("Certificate-aware multimodal retrieval")
    st.divider()

    for page in PAGES:
        if st.button(
            page,
            use_container_width=True,
            type="primary" if st.session_state.page == page else "secondary",
            key=f"nav_{page}",
        ):
            st.session_state.page = page
            st.rerun()

    st.divider()
    files = st.session_state.active_dataset.get("files", [])
    st.caption(f"Active files: {len(files)}")
    groq_ready = bool(get_api_key())
    st.caption("Groq: configured" if groq_ready else "Groq: not configured")


if st.session_state.page == "Dashboard":
    st.title("CAR-RAG Builder")
    st.markdown(
        '<p class="muted">Certificate-aware multimodal retrieval with adaptive evidence checking.</p>',
        unsafe_allow_html=True,
    )

    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Files", len(active_files))
    searchable_items = 0
    if text_store:
        searchable_items += text_store.index.ntotal
    if image_store:
        searchable_items += image_store.index.ntotal
    col2.metric("Searchable Items", searchable_items)
    col3.metric("Embeddings", "MiniLM + CLIP")
    col4.metric("LLM", "Groq")

    st.subheader("Active Dataset")
    if not active_files:
        st.info("No active dataset. Upload files from Documents.")
    else:
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])

    st.subheader("System Status")
    left, right = st.columns(2)
    with left:
        if text_store:
            st.success("Text retrieval ready")
        else:
            st.info("Text retrieval not available")
        if image_store:
            st.success("Image retrieval ready")
        else:
            st.info("Image retrieval not available")
    with right:
        if get_api_key():
            st.success("Groq configured")
        else:
            st.warning("Groq API key not configured")
        if active_files:
            st.success("One Active Dataset loaded")
        else:
            st.info("Waiting for an Active Dataset")


elif st.session_state.page == "Documents":
    st.title("Documents")
    st.markdown(
        '<p class="muted">Upload a mixed batch to create one Active Dataset. '
        "PDF certificates remain a single source even when visual pages are indexed internally.</p>",
        unsafe_allow_html=True,
    )
    st.caption("Supported: PDF, TXT, PNG, JPG/JPEG, WEBP. Visible extensions are not required.")

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        key="document_uploader",
    )

    if uploaded_files:
        st.subheader("Detected files")
        for uploaded_file in uploaded_files:
            detected_type = detect_uploaded_file_type(
                uploaded_file,
                uploaded_file.getvalue(),
            )
            render_file_row(
                uploaded_file.name,
                detected_type.upper(),
                prefix="✓ ",
            )

        if st.button("Process & Build Indexes", type="primary", use_container_width=True):
            try:
                process_uploads(uploaded_files)
            except Exception as error:
                show_error("Processing failed. The previous Active Dataset was not replaced.", error)

    st.divider()
    st.subheader("Current Active Dataset")
    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.info("No files have been processed yet.")
    else:
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])
        st.caption("Extracted PDF page images are internal evidence only. They are not separate documents.")


elif st.session_state.page == "Ask RAG":
    st.title("Ask RAG")

    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.info("Upload and process a dataset first from Documents.")
    else:
        source_names = [file_info["name"] for file_info in active_files]
        selected_source: str | None = None

        search_all = st.checkbox(
            "Search all active files",
            value=False,
            key="search_all_toggle",
            help="When enabled, retrieval queries both text and image indexes across all active documents.",
        )

        st.subheader("Source")
        if len(source_names) == 1:
            selected_source = source_names[0]
            st.info(f"Active Document: **{selected_source}**")
        else:
            default_index = 0
            if st.session_state.selected_source in source_names:
                default_index = source_names.index(st.session_state.selected_source)

            selected_source = st.selectbox(
                "Select source",
                options=source_names,
                index=default_index,
                disabled=search_all,
                key="source_select",
            )
            st.session_state.selected_source = selected_source
            st.caption("Choose the original uploaded filename. Extracted PDF visual pages are internal evidence only.")

            if not search_all:
                st.info(
                    f"Targeting: **{selected_source}**. Retrieval will strictly isolate evidence to this certificate."
                )

        question = st.text_area(
            "Question",
            placeholder="Ask a question about your certificate or document...",
            height=120,
        )

        if st.button("Run CAR-RAG", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                inferred = match_source_from_question(question, source_names)
                source_for_query = None if search_all else (selected_source or inferred)

                if is_ambiguous_certificate_question(
                    question,
                    source_names,
                    source_for_query,
                    search_all,
                ) or (
                    not search_all
                    and len(source_names) > 1
                    and not source_for_query
                ):
                    st.warning(
                        "Multiple certificate documents are available. "
                        "Please select the certificate you want to query."
                    )
                else:
                    if inferred and not selected_source and not search_all:
                        st.caption(f"Matched source from the question: **{inferred}**")

                    with st.spinner(
                        "Query Analyzer → Retrieval → Evidence Check → Generator → Answer Check"
                    ):
                        try:
                            result = run_car_rag(
                                question=question,
                                selected_source=source_for_query,
                                search_all=search_all,
                            )
                            st.session_state.last_result = result
                        except Exception as error:
                            st.session_state.last_result = None
                            show_error(format_groq_error(error), error)

        result = st.session_state.last_result
        if result is not None:
            st.divider()
            st.subheader("Answer")

            if getattr(result, "error", None):
                st.error(result.error)
            elif result.answer:
                st.write(result.answer)
            else:
                st.info("No grounded answer was returned.")

            if not result.evidence_sufficient:
                st.info("The retrieved evidence was not sufficient for a grounded answer.")

            st.subheader("CAR-RAG Decision")
            analysis = result.query_analysis
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Query Type",
                str(getattr(analysis, "query_type", "unknown")).title(),
            )
            c2.metric(
                "Image Retrieval",
                "Yes" if getattr(analysis, "needs_image_retrieval", False) else "No",
            )
            c3.metric("Evidence Count", len(result.evidence or []))
            
            grounding_label = "Verified" if result.grounding_verified else ("Insufficient" if not result.evidence_sufficient else "Not Grounded")
            c4.metric("Grounding", grounding_label)

            if getattr(result, "evidence_reason", ""):
                st.caption(f"**Evidence check:** {result.evidence_reason}")
            if getattr(result, "answer_reason", ""):
                st.caption(f"**Answer check:** {result.answer_reason}")
            if getattr(result, "re_retrieved", False):
                st.caption("Adaptive re-retrieval ran for this question.")

            evidence = result.evidence or []
            if evidence:
                st.subheader("Retrieved Evidence")
                for number, item in enumerate(evidence, start=1):
                    with st.expander(f"{number}. {item.display_label}  ·  Score: {item.score:.3f}"):
                        if item.modality == "image":
                            page_text = f" — page {item.page_number}" if item.page_number is not None else ""
                            st.caption(f"**Source Document:** {item.document_name}{page_text} (Visual Evidence)")
                        else:
                            st.caption(f"**Source Document:** {item.document_name} (Text Evidence)")
                        
                        st.write(item.caption or item.text)
                        if item.image_path and Path(item.image_path).exists():
                            st.image(item.image_path, use_container_width=True)


elif st.session_state.page == "Evidence":
    st.title("Evidence")
    st.markdown(
        '<p class="muted">Text and image tabs are evidence modalities from the same Active Dataset, not separate stores.</p>',
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result
    if result is None:
        st.info("Run a CAR-RAG question first.")
    else:
        evidence = getattr(result, "evidence", None) or []
        if not evidence:
            st.warning("No evidence was returned.")
        else:
            text_evidence = [item for item in evidence if item.modality == "text"]
            image_evidence = [item for item in evidence if item.modality == "image"]
            text_tab, image_tab = st.tabs(
                [f"Text ({len(text_evidence)})", f"Images ({len(image_evidence)})"]
            )

            with text_tab:
                if not text_evidence:
                    st.info("No text evidence for this answer.")
                for item in text_evidence:
                    st.markdown(f"**{item.document_name}**")
                    st.caption(f"Score: {item.score:.3f}")
                    st.write(item.text)
                    st.divider()

            with image_tab:
                if not image_evidence:
                    st.info("No image evidence for this answer.")
                for item in image_evidence:
                    page = (
                        f" · page {item.page_number}"
                        if item.page_number is not None
                        else ""
                    )
                    st.markdown(f"**{item.document_name}**{page}")
                    st.caption(f"Score: {item.score:.3f} · {item.source_type}")
                    if item.image_path and Path(item.image_path).exists():
                        st.image(item.image_path, use_container_width=True)
                    if item.caption:
                        st.write(item.caption)
                    st.divider()


elif st.session_state.page == "Architecture":
    st.title("Architecture")
    st.markdown(
        """
One **Active Dataset** is what you upload and query. Separate FAISS indexes
are an internal implementation detail.

```text
ONE ACTIVE DATASET
        ↓
Automatic file detection
        ↓
 ┌───────────────┬─────────────────┐
 │ Text          │ Images/Visuals  │
 │ MiniLM        │ Vision/CLIP     │
 │ FAISS         │ FAISS           │
 └───────────────┴─────────────────┘
        ↓
Multimodal Retrieval
        ↓
CAR-RAG
  Query Analyzer
  Initial Retrieval
  Adaptive Retrieval
  Evidence Checker
  Re-retrieval when needed
  Generator
  Answer Checker
        ↓
Grounded Answer / Refusal
```
        """
    )
    st.info(
        "A PDF certificate remains one logical source. Visual pages are extracted "
        "internally so retrieval can use both text and image evidence. Those extracted "
        "images are never selectable as their own documents."
    )


elif st.session_state.page == "Status":
    st.title("Status")
    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Files", len(active_files))
    c2.metric("Text chunks", text_store.index.ntotal if text_store else 0)
    c3.metric("Image items", image_store.index.ntotal if image_store else 0)

    if text_store:
        st.success(f"Text index ready — {text_store.index.ntotal} searchable chunks")
    else:
        st.info("Text index unavailable")

    if image_store:
        st.success(f"Image index ready — {image_store.index.ntotal} searchable images")
    else:
        st.info("Image index unavailable")

    if get_api_key():
        st.success("Groq configured")
    else:
        st.warning("GROQ_API_KEY is not configured")

    if active_files:
        st.subheader("Active files")
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])


elif st.session_state.page == "Settings":
    st.title("Settings")
    st.session_state.settings["top_k"] = st.slider(
        "Retrieval Top-K",
        min_value=1,
        max_value=15,
        value=int(st.session_state.settings.get("top_k", 5)),
        help="How many items to retrieve from each internal index before source filtering.",
    )
    st.session_state.settings["retrieval_mode"] = st.selectbox(
        "Default Retrieval Mode",
        options=["both", "text", "image"],
        index=["both", "text", "image"].index(
            st.session_state.settings.get("retrieval_mode", "both")
        ),
        help="Used when a source type is unknown. Search all active files uses every available index in the Active Dataset. A selected PDF always uses both text and visual retrieval.",
    )
    st.session_state.settings["show_debug"] = st.checkbox(
        "Show technical error details",
        value=bool(st.session_state.settings.get("show_debug", False)),
    )
    st.info(
        "Selecting a PDF certificate automatically uses both text and visual retrieval. "
        "Standalone images use image retrieval. TXT files use text retrieval."
    )
    st.caption("Models: text reasoning uses openai/gpt-oss-120b. Vision and judging use qwen/qwen3.8-27b.")

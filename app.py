"""CAR-RAG Builder: Multimodal document, text, and image retrieval application with integrated YOLO26n object detection."""

from __future__ import annotations

import io
import json
import os
import shutil
from collections import Counter
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.answer_checker import AnswerChecker
from src.car_rag import CARRAG, CARRAGResult
from src.chunker import TextChunk
from src.config import Settings
from src.deterministic_detection import (
    answer_detection_query,
    is_direct_object_detection_query,
)
from src.embeddings import EmbeddingService
from src.evidence import RetrievedEvidence
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
from src.object_detection import ObjectDetector
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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CSS & EXACT PIXEL-CLOSE STYLING
# ==============================================================================

def inject_css() -> None:
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after {
    box-sizing: border-box !important;
}

/* Global resets & fixed canvas background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="stMain"], section.main {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: #070e1c !important;
    color: #f8fafc !important;
    -webkit-font-smoothing: antialiased;
    max-width: 100vw !important;
}

/* Hide visible scrollbars globally on main view while preserving mouse/touch/keyboard scrolling */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
section.main,
.main,
.block-container {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

html::-webkit-scrollbar,
body::-webkit-scrollbar,
[data-testid="stAppViewContainer"]::-webkit-scrollbar,
[data-testid="stApp"]::-webkit-scrollbar,
[data-testid="stMain"]::-webkit-scrollbar,
section.main::-webkit-scrollbar,
.main::-webkit-scrollbar,
.block-container::-webkit-scrollbar {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #070e1c !important;
}

/* Hide default Streamlit header bar & collapse button to prevent layout shifts */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
    min-height: 0px !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[data-testid="baseButton-header"],
#MainMenu, footer {
    visibility: hidden !important;
    display: none !important;
}

/* 25% FIXED SIDEBAR (width: 25vw; min: 260px, max: 320px; height: 100vh; no border-right) */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {
    width: 25vw !important;
    min-width: 260px !important;
    max-width: 320px !important;
    height: 100vh !important;
    flex-shrink: 0 !important;
    background-color: #0b152d !important;
    border-right: none !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;
    z-index: 9999 !important;
    overflow-y: auto !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
    box-sizing: border-box !important;
}

/* Hide visible scrollbars on sidebar */
section[data-testid="stSidebar"]::-webkit-scrollbar,
[data-testid="stSidebar"]::-webkit-scrollbar,
[data-testid="stSidebar"] > div::-webkit-scrollbar,
[data-testid="stSidebarContent"]::-webkit-scrollbar,
[data-testid="stSidebarUserContent"]::-webkit-scrollbar {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    padding: 1.5rem 20px !important;
    overflow-y: auto !important;
    box-sizing: border-box !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* 75% MAIN CONTENT CONTAINER (starts strictly AFTER 25vw sidebar) */
section.main,
[data-testid="stMain"],
.stMain,
[data-testid="stAppViewContainer"] > section:nth-of-type(2) {
    margin-left: 25vw !important;
    width: 75vw !important;
    max-width: 75vw !important;
    min-height: 100vh !important;
    background-color: #070e1c !important;
    box-sizing: border-box !important;
    position: relative !important;
    left: 0 !important;
}

@media (max-width: 1040px) {
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        width: 260px !important;
    }
    section.main, [data-testid="stMain"], .stMain {
        margin-left: 260px !important;
        width: calc(100% - 260px) !important;
        max-width: calc(100% - 260px) !important;
    }
}

@media (min-width: 1280px) {
    section[data-testid="stSidebar"], [data-testid="stSidebar"] {
        width: 320px !important;
    }
    section.main, [data-testid="stMain"], .stMain {
        margin-left: 320px !important;
        width: calc(100% - 320px) !important;
        max-width: calc(100% - 320px) !important;
    }
}

/* Main Content inside 75% area: 50px left/right padding */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 50px !important;
    padding-right: 50px !important;
    max-width: 100% !important;
    width: 100% !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

/* Brand area */
.app-brand-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

.app-brand-icon {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    border-radius: 8px;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    font-weight: 700;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
    flex-shrink: 0;
}

.app-brand-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
    line-height: 1.2;
    white-space: nowrap !important;
}

.app-brand-caption {
    font-size: 0.74rem;
    color: #7c8ba1;
    line-height: 1.25;
    margin-top: 0.15rem;
}

.nav-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 1.25rem;
    margin-bottom: 0.4rem;
    padding-left: 0.1rem;
}

/* Sidebar navigation buttons: width ≈ 270–285px, height ≈ 40–42px */
[data-testid="stSidebar"] div.stButton {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebar"] div.stButton > button {
    background-color: #0b1120 !important;
    border: 1px solid #141e30 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    padding: 0 0.85rem !important;
    text-align: left !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: none !important;
    margin-bottom: 0.25rem !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] div.stButton > button p,
[data-testid="stSidebar"] div.stButton > button div[data-testid="stMarkdownContainer"] p {
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
    color: #94a3b8 !important;
    text-align: left !important;
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    border-color: #2563eb !important;
    background-color: #101c36 !important;
}

[data-testid="stSidebar"] div.stButton > button:hover p {
    color: #ffffff !important;
}

/* Active Sidebar button: Bright Blue style */
[data-testid="stSidebar"] div.stButton > button[kind="primary"],
[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: 1px solid #3b82f6 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    padding: 0 0.85rem !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4) !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] p,
[data-testid="stSidebar"] div.stButton > button[kind="primary"] div[data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.sidebar-footer {
    margin-top: 2.2rem;
    padding-top: 0.85rem;
    border-top: 1px solid #131c2e;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

.sidebar-status-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #10b981;
}

/* Main typography & headers */
.badge-pill {
    display: inline-block;
    background: #0c1e3d;
    color: #38bdf8;
    border: 1px solid #1e40af;
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

h1 {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.015em !important;
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
    line-height: 1.2 !important;
}

h2, h3, .stSubheader {
    font-size: 1.3rem !important;
    font-weight: 650 !important;
    color: #ffffff !important;
    letter-spacing: -0.005em !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.75rem !important;
}

.page-subtitle {
    font-size: 0.95rem;
    color: #7c8ba1;
    margin-bottom: 0.85rem;
    line-height: 1.45;
}

hr.dashboard-divider {
    border: none !important;
    border-top: 1px solid #131c2e !important;
    margin: 0.85rem 0 1.5rem 0 !important;
}

.format-caption {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: -0.5rem;
    margin-bottom: 1.15rem;
    line-height: 1.45;
}

.format-caption strong {
    color: #f1f5f9;
}

/* 4 Metric Cards in one row across 75% width */
div[data-testid="stMetric"] {
    background: #0b1325 !important;
    border: 1px solid #162238 !important;
    border-radius: 12px !important;
    padding: 1.15rem 1.35rem !important;
    min-height: 98px !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    border-color: #2563eb !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.35rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1.15 !important;
}

/* Active Dataset Banner: Full width */
.info-banner {
    background: #0d1e3d;
    border: 1px solid #1e3a6d;
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    color: #60a5fa;
    font-size: 0.94rem;
    font-weight: 500;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0.5rem 0 1.25rem 0;
    line-height: 1.45;
    box-sizing: border-box;
}

/* System Status Cards: 2-column layout filling available width */
.status-card {
    background: #0b1325;
    border: 1px solid #162238;
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.85rem;
    width: 100% !important;
    max-width: 100% !important;
    transition: border-color 0.2s ease;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    box-sizing: border-box;
}

.status-card:hover {
    border-color: #2563eb;
}

.status-card-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-dot-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.status-dot-indicator.green {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.7);
}

.status-dot-indicator.amber {
    background: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.7);
}

.status-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.18rem;
}

.status-card-desc {
    font-size: 0.8rem;
    color: #64748b;
}

.status-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.24rem 0.65rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.status-badge.connected,
.status-badge.ready,
.status-badge.active {
    background: #06251b;
    color: #34d399;
    border: 1px solid #065f46;
}

.status-badge.unavailable,
.status-badge.nodata,
.status-badge.missing {
    background: #25180c;
    color: #f59e0b;
    border: 1px solid #78350f;
}

/* File Row */
.file-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1.15rem;
    margin-bottom: 0.5rem;
    background: #0b1325;
    border: 1px solid #162238;
    border-radius: 10px;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
    transition: border-color 0.2s ease;
}

.file-row:hover {
    border-color: #2563eb;
}

.file-name { font-weight: 600; color: #f8fafc; font-size: 0.92rem; }
.file-type {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #cbd5e1;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
}

/* Architecture Pipeline */
.arch-pipeline-container {
    background: #0b1325;
    border: 1px solid #162238;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

.arch-pipeline-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.35rem;
}

.arch-pipeline-desc {
    font-size: 0.86rem;
    color: #94a3b8;
    line-height: 1.45;
    margin-bottom: 1.25rem;
}

.arch-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

@media (max-width: 850px) {
    .arch-grid {
        grid-template-columns: repeat(1, 1fr);
    }
}

.arch-card {
    background: #070d1a;
    border: 1px solid #141f33;
    border-radius: 10px;
    padding: 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    transition: border-color 0.2s ease;
}

.arch-card:hover {
    border-color: #2563eb;
}

.arch-card-icon {
    font-size: 1.3rem;
    margin-bottom: 0.1rem;
}

.arch-card-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #f8fafc;
}

.arch-card-body {
    font-size: 0.79rem;
    color: #94a3b8;
    line-height: 1.45;
}

.arch-decision-box {
    background: #0d1e3d;
    border: 1px solid #1e3a6d;
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    color: #60a5fa;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 1.25rem;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

/* Primary Action Buttons across all pages */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: 1px solid #3b82f6 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.5) !important;
}

/* Textarea and Inputs */
div[data-baseweb="textarea"] {
    background-color: #0b1325 !important;
    border: 1px solid #162238 !important;
    border-radius: 10px !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

div[data-baseweb="textarea"] textarea {
    color: #f8fafc !important;
    font-size: 0.95rem !important;
}

div[data-baseweb="select"] > div {
    background-color: #0b1325 !important;
    border: 1px solid #162238 !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

div[data-testid="stFileUploader"] section {
    background: #0b1325 !important;
    border: 1px dashed #1e3357 !important;
    border-radius: 10px !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
}

hr {
    border-color: #131c2e !important;
    margin: 1.5rem 0 !important;
}
</style>""",
        unsafe_allow_html=True,
    )


inject_css()


# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================

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


OBJECT_EMOJIS = {
    "car": "🚗",
    "truck": "🚚",
    "bus": "🚌",
    "motorcycle": "🏍️",
    "bicycle": "🚲",
    "person": "🧍",
    "traffic light": "🚦",
    "stop sign": "🛑",
    "dog": "🐕",
    "cat": "🐈",
    "backpack": "🎒",
    "chair": "🪑",
    "cell phone": "📱",
    "laptop": "💻",
}


def get_object_emoji(label: str) -> str:
    return OBJECT_EMOJIS.get(label.lower().strip(), "🎯")


# ==============================================================================
# CACHED SERVICES & DETECTOR
# ==============================================================================

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


@st.cache_resource
def get_object_detector() -> ObjectDetector:
    return ObjectDetector(model_name="yolo26n.pt")


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


# ==============================================================================
# INDEX PATHS & STORES
# ==============================================================================

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


# ==============================================================================
# DATASET INGESTION
# ==============================================================================

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
    final_images_dir = get_images_dir()

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

                # Extract OCR text for semantic image understanding
                try:
                    from src.ocr import extract_ocr_text
                    ocr_text = extract_ocr_text(image_path)
                except Exception as ocr_err:
                    ocr_text = ""
                    if debug_enabled():
                        st.caption(f"OCR extraction warning for {name}: {ocr_err}")

                width, height = image_dimensions(image_path)

                # Run YOLO26n object detection during document ingestion
                try:
                    detector = get_object_detector()
                    det_result = detector.detect(image_path=image_path)
                    det_list = [d.to_dict() for d in det_result.detections]
                    det_summary = det_result.summary
                    det_counts = det_result.counts
                    det_context = det_result.rag_context
                except Exception as det_err:
                    det_list = []
                    det_summary = ""
                    det_counts = {}
                    det_context = ""
                    if debug_enabled():
                        st.caption(f"YOLO26n detection warning for {name}: {det_err}")

                record = ImageRecord(
                    image_id=f"{normalize_source_name(name)}_{file_number}",
                    modality="image",
                    document_name=name,
                    source_filename=name,
                    image_path=str(final_images_dir / image_path.name),
                    page_number=None,
                    caption=caption,
                    ocr_text=ocr_text,
                    source_type="standalone_image",
                    width=width,
                    height=height,
                    detections=det_list,
                    detection_summary=det_summary,
                    detection_counts=det_counts,
                    detection_context=det_context,
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

                    # Extract OCR text for semantic visual page understanding
                    try:
                        from src.ocr import extract_ocr_text
                        ocr_text = extract_ocr_text(image_path)
                    except Exception as ocr_err:
                        ocr_text = ""
                        if debug_enabled():
                            st.caption(f"OCR extraction warning for {name} (page {page_number}): {ocr_err}")

                    if not width or not height:
                        width, height = image_dimensions(image_path)

                    # Run YOLO26n object detection on PDF visual asset
                    try:
                        detector = get_object_detector()
                        det_result = detector.detect(image_path=image_path)
                        det_list = [d.to_dict() for d in det_result.detections]
                        det_summary = det_result.summary
                        det_counts = det_result.counts
                        det_context = det_result.rag_context
                    except Exception as det_err:
                        det_list = []
                        det_summary = ""
                        det_counts = {}
                        det_context = ""
                        if debug_enabled():
                            st.caption(f"YOLO26n detection warning for {name} (page {page_number}): {det_err}")

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
                        ocr_text=ocr_text,
                        source_type="pdf_visual",
                        width=width,
                        height=height,
                        detections=det_list,
                        detection_summary=det_summary,
                        detection_counts=det_counts,
                        detection_context=det_context,
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
    if unique_warnings:
        rate_limits = [w for w in unique_warnings if "rate limit" in w.lower()]
        other_warnings = [w for w in unique_warnings if "rate limit" not in w.lower()]
        for warning in other_warnings[:4]:
            st.warning(warning)
        if rate_limits and debug_enabled():
            for w in rate_limits[:3]:
                st.caption(f"Note: {w}")


# ==============================================================================
# RETRIEVAL & CAR-RAG PIPELINE
# ==============================================================================

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


def run_car_rag_pipeline(
    question: str,
    selected_source: str | None = None,
    search_all: bool = False,
    extra_evidence: list[RetrievedEvidence] | None = None,
) -> CARRAGResult:
    """Execute the CAR-RAG pipeline combining indexed documents and any extra evidence."""
    text_store = load_text_store()
    image_store = load_image_store()
    analyzer = get_query_analyzer()
    analysis = analyzer.analyze(question)

    retrieval_mode = retrieval_mode_for_source(selected_source, search_all)
    prefer_images = analysis.needs_image_retrieval or retrieval_mode == "image"
    include_detection = (
        analysis.needs_detection_evidence
        or analysis.query_mode in {"OBJECT_DETECTION", "MULTIMODAL_RAG"}
    )
    source_filter = None if search_all else selected_source

    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=get_embedding_service(),
        image_embeddings=get_image_embedding_service(),
    )

    top_k = int(st.session_state.settings.get("top_k", 5))

    mode_to_use = retrieval_mode
    if retrieval_mode == "both":
        if analysis.query_mode == "TEXT_RAG":
            mode_to_use = "text"
        elif analysis.query_mode in {"IMAGE_RAG", "OBJECT_DETECTION"}:
            mode_to_use = "image"
        else:
            mode_to_use = "both"

    retrieved_items: list[RetrievedEvidence] = []
    if text_store is not None or image_store is not None:
        retrieved_items = list(
            retriever.retrieve(
                question=question,
                top_k=top_k,
                mode=mode_to_use,
                prefer_images=prefer_images,
                source_filter=source_filter,
                include_detection=include_detection,
            )
        )

    # Prepend any extra evidence
    combined_evidence: list[RetrievedEvidence] = []
    if extra_evidence:
        combined_evidence.extend(extra_evidence)
    combined_evidence.extend(retrieved_items)

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
        initial_evidence=combined_evidence,
        query_analysis=analysis,
        source_filter=source_filter,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        prefer_images=prefer_images,
    )

    if source_filter and hasattr(result, "evidence") and result.evidence:
        allowed_evidence = []
        for ev in result.evidence:
            if ev.source_type == "object_detection" or same_source_name(ev.document_name, source_filter):
                allowed_evidence.append(ev)
        result.evidence = allowed_evidence

    return result


# ==============================================================================
# SIDEBAR NAVIGATION (MATCHING SCREENSHOT)
# ==============================================================================

NAV_SECTIONS = [
    (
        "MAIN",
        [
            ("Dashboard", "📊"),
            ("Documents", "📁"),
            ("Ask RAG", "⚡"),
        ],
    ),
    (
        "ANALYSIS",
        [
            ("Evidence", "📄"),
            ("Architecture", "🏗"),
            ("Status", "🟢"),
        ],
    ),
    (
        "SYSTEM",
        [
            ("Settings", "⚙"),
        ],
    ),
]

with st.sidebar:
    st.markdown(
        """<div class="app-brand-container">
    <div class="app-brand-icon">⚡</div>
    <div>
        <div class="app-brand-title">CAR-RAG Builder</div>
        <div class="app-brand-caption">Multimodal AI Retrieval System</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    for section_label, items in NAV_SECTIONS:
        st.markdown(f'<div class="nav-section-label">{section_label}</div>', unsafe_allow_html=True)
        for page_name, icon in items:
            is_active = (st.session_state.page == page_name)
            if st.button(
                f"{icon}  {page_name}",
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_name
                st.rerun()

    st.markdown(
        """<div class="sidebar-footer">
    <div class="sidebar-status-row">🟢 CAR-RAG Ready</div>
</div>""",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: DASHBOARD (EXACT PIXEL-MATCH TO SCREENSHOT)
# ==============================================================================

if st.session_state.page == "Dashboard":
    st.markdown('<div class="badge-pill">SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("<h1>Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real-time multimodal indexing, vector store diagnostics, and active dataset status.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="dashboard-divider" />', unsafe_allow_html=True)

    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ACTIVE FILES", len(active_files))
    searchable_items = 0
    if text_store:
        searchable_items += text_store.index.ntotal
    if image_store:
        searchable_items += image_store.index.ntotal
    col2.metric("SEARCHABLE ITEMS", searchable_items)
    col3.metric("EMBEDDINGS", "MiniLM + CLIP")
    col4.metric("LLM ENGINE", "Groq")

    st.markdown("<h2>Active Dataset</h2>", unsafe_allow_html=True)
    if not active_files:
        st.markdown(
            '<div class="info-banner">No active dataset loaded. Please navigate to Documents to upload source files.</div>',
            unsafe_allow_html=True,
        )
    else:
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])

    st.markdown("<h2>System Status</h2>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        # Card 1: Text Retrieval Index
        t_online = text_store is not None
        t_title = "Text Retrieval Index"
        t_desc = f"Dense Text FAISS ready ({text_store.index.ntotal} chunks)" if t_online else "Index offline"
        t_dot = "green" if t_online else "amber"
        t_badge_class = "ready" if t_online else "unavailable"
        t_badge_text = "Ready" if t_online else "Unavailable"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {t_dot}"></div>
        <div>
            <div class="status-card-title">{t_title}</div>
            <div class="status-card-desc">{t_desc}</div>
        </div>
    </div>
    <span class="status-badge {t_badge_class}">{t_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

        # Card 3: Image Retrieval Index
        i_online = image_store is not None
        i_title = "Image Retrieval Index"
        i_desc = f"Index online (CLIP + YOLO26n - {image_store.index.ntotal} items)" if i_online else "Index offline"
        i_dot = "green" if i_online else "amber"
        i_badge_class = "ready" if i_online else "unavailable"
        i_badge_text = "Ready" if i_online else "Unavailable"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {i_dot}"></div>
        <div>
            <div class="status-card-title">{i_title}</div>
            <div class="status-card-desc">{i_desc}</div>
        </div>
    </div>
    <span class="status-badge {i_badge_class}">{i_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

    with right_col:
        # Card 2: Groq LLM Connection
        g_ready = bool(get_api_key())
        g_title = "Groq LLM Connection"
        g_desc = "API key configured & ready" if g_ready else "API key missing"
        g_dot = "green" if g_ready else "amber"
        g_badge_class = "connected" if g_ready else "missing"
        g_badge_text = "Connected" if g_ready else "Unavailable"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {g_dot}"></div>
        <div>
            <div class="status-card-title">{g_title}</div>
            <div class="status-card-desc">{g_desc}</div>
        </div>
    </div>
    <span class="status-badge {g_badge_class}">{g_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

        # Card 4: Dataset State
        d_count = len(active_files)
        d_title = "Dataset State"
        d_desc = f"{d_count} file(s) indexed" if d_count > 0 else "Waiting for dataset"
        d_dot = "green" if d_count > 0 else "amber"
        d_badge_class = "active" if d_count > 0 else "nodata"
        d_badge_text = "Active" if d_count > 0 else "No Data"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {d_dot}"></div>
        <div>
            <div class="status-card-title">{d_title}</div>
            <div class="status-card-desc">{d_desc}</div>
        </div>
    </div>
    <span class="status-badge {d_badge_class}">{d_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: DOCUMENTS (EXACT PIXEL-MATCH TO SCREENSHOT)
# ==============================================================================

elif st.session_state.page == "Documents":
    st.markdown('<div class="badge-pill">DATA MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown("<h1>Documents</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload mixed documents and images to build unified text and visual searchable FAISS indexes.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="format-caption"><strong>Supported formats:</strong> PDF, TXT, PNG, JPG, JPEG, WEBP. PDF visual pages are automatically extracted and indexed internally as visual evidence.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        key="document_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown("<h2>Detected files</h2>", unsafe_allow_html=True)
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

    st.markdown("<h2>Current Active Dataset</h2>", unsafe_allow_html=True)
    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.markdown(
            '<div class="info-banner">No active files have been processed yet. Upload documents above to begin.</div>',
            unsafe_allow_html=True,
        )
    else:
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])
        st.caption("Extracted PDF page images are internal evidence only. They are not separate documents.")


# ==============================================================================
# PAGE 3: ASK RAG (EXACT PIXEL-MATCH TO SCREENSHOT)
# ==============================================================================

elif st.session_state.page == "Ask RAG":
    st.markdown('<div class="badge-pill">MULTIMODAL Q&A</div>', unsafe_allow_html=True)
    st.markdown("<h1>Ask RAG</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Multimodal question answering powered by adaptive retrieval, vision grounding, and object verification.</div>',
        unsafe_allow_html=True,
    )

    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.markdown(
            '<div class="info-banner">Please upload and process a dataset from Documents before querying.</div>',
            unsafe_allow_html=True,
        )
    else:
        source_names = [file_info["name"] for file_info in active_files]
        selected_source: str | None = None

        search_all = st.checkbox(
            "Search all active files",
            value=False,
            key="search_all_toggle",
            help="When enabled, retrieval queries both text and image indexes across all active documents.",
        )

        st.markdown("<h2>Source</h2>", unsafe_allow_html=True)
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
                    f"Targeting: **{selected_source}**. Retrieval will strictly isolate evidence to this document."
                )

        question = st.text_area(
            "Question",
            placeholder="Ask a question about your document, text, or image...",
            height=120,
        )

        if st.button("⚡ Ask CAR-RAG", type="primary", use_container_width=True):
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
                        "Multiple documents are available. "
                        "Please select the document you want to query."
                    )
                else:
                    if inferred and not selected_source and not search_all:
                        st.caption(f"Matched source from the question: **{inferred}**")

                    with st.spinner(
                        "Query Analyzer → Retrieval → Evidence Check → Generator → Answer Check"
                    ):
                        try:
                            result = run_car_rag_pipeline(
                                question=question.strip(),
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
            st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)

            if getattr(result, "error", None):
                st.error(result.error)
            elif result.answer:
                st.write(result.answer)
            else:
                st.info("No grounded answer was returned.")

            if not result.evidence_sufficient:
                st.info("The retrieved evidence was not sufficient for a grounded answer.")

            # Compact Object Detection Evidence (shown only when object detection evidence is actually relevant/used)
            evidence_list = getattr(result, "evidence", []) or []
            detection_evidence_items = [
                item for item in evidence_list if item.source_type == "object_detection"
            ]
            is_object_query = (
                getattr(result.query_analysis, "query_mode", "") == "OBJECT_DETECTION"
                or getattr(result.query_analysis, "needs_detection_evidence", False)
            )

            if detection_evidence_items:
                st.markdown("<h2>Object Detection Evidence</h2>", unsafe_allow_html=True)
                for det_ev in detection_evidence_items:
                    extra = getattr(det_ev, "extra", {}) or {}
                    counts = extra.get("detection_counts", {})
                    if not counts and extra.get("detections"):
                        counts = dict(Counter(d.get("label", "") for d in extra.get("detections", []) if d.get("label")))

                    if counts:
                        summary_cols = st.columns(min(4, max(1, len(counts))))
                        for col, (label, count) in zip(summary_cols, sorted(counts.items())):
                            emoji = get_object_emoji(label)
                            col.metric(f"{emoji} {label.title()}", count)

                    detections = extra.get("detections", [])
                    if detections:
                        with st.expander(f"▼ Detection Details & Bounding Boxes ({det_ev.document_name})", expanded=False):
                            for idx, det in enumerate(detections, start=1):
                                lbl = det.get("label", "object")
                                conf = float(det.get("confidence", 0.0))
                                x1 = float(det.get("x1", 0.0))
                                y1 = float(det.get("y1", 0.0))
                                x2 = float(det.get("x2", 0.0))
                                y2 = float(det.get("y2", 0.0))
                                emoji = get_object_emoji(lbl)
                                st.write(
                                    f"**{idx}. {emoji} {lbl.title()}** — "
                                    f"Confidence: `{conf:.1%}` | "
                                    f"Bounding Box: `({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})`"
                                )
                    elif not counts:
                        st.write("No supported objects were detected in this indexed image.")
            elif is_object_query:
                st.markdown("<h2>Object Detection Evidence</h2>", unsafe_allow_html=True)
                st.write("No supported objects were detected in this indexed image.")

            st.markdown("<h2>CAR-RAG Decision</h2>", unsafe_allow_html=True)
            analysis = result.query_analysis
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Query Type",
                str(getattr(analysis, "query_type", "unknown")).title(),
            )
            c2.metric(
                "Image Retrieval",
                "Yes" if (getattr(analysis, "needs_image_retrieval", False) or any(item.modality == "image" for item in (result.evidence or []))) else "No",
            )
            c3.metric("Evidence Count", len(result.evidence or []))
            
            grounding_label = "Grounded" if result.grounding_verified else ("Insufficient" if not result.evidence_sufficient else "Not Grounded")
            c4.metric("Grounding", grounding_label)

            if getattr(result, "evidence_reason", ""):
                st.caption(f"**Evidence check:** {result.evidence_reason}")
            if getattr(result, "answer_reason", ""):
                st.caption(f"**Answer check:** {result.answer_reason}")
            if getattr(result, "re_retrieved", False):
                st.caption("Adaptive re-retrieval ran for this question.")

            evidence = result.evidence or []
            if evidence:
                st.markdown("<h2>Retrieved Evidence</h2>", unsafe_allow_html=True)
                for number, item in enumerate(evidence, start=1):
                    label = item.display_label
                    if item.source_type == "object_detection":
                        label = f"🎯 {item.document_name} — YOLO26n Object Detection Evidence"

                    with st.expander(f"{number}. {label}  ·  Score: {item.score:.3f}"):
                        if item.modality == "image":
                            page_text = f" — page {item.page_number}" if item.page_number is not None else ""
                            st.caption(f"**Source Document:** {item.document_name}{page_text} (Visual Evidence)")
                        elif item.source_type == "object_detection":
                            st.caption(f"**Source Document:** {item.document_name} (YOLO26n Object Detection)")
                        else:
                            st.caption(f"**Source Document:** {item.document_name} (Text Evidence)")
                        
                        st.write(item.caption or item.text)
                        if item.image_path and Path(item.image_path).exists():
                            st.image(item.image_path, use_container_width=True)


# ==============================================================================
# PAGE 4: EVIDENCE (EXACT PIXEL-MATCH TO SCREENSHOT)
# ==============================================================================

elif st.session_state.page == "Evidence":
    st.markdown('<div class="badge-pill">TRACEABILITY</div>', unsafe_allow_html=True)
    st.markdown("<h1>Evidence</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Inspect retrieved textual passages and visual evidence extracted from the active dataset.</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result
    if result is None:
        st.markdown(
            '<div class="info-banner">No query result in session. Execute a question on the Ask RAG page to inspect evidence.</div>',
            unsafe_allow_html=True,
        )
    else:
        evidence = getattr(result, "evidence", None) or []
        if not evidence:
            st.warning("No evidence was returned.")
        else:
            text_evidence = [item for item in evidence if item.modality == "text" and item.source_type != "object_detection"]
            image_evidence = [item for item in evidence if item.modality == "image"]
            det_evidence = [item for item in evidence if item.source_type == "object_detection"]

            tab_titles = [f"Text ({len(text_evidence)})", f"Images ({len(image_evidence)})"]
            if det_evidence:
                tab_titles.append(f"Detections ({len(det_evidence)})")

            tabs = st.tabs(tab_titles)

            with tabs[0]:
                if not text_evidence:
                    st.info("No text evidence for this answer.")
                for item in text_evidence:
                    st.markdown(f"**{item.document_name}**")
                    st.caption(f"Score: {item.score:.3f}")
                    st.write(item.text)
                    st.divider()

            with tabs[1]:
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

            if len(tabs) > 2 and det_evidence:
                with tabs[2]:
                    for item in det_evidence:
                        st.markdown(f"**🎯 {item.document_name}**")
                        st.caption(f"Score: {item.score:.3f} · Object Detection")
                        st.code(item.text, language="text")
                        if item.image_path and Path(item.image_path).exists():
                            st.image(item.image_path, use_container_width=True)
                        st.divider()


# ==============================================================================
# PAGE 5: ARCHITECTURE (EXACT PIXEL-MATCH TO SCREENSHOT)
# ==============================================================================

elif st.session_state.page == "Architecture":
    st.markdown('<div class="badge-pill">SYSTEM DESIGN</div>', unsafe_allow_html=True)
    st.markdown("<h1>Architecture</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Multimodal Retrieval-Augmented Generation with Vision & YOLO26n Object Detection.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="arch-pipeline-container">
    <div class="arch-pipeline-header">⚡ End-to-End Multimodal Pipeline</div>
    <div class="arch-pipeline-desc">Uploaded documents and images remain unified logical entities while generating parallel searchable text chunks and visual page representations.</div>
    <div class="arch-grid">
        <div class="arch-card">
            <div class="arch-card-icon">📁</div>
            <div class="arch-card-title">1. Document Ingestion</div>
            <div class="arch-card-body">Automatic file type detection (PDF, TXT, PNG, JPG, WEBP). PDFs undergo text extraction and page visual extraction.</div>
        </div>
        <div class="arch-card">
            <div class="arch-card-icon">🧠</div>
            <div class="arch-card-title">2. Dual Embeddings</div>
            <div class="arch-card-body">Text: all-MiniLM-L6-v2 (384-dim)<br>Visual: CLIP ViT-B/32 multimodal embeddings.</div>
        </div>
        <div class="arch-card">
            <div class="arch-card-icon">🔍</div>
            <div class="arch-card-title">3. Multimodal FAISS Index</div>
            <div class="arch-card-body">Normalized inner-product vector indexing enabling exact cosine similarity retrieval across modalities.</div>
        </div>
        <div class="arch-card">
            <div class="arch-card-icon">⚡</div>
            <div class="arch-card-title">4. Query Analyzer</div>
            <div class="arch-card-body">Classifies query intent, identifies targeting, and routes retrieval mode (Text, Image, or Hybrid).</div>
        </div>
        <div class="arch-card">
            <div class="arch-card-icon">🛡</div>
            <div class="arch-card-title">5. Evidence & Answer Verification</div>
            <div class="arch-card-body">Evaluates evidence sufficiency before generation and verifies grounded claims to prevent hallucinations.</div>
        </div>
        <div class="arch-card">
            <div class="arch-card-icon">💬</div>
            <div class="arch-card-title">6. Grounded Generator</div>
            <div class="arch-card-body">High-performance Groq LLM generation strictly constrained to verified source evidence citations.</div>
        </div>
    </div>
</div>
<div class="arch-decision-box">💡 <strong>Key Architectural Decision:</strong> Multi-page PDFs and documents remain unified logical entities. Visual pages and images are indexed internally so retrieval can leverage both textual data and visual evidence (e.g. diagrams, charts, stamps, objects) while maintaining strict document provenance.</div>""",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 6: STATUS
# ==============================================================================

elif st.session_state.page == "Status":
    st.markdown('<div class="badge-pill">DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.markdown("<h1>Status</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real-time health status of local vector indices, embedding models, and LLM connections.</div>',
        unsafe_allow_html=True,
    )

    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Files", len(active_files))
    c2.metric("Text chunks", text_store.index.ntotal if text_store else 0)
    c3.metric("Image items", image_store.index.ntotal if image_store else 0)

    st.markdown("<h2>Component Health</h2>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        t_online = text_store is not None
        t_desc = f"{text_store.index.ntotal} searchable chunks" if t_online else "Text index unavailable"
        t_dot = "green" if t_online else "amber"
        t_badge_class = "ready" if t_online else "unavailable"
        t_badge_text = "Ready" if t_online else "Unavailable"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {t_dot}"></div>
        <div>
            <div class="status-card-title">Dense Text FAISS Index</div>
            <div class="status-card-desc">{t_desc}</div>
        </div>
    </div>
    <span class="status-badge {t_badge_class}">{t_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

        i_online = image_store is not None
        i_desc = f"{image_store.index.ntotal} searchable visual vectors" if i_online else "Visual index unavailable"
        i_dot = "green" if i_online else "amber"
        i_badge_class = "ready" if i_online else "unavailable"
        i_badge_text = "Ready" if i_online else "Unavailable"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {i_dot}"></div>
        <div>
            <div class="status-card-title">Dense Image CLIP FAISS Index</div>
            <div class="status-card-desc">{i_desc}</div>
        </div>
    </div>
    <span class="status-badge {i_badge_class}">{i_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator green"></div>
        <div>
            <div class="status-card-title">YOLO26n Object Detector</div>
            <div class="status-card-desc">Ultralytics YOLO26n Ready</div>
        </div>
    </div>
    <span class="status-badge ready">Ready</span>
</div>""",
            unsafe_allow_html=True,
        )

        g_ready = bool(get_api_key())
        g_desc = "Connected (openai/gpt-oss-120b & qwen/qwen3.8-27b)" if g_ready else "GROQ_API_KEY is not configured"
        g_dot = "green" if g_ready else "amber"
        g_badge_class = "connected" if g_ready else "missing"
        g_badge_text = "Connected" if g_ready else "Missing"

        st.markdown(
            f"""<div class="status-card">
    <div class="status-card-left">
        <div class="status-dot-indicator {g_dot}"></div>
        <div>
            <div class="status-card-title">Groq LLM Engine</div>
            <div class="status-card-desc">{g_desc}</div>
        </div>
    </div>
    <span class="status-badge {g_badge_class}">{g_badge_text}</span>
</div>""",
            unsafe_allow_html=True,
        )

    if active_files:
        st.markdown("<h2>Active files</h2>", unsafe_allow_html=True)
        for file_info in active_files:
            render_file_row(file_info["name"], file_info["type"])


# ==============================================================================
# PAGE 7: SETTINGS
# ==============================================================================

elif st.session_state.page == "Settings":
    st.markdown('<div class="badge-pill">CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("<h1>Settings</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Configure retrieval depth, default modality selection, and technical diagnostics.</div>',
        unsafe_allow_html=True,
    )

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
        "Selecting a PDF document automatically uses both text and visual retrieval. "
        "Standalone images use image retrieval. TXT files use text retrieval."
    )
    st.caption("Models: text reasoning uses openai/gpt-oss-120b. Vision and judging use qwen/qwen3.8-27b.")
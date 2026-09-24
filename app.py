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

# Streamlit Page Setup
st.set_page_config(
    page_title="CAR-RAG Builder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">

        <style>
            /* ==========================================================================
               GLOBAL DESIGN SYSTEM TOKENS
               ========================================================================== */
            :root {
                --bg-main: #090D16;
                --bg-surface: #0F172A;
                --bg-card: #141E33;
                --bg-card-hover: #1A2744;
                --bg-card-subtle: #111A2E;
                --bg-sidebar: #0B1120;
                
                --border-subtle: #1E293B;
                --border-card: #22304A;
                --border-highlight: #33476B;
                --border-accent: rgba(59, 130, 246, 0.4);
                
                --primary: #3B82F6;
                --primary-gradient: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
                --primary-glow: rgba(59, 130, 246, 0.18);
                --cyan-accent: #06B6D4;
                --emerald-accent: #10B981;
                --amber-accent: #F59E0B;
                --rose-accent: #F43F5E;
                --indigo-accent: #6366F1;
                
                --text-main: #F8FAFC;
                --text-muted: #94A3B8;
                --text-dim: #64748B;
                --text-bright: #FFFFFF;
                
                --radius-sm: 6px;
                --radius-md: 10px;
                --radius-lg: 14px;
                --radius-xl: 18px;
                --radius-full: 9999px;
                
                --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                --font-heading: 'Plus Jakarta Sans', var(--font-sans);
                --font-mono: 'JetBrains Mono', monospace;
            }

            /* Global Typography & Background */
            html, body, [class*="css"], .stApp {
                font-family: var(--font-sans);
                background-color: var(--bg-main) !important;
                color: var(--text-main) !important;
                letter-spacing: -0.01em;
            }

            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: var(--bg-main);
            }
            ::-webkit-scrollbar-thumb {
                background: #1E293B;
                border-radius: var(--radius-full);
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #334155;
            }

            /* Top Streamlit bar styling */
            [data-testid="stHeader"] {
                background: transparent !important;
            }

            .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 3.5rem !important;
                max-width: 1200px !important;
            }

            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                font-family: var(--font-heading) !important;
                font-weight: 700 !important;
                color: var(--text-bright) !important;
                letter-spacing: -0.02em !important;
            }
            h1 { font-size: 1.95rem !important; margin-bottom: 0.25rem !important; }
            h2 { font-size: 1.35rem !important; margin-top: 1.25rem !important; margin-bottom: 0.75rem !important; }
            h3 { font-size: 1.1rem !important; margin-top: 1rem !important; }

            /* ==========================================================================
               SIDEBAR STYLING
               ========================================================================== */
            [data-testid="stSidebar"] {
                background: var(--bg-sidebar) !important;
                border-right: 1px solid var(--border-card) !important;
                padding-top: 0.5rem !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-top: 1rem !important;
            }
            
            /* Sidebar Brand Header */
            .sidebar-brand {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.6rem 0.6rem 1.1rem 0.6rem;
                margin-bottom: 0.75rem;
                border-bottom: 1px solid var(--border-subtle);
            }
            .sidebar-brand-icon {
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
                border: 1px solid rgba(96, 165, 250, 0.4);
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.1rem;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            }
            .sidebar-brand-text {
                display: flex;
                flex-direction: column;
            }
            .sidebar-brand-title {
                font-family: var(--font-heading);
                font-weight: 700;
                font-size: 1.05rem;
                color: var(--text-bright);
                letter-spacing: -0.02em;
                line-height: 1.2;
            }
            .sidebar-brand-subtitle {
                font-size: 0.72rem;
                color: var(--text-muted);
                font-weight: 400;
                letter-spacing: 0.01em;
            }

            /* Sidebar Section Labels */
            .sidebar-nav-header {
                font-size: 0.68rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--text-dim);
                margin-top: 1rem;
                margin-bottom: 0.4rem;
                padding-left: 0.5rem;
            }

            /* Sidebar Navigation Buttons */
            [data-testid="stSidebar"] .stButton > button {
                width: 100% !important;
                text-align: left !important;
                justify-content: flex-start !important;
                padding: 0.55rem 0.85rem !important;
                font-size: 0.88rem !important;
                font-weight: 500 !important;
                border-radius: var(--radius-md) !important;
                transition: all 0.18s ease-in-out !important;
                margin-bottom: 0.2rem !important;
                border: 1px solid transparent !important;
                background: transparent !important;
                color: var(--text-muted) !important;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                background: rgba(30, 41, 59, 0.6) !important;
                color: var(--text-main) !important;
                border-color: var(--border-subtle) !important;
                transform: translateX(2px);
            }
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: rgba(37, 99, 235, 0.14) !important;
                color: #60A5FA !important;
                border: 1px solid rgba(59, 130, 246, 0.35) !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
            }

            /* Sidebar Footer */
            .sidebar-footer {
                margin-top: 1.5rem;
                padding: 0.85rem;
                background: var(--bg-surface);
                border: 1px solid var(--border-subtle);
                border-radius: var(--radius-lg);
            }
            .sidebar-status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.8rem;
                font-weight: 600;
                color: #34D399;
                margin-bottom: 0.35rem;
            }
            .status-dot {
                width: 7px;
                height: 7px;
                border-radius: var(--radius-full);
                background-color: #10B981;
                box-shadow: 0 0 8px #10B981;
                display: inline-block;
            }
            .status-dot-amber {
                background-color: #F59E0B;
                box-shadow: 0 0 8px #F59E0B;
            }
            .sidebar-footer-stat {
                font-size: 0.75rem;
                color: var(--text-muted);
                display: flex;
                justify-content: space-between;
                margin-top: 0.25rem;
            }

            /* ==========================================================================
               PAGE HEADER COMPONENT
               ========================================================================== */
            .page-header {
                margin-bottom: 1.5rem;
                padding-bottom: 0.85rem;
                border-bottom: 1px solid var(--border-subtle);
            }
            .page-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: #60A5FA;
                background: rgba(37, 99, 235, 0.12);
                border: 1px solid rgba(59, 130, 246, 0.25);
                padding: 0.2rem 0.6rem;
                border-radius: var(--radius-full);
                margin-bottom: 0.45rem;
            }
            .page-title {
                font-size: 1.75rem;
                font-weight: 800;
                color: var(--text-bright);
                letter-spacing: -0.025em;
                line-height: 1.2;
                margin: 0;
            }
            .page-subtitle {
                font-size: 0.92rem;
                color: var(--text-muted);
                margin-top: 0.35rem;
                margin-bottom: 0;
                line-height: 1.45;
            }

            /* ==========================================================================
               CARDS & METRICS
               ========================================================================== */
            div[data-testid="stMetric"] {
                background: var(--bg-card) !important;
                border: 1px solid var(--border-card) !important;
                border-radius: var(--radius-lg) !important;
                padding: 0.85rem 1.1rem !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
                transition: all 0.2s ease !important;
            }
            div[data-testid="stMetric"]:hover {
                border-color: var(--border-highlight) !important;
                background: var(--bg-card-hover) !important;
                transform: translateY(-1px);
            }
            [data-testid="stMetricLabel"] p {
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                color: var(--text-muted) !important;
            }
            [data-testid="stMetricValue"] {
                font-family: var(--font-heading) !important;
                font-size: 1.6rem !important;
                font-weight: 700 !important;
                color: var(--text-bright) !important;
            }

            /* Custom UI Cards */
            .ui-card {
                background: var(--bg-card);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-lg);
                padding: 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            }
            .ui-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.85rem;
                padding-bottom: 0.65rem;
                border-bottom: 1px solid var(--border-subtle);
            }
            .ui-card-title {
                font-size: 0.98rem;
                font-weight: 700;
                color: var(--text-bright);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            /* Document / File Row Item */
            .file-card {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.85rem 1.1rem;
                margin-bottom: 0.6rem;
                background: var(--bg-card-subtle);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-md);
                transition: all 0.18s ease;
            }
            .file-card:hover {
                background: var(--bg-card);
                border-color: var(--border-highlight);
            }
            .file-info {
                display: flex;
                align-items: center;
                gap: 0.85rem;
                min-width: 0;
            }
            .file-icon {
                width: 38px;
                height: 38px;
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.15rem;
                flex-shrink: 0;
            }
            .file-icon-pdf {
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #F87171;
            }
            .file-icon-img {
                background: rgba(147, 51, 234, 0.15);
                border: 1px solid rgba(147, 51, 234, 0.3);
                color: #C084FC;
            }
            .file-icon-txt {
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.3);
                color: #60A5FA;
            }
            .file-name-text {
                font-weight: 600;
                font-size: 0.92rem;
                color: var(--text-bright);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .file-subtext {
                font-size: 0.74rem;
                color: var(--text-muted);
                margin-top: 0.15rem;
            }
            .file-badges {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                flex-shrink: 0;
            }

            /* Badges & Tags */
            .badge {
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                padding: 0.25rem 0.65rem;
                border-radius: var(--radius-full);
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
            }
            .badge-blue {
                background: rgba(59, 130, 246, 0.14);
                border: 1px solid rgba(59, 130, 246, 0.3);
                color: #93C5FD;
            }
            .badge-emerald {
                background: rgba(16, 185, 129, 0.14);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #6EE7B7;
            }
            .badge-purple {
                background: rgba(168, 85, 247, 0.14);
                border: 1px solid rgba(168, 85, 247, 0.3);
                color: #D8B4FE;
            }
            .badge-amber {
                background: rgba(245, 158, 11, 0.14);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #FCD34D;
            }

            /* Status Card Component */
            .status-card {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.95rem 1.15rem;
                background: var(--bg-card);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-lg);
                margin-bottom: 0.65rem;
            }
            .status-card-left {
                display: flex;
                align-items: center;
                gap: 0.85rem;
            }
            .status-card-title {
                font-weight: 600;
                font-size: 0.9rem;
                color: var(--text-bright);
            }
            .status-card-subtitle {
                font-size: 0.75rem;
                color: var(--text-muted);
            }

            /* ==========================================================================
               RAG CHAT & ANSWER WORKSPACE
               ========================================================================== */
            .answer-card {
                background: linear-gradient(180deg, #16223B 0%, #10192C 100%);
                border: 1px solid rgba(59, 130, 246, 0.35);
                border-radius: var(--radius-xl);
                padding: 1.5rem;
                margin-top: 1rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), 0 0 16px rgba(37, 99, 235, 0.08);
            }
            .answer-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1rem;
                padding-bottom: 0.75rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }
            .answer-card-title {
                font-family: var(--font-heading);
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text-bright);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .answer-body {
                font-size: 1rem;
                line-height: 1.65;
                color: #E2E8F0;
                letter-spacing: -0.005em;
            }

            /* Evidence Items */
            .evidence-card {
                background: var(--bg-card);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-md);
                padding: 1rem;
                margin-bottom: 0.75rem;
                transition: border-color 0.18s ease;
            }
            .evidence-card:hover {
                border-color: var(--border-highlight);
            }
            .evidence-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.6rem;
            }
            .evidence-source {
                font-weight: 600;
                font-size: 0.86rem;
                color: var(--text-bright);
                display: flex;
                align-items: center;
                gap: 0.45rem;
            }
            .evidence-score {
                font-family: var(--font-mono);
                font-size: 0.75rem;
                font-weight: 600;
                color: #60A5FA;
                background: rgba(37, 99, 235, 0.12);
                padding: 0.2rem 0.55rem;
                border-radius: var(--radius-sm);
                border: 1px solid rgba(59, 130, 246, 0.25);
            }
            .evidence-snippet {
                font-family: var(--font-sans);
                font-size: 0.85rem;
                line-height: 1.55;
                color: #CBD5E1;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: var(--radius-sm);
                padding: 0.75rem 0.95rem;
            }

            /* Architecture Diagram Container */
            .arch-pipeline {
                background: var(--bg-card);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-xl);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .arch-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .arch-box {
                background: var(--bg-card-subtle);
                border: 1px solid var(--border-card);
                border-radius: var(--radius-lg);
                padding: 1.1rem;
                transition: all 0.2s ease;
            }
            .arch-box:hover {
                border-color: rgba(59, 130, 246, 0.4);
                background: var(--bg-card);
                transform: translateY(-2px);
            }
            .arch-box-icon {
                font-size: 1.4rem;
                margin-bottom: 0.5rem;
            }
            .arch-box-title {
                font-weight: 700;
                font-size: 0.95rem;
                color: var(--text-bright);
                margin-bottom: 0.35rem;
            }
            .arch-box-desc {
                font-size: 0.82rem;
                color: var(--text-muted);
                line-height: 1.45;
            }

            /* ==========================================================================
               INPUTS, BUTTONS & FORM CONTROLS
               ========================================================================== */
            .stTextArea textarea, .stTextInput input, .stSelectbox select {
                background-color: var(--bg-card-subtle) !important;
                border: 1px solid var(--border-card) !important;
                border-radius: var(--radius-md) !important;
                color: var(--text-bright) !important;
                font-size: 0.92rem !important;
                transition: all 0.2s ease !important;
            }
            .stTextArea textarea:focus, .stTextInput input:focus {
                border-color: #3B82F6 !important;
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
                background-color: var(--bg-surface) !important;
            }

            /* Primary Action Buttons */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
                color: #FFFFFF !important;
                border: 1px solid rgba(96, 165, 250, 0.4) !important;
                font-weight: 600 !important;
                border-radius: var(--radius-md) !important;
                padding: 0.6rem 1.25rem !important;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
                box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
                transform: translateY(-1px);
            }

            /* Secondary Action Buttons */
            .stButton > button[kind="secondary"] {
                background: var(--bg-card-subtle) !important;
                border: 1px solid var(--border-card) !important;
                color: var(--text-main) !important;
                border-radius: var(--radius-md) !important;
                font-weight: 500 !important;
                transition: all 0.18s ease !important;
            }
            .stButton > button[kind="secondary"]:hover {
                background: var(--bg-card-hover) !important;
                border-color: var(--border-highlight) !important;
            }

            /* File Uploader Container */
            [data-testid="stFileUploader"] {
                background: var(--bg-card-subtle) !important;
                border: 1px dashed var(--border-highlight) !important;
                border-radius: var(--radius-lg) !important;
                padding: 1.25rem !important;
                transition: all 0.2s ease !important;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #3B82F6 !important;
                background: rgba(15, 23, 42, 0.8) !important;
            }

            /* Tabs Styling */
            div[data-baseweb="tab-list"] {
                gap: 0.5rem !important;
                background-color: transparent !important;
                border-bottom: 1px solid var(--border-subtle) !important;
                margin-bottom: 1rem !important;
            }
            div[data-baseweb="tab"] {
                border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
                background-color: transparent !important;
                color: var(--text-muted) !important;
                font-weight: 600 !important;
                font-size: 0.88rem !important;
                padding: 0.6rem 1.1rem !important;
                border: none !important;
            }
            div[data-baseweb="tab"][aria-selected="true"] {
                color: #60A5FA !important;
                border-bottom: 2px solid #3B82F6 !important;
                background: rgba(37, 99, 235, 0.08) !important;
            }

            /* Expanders */
            .streamlit-expanderHeader {
                background: var(--bg-card) !important;
                border: 1px solid var(--border-card) !important;
                border-radius: var(--radius-md) !important;
                color: var(--text-main) !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
            }
            .streamlit-expanderContent {
                background: var(--bg-card-subtle) !important;
                border: 1px solid var(--border-card) !important;
                border-top: none !important;
                border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
                padding: 1rem !important;
            }

            /* Alerts & Notifications */
            div[data-testid="stAlert"] {
                border-radius: var(--radius-md) !important;
                border: 1px solid var(--border-card) !important;
                background: var(--bg-card) !important;
            }

            /* Utility classes */
            .muted { color: var(--text-muted) !important; font-size: 0.88rem !important; }
            .mono { font-family: var(--font-mono) !important; }
            .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
            @media (max-width: 768px) {
                .grid-2 { grid-template-columns: 1fr; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ==============================================================================
# STATE & DATASET MANAGEMENT
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


# ==============================================================================
# UI HELPER RENDERING FUNCTIONS
# ==============================================================================

def render_page_header(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-badge">{escape(badge)}</div>
            <h1 class="page-title">{escape(title)}</h1>
            <p class="page-subtitle">{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_card(name: str, file_type: str, modality: str = "", is_staged: bool = False) -> None:
    ft = file_type.upper()
    if ft == "PDF":
        icon_class = "file-icon-pdf"
        icon_char = "📄"
        type_badge = '<span class="badge badge-purple">PDF DOCUMENT</span>'
    elif ft in {"PNG", "JPG", "JPEG", "WEBP"}:
        icon_class = "file-icon-img"
        icon_char = "🖼️"
        type_badge = '<span class="badge badge-blue">IMAGE MODALITY</span>'
    else:
        icon_class = "file-icon-txt"
        icon_char = "📝"
        type_badge = '<span class="badge badge-emerald">TEXT SOURCE</span>'

    status_badge = (
        '<span class="badge badge-amber">● Staged</span>'
        if is_staged
        else '<span class="badge badge-emerald">● Indexed</span>'
    )

    st.markdown(
        f"""
        <div class="file-card">
            <div class="file-info">
                <div class="file-icon {icon_class}">{icon_char}</div>
                <div>
                    <div class="file-name-text">{escape(name)}</div>
                    <div class="file-subtext">{ft} Format · Internal Multimodal Unit</div>
                </div>
            </div>
            <div class="file-badges">
                {type_badge}
                {status_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(title: str, subtitle: str, is_ready: bool, ready_label: str = "Ready", unready_label: str = "Offline") -> None:
    dot_class = "status-dot" if is_ready else "status-dot status-dot-amber"
    badge_class = "badge-emerald" if is_ready else "badge-amber"
    label_text = ready_label if is_ready else unready_label

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-card-left">
                <span class="{dot_class}"></span>
                <div>
                    <div class="status-card-title">{escape(title)}</div>
                    <div class="status-card-subtitle">{escape(subtitle)}</div>
                </div>
            </div>
            <div>
                <span class="badge {badge_class}">{escape(label_text)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MODEL & PIPELINE SERVICES
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

        status.write("Building FAISS multimodal indexes...")

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


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

NAV_SECTIONS = [
    {
        "header": "MAIN",
        "items": [
            ("Dashboard", "📊"),
            ("Documents", "📁"),
            ("Ask RAG", "⚡"),
        ],
    },
    {
        "header": "ANALYSIS",
        "items": [
            ("Evidence", "📑"),
            ("Architecture", "🏗️"),
            ("Status", "🟢"),
        ],
    },
    {
        "header": "SYSTEM",
        "items": [
            ("Settings", "⚙️"),
        ],
    },
]

with st.sidebar:
    # Top Branding
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">⚡</div>
            <div class="sidebar-brand-text">
                <span class="sidebar-brand-title">CAR-RAG Builder</span>
                <span class="sidebar-brand-subtitle">Certificate-aware multimodal retrieval</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Groups
    for section in NAV_SECTIONS:
        st.markdown(f'<div class="sidebar-nav-header">{section["header"]}</div>', unsafe_allow_html=True)
        for page_name, icon in section["items"]:
            is_active = st.session_state.page == page_name
            btn_label = f"{icon}  {page_name}"
            if st.button(
                btn_label,
                use_container_width=True,
                type="primary" if is_active else "secondary",
                key=f"nav_{page_name}",
            ):
                st.session_state.page = page_name
                st.rerun()

    # Footer Card
    active_files = st.session_state.active_dataset.get("files", [])
    groq_ready = bool(get_api_key())
    
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="sidebar-status-pill">
                <span class="status-dot"></span>
                <span>CAR-RAG Ready</span>
            </div>
            <div class="sidebar-footer-stat">
                <span>Active Files</span>
                <strong style="color: #F8FAFC;">{len(active_files)}</strong>
            </div>
            <div class="sidebar-footer-stat">
                <span>LLM Engine</span>
                <strong style="color: {'#34D399' if groq_ready else '#F59E0B'};">{'Groq Online' if groq_ready else 'Unset'}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================

if st.session_state.page == "Dashboard":
    render_page_header(
        badge="System Overview",
        title="Dashboard",
        subtitle="Real-time multimodal indexing, vector store diagnostics, and active dataset status.",
    )

    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    searchable_items = 0
    if text_store:
        searchable_items += text_store.index.ntotal
    if image_store:
        searchable_items += image_store.index.ntotal

    # High Level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Files", len(active_files))
    m2.metric("Searchable Items", searchable_items)
    m3.metric("Embeddings", "MiniLM + CLIP")
    m4.metric("LLM Engine", "Groq")

    # Active Dataset Section
    st.markdown("### Active Dataset")
    if not active_files:
        st.info("No active dataset loaded. Please navigate to Documents to upload source files.")
    else:
        for file_info in active_files:
            render_file_card(
                name=file_info["name"],
                file_type=file_info["type"],
                modality=file_info.get("modality", "text"),
                is_staged=False,
            )

    # System Status Section
    st.markdown("### System Status")
    left, right = st.columns(2)
    with left:
        render_status_row(
            title="Text Retrieval Index",
            subtitle=f"{text_store.index.ntotal} FAISS chunks ready" if text_store else "Index offline",
            is_ready=text_store is not None,
            ready_label="Ready",
            unready_label="Unavailable",
        )
        render_status_row(
            title="Image Retrieval Index",
            subtitle=f"{image_store.index.ntotal} visual vectors ready" if image_store else "Index offline",
            is_ready=image_store is not None,
            ready_label="Ready",
            unready_label="Unavailable",
        )
    with right:
        render_status_row(
            title="Groq LLM Connection",
            subtitle="API key configured & ready" if get_api_key() else "API key required in settings",
            is_ready=bool(get_api_key()),
            ready_label="Connected",
            unready_label="Missing Key",
        )
        render_status_row(
            title="Dataset State",
            subtitle=f"{len(active_files)} active document(s) synchronized" if active_files else "Waiting for dataset",
            is_ready=bool(active_files),
            ready_label="Dataset Loaded Successfully",
            unready_label="No Data",
        )


# ==============================================================================
# PAGE 2: DOCUMENTS
# ==============================================================================

elif st.session_state.page == "Documents":
    render_page_header(
        badge="Data Management",
        title="Documents",
        subtitle="Upload mixed documents and certificates to build unified text and visual searchable FAISS indexes.",
    )

    st.markdown(
        """
        <div style="margin-bottom: 0.85rem; font-size: 0.82rem; color: #94A3B8;">
            <strong style="color: #F8FAFC;">Supported formats:</strong> PDF, TXT, PNG, JPG, JPEG, WEBP. PDF visual pages are automatically extracted and indexed internally as visual evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        accept_multiple_files=True,
        key="document_uploader",
    )

    if uploaded_files:
        st.markdown("### Detected Files for Ingestion")
        for uploaded_file in uploaded_files:
            detected_type = detect_uploaded_file_type(
                uploaded_file,
                uploaded_file.getvalue(),
            )
            render_file_card(
                name=uploaded_file.name,
                file_type=detected_type.upper(),
                is_staged=True,
            )

        if st.button("⚡ Process & Build Multimodal Indexes", type="primary", use_container_width=True):
            try:
                process_uploads(uploaded_files)
            except Exception as error:
                show_error("Processing failed. The previous Active Dataset was retained.", error)

    st.markdown("### Current Active Dataset")
    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.info("No active files have been processed yet. Upload documents above to begin.")
    else:
        for file_info in active_files:
            render_file_card(
                name=file_info["name"],
                file_type=file_info["type"],
                modality=file_info.get("modality", "text"),
                is_staged=False,
            )
        st.caption("ℹ️ Extracted PDF page images serve as internal multimodal evidence and remain traceable to their parent certificate.")


# ==============================================================================
# PAGE 3: ASK RAG (AI RETRIEVAL INTERFACE)
# ==============================================================================

elif st.session_state.page == "Ask RAG":
    render_page_header(
        badge="Multimodal Q&A",
        title="Ask RAG",
        subtitle="Certificate-aware question answering powered by adaptive retrieval and strict evidence verification.",
    )

    active_files = st.session_state.active_dataset.get("files", [])
    if not active_files:
        st.info("Please upload and process a dataset from Documents before querying.")
    else:
        source_names = [file_info["name"] for file_info in active_files]
        selected_source: str | None = None

        # Source Selection Controls
        s_col1, s_col2 = st.columns([3, 2])
        with s_col1:
            if len(source_names) == 1:
                selected_source = source_names[0]
                st.markdown(
                    f"""
                    <div style="background: #141E33; border: 1px solid #22304A; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.88rem; color: #F8FAFC;">
                        Active Source: <strong style="color: #60A5FA;">{escape(selected_source)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                default_index = 0
                if st.session_state.selected_source in source_names:
                    default_index = source_names.index(st.session_state.selected_source)

                search_all = st.session_state.get("search_all_toggle", False)
                selected_source = st.selectbox(
                    "Select Target Document",
                    options=source_names,
                    index=default_index,
                    disabled=search_all,
                    key="source_select",
                    help="Choose a specific source to strictly isolate retrieval to that document.",
                )
                st.session_state.selected_source = selected_source

        with s_col2:
            search_all = st.checkbox(
                "Search all active files",
                value=False,
                key="search_all_toggle",
                help="When enabled, queries both text and image indexes across all active documents.",
            )

        question = st.text_area(
            "Your Question",
            placeholder="Ask a specific question about the certificate, issuer, dates, visual logos, or criteria...",
            height=100,
        )

        if st.button("⚡ Run CAR-RAG Query", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question to execute retrieval.")
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
                        "Please select the specific certificate you want to query."
                    )
                else:
                    if inferred and not selected_source and not search_all:
                        st.caption(f"Matched source context from question: **{inferred}**")

                    with st.spinner("Analyzing Query → Multimodal Retrieval → Evidence Verification → Grounded Generation → Answer Verification..."):
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

        # Result Display Area
        result = st.session_state.last_result
        if result is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Grounded Answer Container
            grounding_verified = getattr(result, "grounding_verified", False)
            evidence_sufficient = getattr(result, "evidence_sufficient", False)
            
            if grounding_verified:
                status_pill = '<span class="badge badge-emerald">✓ Verified Grounded</span>'
            elif not evidence_sufficient:
                status_pill = '<span class="badge badge-amber">⚠ Insufficient Evidence</span>'
            else:
                status_pill = '<span class="badge badge-purple">● Generated</span>'

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-card-header">
                        <div class="answer-card-title">
                            <span>⚡</span> Grounded Answer
                        </div>
                        <div>{status_pill}</div>
                    </div>
                    <div class="answer-body">
                        {escape(result.answer) if result.answer else "<em>No grounded answer was returned based on the available evidence.</em>"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CAR-RAG Pipeline Decision Diagnostics
            st.markdown("### CAR-RAG Pipeline Decision")
            analysis = result.query_analysis
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Query Type", str(getattr(analysis, "query_type", "unknown")).title())
            c2.metric("Image Retrieval", "Active" if getattr(analysis, "needs_image_retrieval", False) else "Inactive")
            c3.metric("Evidence Count", len(result.evidence or []))
            
            grounding_text = "Verified" if result.grounding_verified else ("Insufficient" if not result.evidence_sufficient else "Unverified")
            c4.metric("Verification", grounding_text)

            # Reasoning Diagnostics
            with st.expander("🔍 Inspection & Verification Diagnostics", expanded=False):
                if getattr(result, "evidence_reason", ""):
                    st.markdown(f"**Evidence Check Evaluation:** {result.evidence_reason}")
                if getattr(result, "answer_reason", ""):
                    st.markdown(f"**Answer Check Verification:** {result.answer_reason}")
                if getattr(result, "re_retrieved", False):
                    st.markdown("**Adaptive Pipeline:** Re-retrieval was automatically triggered to resolve insufficient initial evidence.")

            # Retrieved Evidence Items
            evidence = result.evidence or []
            if evidence:
                st.markdown("### Retrieved Supporting Evidence")
                for number, item in enumerate(evidence, start=1):
                    with st.expander(f"{number}. {item.display_label}  ·  Score: {item.score:.3f}"):
                        if item.modality == "image":
                            page_text = f" — Page {item.page_number}" if item.page_number is not None else ""
                            st.caption(f"**Source Document:** {item.document_name}{page_text} (Visual Modality)")
                        else:
                            st.caption(f"**Source Document:** {item.document_name} (Text Modality)")
                        
                        st.markdown(f'<div class="evidence-snippet">{escape(item.caption or item.text)}</div>', unsafe_allow_html=True)
                        if item.image_path and Path(item.image_path).exists():
                            st.image(item.image_path, use_container_width=True)


# ==============================================================================
# PAGE 4: EVIDENCE
# ==============================================================================

elif st.session_state.page == "Evidence":
    render_page_header(
        badge="Traceability",
        title="Evidence",
        subtitle="Inspect retrieved textual passages and visual evidence extracted from the active dataset.",
    )

    result = st.session_state.last_result
    if result is None:
        st.info("No query result in session. Execute a question on the Ask RAG page to inspect evidence.")
    else:
        evidence = getattr(result, "evidence", None) or []
        if not evidence:
            st.warning("No evidence was retrieved for the previous query.")
        else:
            text_evidence = [item for item in evidence if item.modality == "text"]
            image_evidence = [item for item in evidence if item.modality == "image"]
            
            text_tab, image_tab = st.tabs(
                [f"📄 Text Passages ({len(text_evidence)})", f"🖼️ Visual Evidence ({len(image_evidence)})"]
            )

            with text_tab:
                if not text_evidence:
                    st.info("No textual evidence matched this query.")
                for item in text_evidence:
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="evidence-header">
                                <div class="evidence-source">📄 {escape(item.document_name)}</div>
                                <div class="evidence-score">Cosine Score: {item.score:.3f}</div>
                            </div>
                            <div class="evidence-snippet">{escape(item.text)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with image_tab:
                if not image_evidence:
                    st.info("No visual evidence matched this query.")
                for item in image_evidence:
                    page = f" · Page {item.page_number}" if item.page_number is not None else ""
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="evidence-header">
                                <div class="evidence-source">🖼️ {escape(item.document_name)}{page}</div>
                                <div class="evidence-score">Cosine Score: {item.score:.3f} · {escape(item.source_type)}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if item.image_path and Path(item.image_path).exists():
                        st.image(item.image_path, use_container_width=True)
                    if item.caption:
                        st.caption(f"**Visual Caption:** {item.caption}")


# ==============================================================================
# PAGE 5: ARCHITECTURE
# ==============================================================================

elif st.session_state.page == "Architecture":
    render_page_header(
        badge="System Design",
        title="Architecture",
        subtitle="Certificate-Aware Multimodal Retrieval-Augmented Generation system workflow.",
    )

    st.markdown(
        """
        <div class="arch-pipeline">
            <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 0.5rem;">
                ⚡ End-to-End Multimodal Pipeline
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 1rem;">
                Uploaded certificates and documents remain unified logical entities while generating parallel searchable text chunks and visual page representations.
            </div>
            <div class="arch-grid">
                <div class="arch-box">
                    <div class="arch-box-icon">📁</div>
                    <div class="arch-box-title">1. Document Ingestion</div>
                    <div class="arch-box-desc">
                        Automatic file type detection (PDF, TXT, PNG, JPG). PDFs undergo text extraction and page visual extraction.
                    </div>
                </div>
                <div class="arch-box">
                    <div class="arch-box-icon">🧠</div>
                    <div class="arch-box-title">2. Dual Embeddings</div>
                    <div class="arch-box-desc">
                        <strong>Text:</strong> all-MiniLM-L6-v2 (384-dim)<br>
                        <strong>Visual:</strong> CLIP ViT-B/32 multimodal embeddings.
                    </div>
                </div>
                <div class="arch-box">
                    <div class="arch-box-icon">🔍</div>
                    <div class="arch-box-title">3. Multimodal FAISS Index</div>
                    <div class="arch-box-desc">
                        Normalized inner-product vector indexing enabling exact cosine similarity retrieval across modalities.
                    </div>
                </div>
                <div class="arch-box">
                    <div class="arch-box-icon">⚡</div>
                    <div class="arch-box-title">4. Query Analyzer</div>
                    <div class="arch-box-desc">
                        Classifies query intent, identifies certificate targeting, and routes retrieval mode (Text, Image, or Hybrid).
                    </div>
                </div>
                <div class="arch-box">
                    <div class="arch-box-icon">🛡️</div>
                    <div class="arch-box-title">5. Evidence & Answer Verification</div>
                    <div class="arch-box-desc">
                        Evaluates evidence sufficiency before generation and verifies grounded claims to prevent hallucinations.
                    </div>
                </div>
                <div class="arch-box">
                    <div class="arch-box-icon">🤖</div>
                    <div class="arch-box-title">6. Grounded Generator</div>
                    <div class="arch-box-desc">
                        High-performance Groq LLM generation strictly constrained to verified source evidence citations.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "💡 **Key Architectural Decision:** A PDF certificate remains one logical source. Visual pages are extracted internally so retrieval can leverage both textual data and visual evidence (e.g. logos, stamps, signatures) while maintaining strict document provenance."
    )


# ==============================================================================
# PAGE 6: STATUS
# ==============================================================================

elif st.session_state.page == "Status":
    render_page_header(
        badge="Diagnostics",
        title="System Status",
        subtitle="Operational metrics, model connectivity, and index readiness.",
    )

    text_store = load_text_store()
    image_store = load_image_store()
    active_files = st.session_state.active_dataset.get("files", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Files", len(active_files))
    c2.metric("Text Chunks", text_store.index.ntotal if text_store else 0)
    c3.metric("Image Vectors", image_store.index.ntotal if image_store else 0)

    st.markdown("### Component Health")

    render_status_row(
        title="Text Retrieval Store",
        subtitle=f"{text_store.index.ntotal} searchable text chunks in FAISS" if text_store else "Index not loaded",
        is_ready=text_store is not None,
        ready_label="Online",
        unready_label="Offline",
    )

    render_status_row(
        title="Visual Retrieval Store",
        subtitle=f"{image_store.index.ntotal} visual feature vectors in FAISS" if image_store else "Index not loaded",
        is_ready=image_store is not None,
        ready_label="Online",
        unready_label="Offline",
    )

    render_status_row(
        title="Embeddings Pipeline",
        subtitle="SentenceTransformers (MiniLM) + CLIP ViT-B/32",
        is_ready=True,
        ready_label="Active",
        unready_label="Inactive",
    )

    render_status_row(
        title="Groq LLM Service",
        subtitle="OpenAI-compatible inference runtime" if get_api_key() else "GROQ_API_KEY environment variable missing",
        is_ready=bool(get_api_key()),
        ready_label="Connected",
        unready_label="Missing Key",
    )

    if active_files:
        st.markdown("### Loaded Documents")
        for file_info in active_files:
            render_file_card(file_info["name"], file_info["type"], is_staged=False)


# ==============================================================================
# PAGE 7: SETTINGS
# ==============================================================================

elif st.session_state.page == "Settings":
    render_page_header(
        badge="Configuration",
        title="Settings",
        subtitle="Configure retrieval parameters, default search modes, and diagnostic logging.",
    )

    st.markdown("### Retrieval Parameters")

    st.session_state.settings["top_k"] = st.slider(
        "Retrieval Top-K",
        min_value=1,
        max_value=15,
        value=int(st.session_state.settings.get("top_k", 5)),
        help="Number of items to retrieve from each internal index before source filtering.",
    )

    st.session_state.settings["retrieval_mode"] = st.selectbox(
        "Default Retrieval Mode",
        options=["both", "text", "image"],
        index=["both", "text", "image"].index(
            st.session_state.settings.get("retrieval_mode", "both")
        ),
        help="Fallback mode when source type is unspecified. PDFs automatically search both text and visual indices.",
    )

    st.markdown("### Diagnostics & Logging")
    st.session_state.settings["show_debug"] = st.checkbox(
        "Show technical error details",
        value=bool(st.session_state.settings.get("show_debug", False)),
        help="Enable detailed stack traces and Groq raw error payloads.",
    )

    st.markdown("### Active Model Configuration")
    st.markdown(
        """
        <div class="status-card">
            <div>
                <div class="status-card-title">LLM Reasoning & Generation</div>
                <div class="status-card-subtitle">openai/gpt-oss-120b (via Groq)</div>
            </div>
            <div><span class="badge badge-blue">Reasoning</span></div>
        </div>
        <div class="status-card">
            <div>
                <div class="status-card-title">Vision & Grounding Verifier</div>
                <div class="status-card-subtitle">qwen/qwen3.8-27b (via Groq)</div>
            </div>
            <div><span class="badge badge-purple">Multimodal Vision</span></div>
        </div>
        <div class="status-card">
            <div>
                <div class="status-card-title">Dense Embeddings</div>
                <div class="status-card-subtitle">all-MiniLM-L6-v2 + CLIP ViT-B/32</div>
            </div>
            <div><span class="badge badge-emerald">Local Embeddings</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

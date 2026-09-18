from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application configuration."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_mode: str = "both"

    groq_api_key: str = ""

    def __post_init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
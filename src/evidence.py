"""Unified evidence records for text and image retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Modality = Literal["text", "image"]


@dataclass
class RetrievedEvidence:
    """One retrieved item passed through the CAR-RAG workflow."""

    modality: Modality
    document_name: str
    score: float
    text: str
    source_type: str
    chunk_index: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    image_id: str | None = None
    image_path: str | None = None
    page_number: int | None = None
    caption: str | None = None
    source_filename: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        source = self.document_name
        if self.modality == "image":
            page = f" — page {self.page_number}" if self.page_number else ""
            return f"{source} — image evidence{page}"
        return f"{source} — text evidence"

    def evidence_block(self, position: int) -> str:
        """Plain-text block used by checkers and the text generator."""

        if self.modality == "image":
            page = f", page {self.page_number}" if self.page_number else ""
            caption = self.caption or self.text
            path = self.image_path or ""
            return (
                f"[Evidence {position}: IMAGE | {self.document_name}{page}]\n"
                f"Image ID: {self.image_id or 'unknown'}\n"
                f"File: {path}\n"
                f"Caption / OCR: {caption}"
            )

        offsets = ""
        if self.start_char is not None and self.end_char is not None:
            offsets = f", chars {self.start_char}–{self.end_char}"
        chunk = f", chunk {(self.chunk_index or 0) + 1}"
        return (
            f"[Evidence {position}: TEXT | {self.document_name}{chunk}{offsets}]\n"
            f"{self.text}"
        )

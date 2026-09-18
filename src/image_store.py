"""Separate FAISS index for CLIP image embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.evidence import RetrievedEvidence
from src.vector_store import VectorStoreError


IMAGE_METADATA_VERSION = 1


@dataclass
class ImageRecord:
    """Persisted metadata for one indexed image."""

    image_id: str
    modality: str
    document_name: str
    source_filename: str
    image_path: str
    page_number: int | None
    caption: str
    source_type: str
    width: int
    height: int

    def to_metadata(self) -> dict:
        return {
            "image_id": self.image_id,
            "modality": "image",
            "document_name": self.document_name,
            "source_filename": self.source_filename,
            "image_path": self.image_path,
            "page_number": self.page_number,
            "caption": self.caption,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "ImageRecord":
        page = metadata.get("page_number")
        return cls(
            image_id=str(metadata["image_id"]),
            modality="image",
            document_name=str(metadata["document_name"]),
            source_filename=str(metadata.get("source_filename", metadata["document_name"])),
            image_path=str(metadata["image_path"]),
            page_number=int(page) if page is not None else None,
            caption=str(metadata.get("caption", "")),
            source_type=str(metadata.get("source_type", "standalone_image")),
            width=int(metadata.get("width", 0)),
            height=int(metadata.get("height", 0)),
        )

    def to_evidence(self, score: float) -> RetrievedEvidence:
        caption = self.caption.strip() or f"Image from {self.document_name}"
        return RetrievedEvidence(
            modality="image",
            document_name=self.document_name,
            score=float(score),
            text=caption,
            source_type=self.source_type,
            image_id=self.image_id,
            image_path=self.image_path,
            page_number=self.page_number,
            caption=caption,
            source_filename=self.source_filename,
        )


class ImageStore:
    """FAISS inner-product index over normalized CLIP vectors."""

    def __init__(self, index: faiss.Index, records: list[ImageRecord]) -> None:
        self.index = index
        self.records = records

    @property
    def document_names(self) -> list[str]:
        return list(dict.fromkeys(record.document_name for record in self.records))

    @classmethod
    def from_embeddings(
        cls,
        records: list[ImageRecord],
        embeddings: list[list[float]],
    ) -> "ImageStore":
        if not records or not embeddings:
            raise VectorStoreError("Cannot build an image index without records and embeddings.")
        if len(records) != len(embeddings):
            raise VectorStoreError("The image record and embedding counts do not match.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(records):
            raise VectorStoreError("Image embeddings must be a two-dimensional matrix.")
        if not np.isfinite(matrix).all():
            raise VectorStoreError("Image embeddings contain invalid numeric values.")

        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, records=records)

    @staticmethod
    def exists(index_path: Path, metadata_path: Path) -> bool:
        return index_path.is_file() and metadata_path.is_file()

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(index_path))
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": IMAGE_METADATA_VERSION,
                        "modality": "image",
                        "embedding_model": "clip-ViT-B-32",
                        "records": [record.to_metadata() for record in self.records],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to save the image index: {error}") from error

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "ImageStore":
        try:
            index = faiss.read_index(str(index_path))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = [ImageRecord.from_metadata(item) for item in payload["records"]]
        except Exception as error:
            raise VectorStoreError(f"Unable to load the image index: {error}") from error

        if index.ntotal != len(records):
            raise VectorStoreError("The saved image index and metadata are out of sync.")
        return cls(index=index, records=records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        if not self.records or self.index.ntotal == 0:
            raise VectorStoreError("No image index is available. Process images first.")
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")

        query = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: list[RetrievedEvidence] = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                results.append(self.records[int(index)].to_evidence(float(score)))
        return results

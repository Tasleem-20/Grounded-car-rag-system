"""Separate FAISS index for CLIP image embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.evidence import RetrievedEvidence
from src.vector_store import VectorStoreError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"

IMAGE_METADATA_VERSION = 2


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
    ocr_text: str = ""
    detections: list[dict[str, Any]] = field(default_factory=list)
    detection_summary: str = ""
    detection_counts: dict[str, int] = field(default_factory=dict)
    detection_context: str = ""
    detection_error: str = ""

    def to_metadata(self) -> dict:
        return {
            "image_id": self.image_id,
            "modality": "image",
            "document_name": self.document_name,
            "source_filename": self.source_filename,
            "image_path": self.image_path,
            "page_number": self.page_number,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "detections": self.detections,
            "detection_summary": self.detection_summary,
            "detection_counts": self.detection_counts,
            "detection_context": self.detection_context,
            "detection_error": self.detection_error,
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
            ocr_text=str(metadata.get("ocr_text", "")),
            source_type=str(metadata.get("source_type", "standalone_image")),
            width=int(metadata.get("width", 0)),
            height=int(metadata.get("height", 0)),
            detections=list(metadata.get("detections", [])),
            detection_summary=str(metadata.get("detection_summary", "")),
            detection_counts=dict(metadata.get("detection_counts", {})),
            detection_context=str(metadata.get("detection_context", "")),
            detection_error=str(metadata.get("detection_error", "")),
        )

    def ensure_ocr(self) -> None:
        """If OCR text is not present but image file exists, run OCR and populate."""
        if not self.ocr_text:
            path = Path(self.image_path)
            if not path.is_file():
                candidates = [
                    Path(self.image_path),
                    PROJECT_ROOT / self.image_path,
                    PROJECT_ROOT / "data" / "images" / path.name,
                    PROJECT_ROOT / "data" / path.name,
                    Path("data") / "images" / path.name,
                    Path("data") / path.name,
                    Path("data") / "images_staging" / path.name,
                ]
                for cand in candidates:
                    if cand.is_file():
                        path = cand
                        self.image_path = str(cand)
                        break

            if path.is_file():
                try:
                    from src.ocr import extract_ocr_text
                    self.ocr_text = extract_ocr_text(path)
                except Exception as err:
                    import logging
                    logging.getLogger(__name__).warning("OCR failed on %s: %s", path, err)

    def ensure_detections(self) -> None:
        """If detections are not present but image file exists, run YOLO26n and populate."""
        if not self.detections and not self.detection_context and not self.detection_error:
            path = Path(self.image_path)
            if not path.is_file():
                candidates = [
                    Path(self.image_path),
                    PROJECT_ROOT / self.image_path,
                    PROJECT_ROOT / "data" / "images" / path.name,
                    PROJECT_ROOT / "data" / path.name,
                    Path("data") / "images" / path.name,
                    Path("data") / path.name,
                    Path("data") / "images_staging" / path.name,
                ]
                for cand in candidates:
                    if cand.is_file():
                        path = cand
                        self.image_path = str(cand)
                        break

            if path.is_file():
                try:
                    if not YOLO_MODEL_PATH.is_file():
                        raise FileNotFoundError(
                            f"YOLO model file not found at: {YOLO_MODEL_PATH}"
                        )
                    from src.object_detection import ObjectDetector
                    detector = ObjectDetector(model_name=str(YOLO_MODEL_PATH))
                    det_res = detector.detect(path)
                    self.detections = [d.to_dict() for d in det_res.detections]
                    self.detection_summary = det_res.summary
                    self.detection_counts = det_res.counts
                    self.detection_context = det_res.rag_context
                    self.detection_error = ""
                except Exception as det_err:
                    self.detection_error = str(det_err)
                    self.detection_summary = f"Detection unavailable: {det_err}"
                    self.detection_context = f"OBJECT DETECTION ERROR\nCould not perform YOLO26n object detection: {det_err}"

    def to_evidence(self, score: float) -> RetrievedEvidence:
        self.ensure_ocr()
        self.ensure_detections()
        caption = self.caption.strip()
        ocr_text = self.ocr_text.strip()

        evidence_parts = []
        if caption and "Caption unavailable" not in caption:
            evidence_parts.append(f"Visual Description: {caption}")
        if ocr_text:
            evidence_parts.append(f"Visible Text (OCR):\n{ocr_text}")
        elif caption:
            evidence_parts.append(caption)
        else:
            evidence_parts.append(f"Image from {self.document_name}")

        full_text = "\n\n".join(evidence_parts)

        return RetrievedEvidence(
            modality="image",
            document_name=self.document_name,
            score=float(score),
            text=full_text,
            source_type=self.source_type,
            image_id=self.image_id,
            image_path=self.image_path,
            page_number=self.page_number,
            caption=caption or ocr_text or f"Image from {self.document_name}",
            source_filename=self.source_filename,
            extra={
                "ocr_text": self.ocr_text,
                "caption": self.caption,
                "detections": self.detections,
                "detection_summary": self.detection_summary,
                "detection_counts": self.detection_counts,
                "detection_context": self.detection_context,
                "detection_error": self.detection_error,
            },
        )

    def to_detection_evidence(self, score: float = 1.0) -> RetrievedEvidence:
        self.ensure_detections()
        if self.detection_error:
            text_context = (
                f"OBJECT DETECTION EVIDENCE (ERROR)\n"
                f"Source: {self.document_name}\n"
                f"Status: Detection failed ({self.detection_error})"
            )
        elif self.detection_context:
            text_context = self.detection_context
        else:
            if self.detections:
                lines = ["OBJECT DETECTION EVIDENCE", self.detection_summary]
                for idx, d in enumerate(self.detections, start=1):
                    lbl = d.get("label", "object")
                    conf = float(d.get("confidence", 0.0))
                    x1 = float(d.get("x1", 0.0))
                    y1 = float(d.get("y1", 0.0))
                    x2 = float(d.get("x2", 0.0))
                    y2 = float(d.get("y2", 0.0))
                    lines.append(f"{idx}. {lbl} | confidence={conf:.3f} | bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                text_context = "\n".join(lines)
            else:
                text_context = (
                    f"OBJECT DETECTION EVIDENCE\n"
                    f"Source: {self.document_name}\n"
                    f"No objects were detected above the confidence threshold."
                )

        return RetrievedEvidence(
            modality="text",
            document_name=self.document_name,
            score=float(score),
            text=text_context,
            source_type="object_detection",
            image_id=self.image_id,
            image_path=self.image_path,
            page_number=self.page_number,
            caption=self.detection_summary or self.caption,
            source_filename=self.source_filename,
            extra={
                "ocr_text": self.ocr_text,
                "detections": self.detections,
                "detection_summary": self.detection_summary,
                "detection_counts": self.detection_counts,
                "detection_context": text_context,
                "detection_error": self.detection_error,
            },
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

        dirty = False
        for record in records:
            if not record.ocr_text:
                record.ensure_ocr()
                if record.ocr_text:
                    dirty = True
            if not record.detections or not record.detection_context:
                record.ensure_detections()
                dirty = True

        store = cls(index=index, records=records)
        if dirty or payload.get("version", 1) < IMAGE_METADATA_VERSION:
            try:
                store.save(index_path, metadata_path)
            except Exception:
                pass

        return store

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

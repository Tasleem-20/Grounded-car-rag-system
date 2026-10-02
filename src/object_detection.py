"""YOLO object detection for CAR-RAG image analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_MODEL = "yolo26n.pt"


class ObjectDetectionError(Exception):
    """Raised when object detection cannot be performed."""


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Detection has no attribute {item!r}")

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass
class DetectionResult:
    image_path: str
    detections: list[Detection]
    annotated_image: Image.Image

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"DetectionResult has no attribute {item!r}")

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(item.label for item in self.detections))

    @property
    def summary(self) -> str:
        if not self.detections:
            return "No objects were detected."

        parts = [
            f"{count} {label}"
            for label, count in sorted(self.counts.items())
        ]

        return "Detected: " + ", ".join(parts) + "."

    @property
    def rag_context(self) -> str:
        lines = [
            "OBJECT DETECTION EVIDENCE",
            self.summary,
        ]

        for index, item in enumerate(self.detections, start=1):
            lines.append(
                f"{index}. {item.label} | "
                f"confidence={item.confidence:.3f} | "
                f"bbox=({item.x1:.1f},"
                f"{item.y1:.1f},"
                f"{item.x2:.1f},"
                f"{item.y2:.1f})"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "annotated_image": self.annotated_image,
            "detections": [d.to_dict() for d in self.detections],
            "counts": self.counts,
            "summary": self.summary,
            "rag_context": self.rag_context,
        }


class ObjectDetector:
    """Reusable YOLO inference service."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence: float = 0.25,
    ) -> None:

        if YOLO is None:
            raise ObjectDetectionError(
                "Ultralytics is not installed. "
                "Add 'ultralytics' to requirements.txt."
            )

        self.model_name = model_name
        self.confidence = confidence

        try:
            self.model = YOLO(model_name)
        except Exception as error:
            raise ObjectDetectionError(
                f"Could not load YOLO model "
                f"'{model_name}': {error}"
            ) from error

    def detect(
        self,
        image_path: str | Path,
        confidence: float | None = None,
        image_size: int = 640,
    ) -> DetectionResult:

        path = Path(image_path)

        if not path.is_file():
            raise ObjectDetectionError(
                f"Image file not found: {path}"
            )

        conf = float(
            confidence
            if confidence is not None
            else self.confidence
        )

        try:
            results = self.model.predict(
                source=str(path),
                conf=conf,
                imgsz=image_size,
                verbose=False,
            )
        except Exception as error:
            raise ObjectDetectionError(
                f"YOLO inference failed: {error}"
            ) from error

        if not results:
            raise ObjectDetectionError(
                "YOLO returned no result."
            )

        result = results[0]

        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)

        detections: list[Detection] = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()

            for coords, score, class_id in zip(
                xyxy,
                confs,
                classes,
            ):
                detections.append(
                    Detection(
                        label=names.get(
                            int(class_id),
                            str(int(class_id)),
                        ),
                        confidence=float(score),
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                    )
                )

        try:
            plotted = result.plot()

            annotated_image = Image.fromarray(
                plotted[..., ::-1]
            )

        except Exception as error:
            raise ObjectDetectionError(
                f"Could not render detection result: {error}"
            ) from error

        return DetectionResult(
            image_path=str(path),
            detections=detections,
            annotated_image=annotated_image,
        )
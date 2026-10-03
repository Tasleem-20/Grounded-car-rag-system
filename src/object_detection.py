"""YOLO object detection for CAR-RAG image analysis."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("yolo_detector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_MODEL = str(YOLO_MODEL_PATH)


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
    """Reusable YOLO inference service with CPU execution and full diagnostics."""

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

        model_path = Path(model_name)
        if not model_path.is_file():
            cand = PROJECT_ROOT / model_path.name
            if cand.is_file():
                self.model_name = str(cand)
            else:
                raise ObjectDetectionError(
                    f"YOLO model file not found: '{model_name}'. Please ensure yolo26n.pt exists in the repository root."
                )
        else:
            self.model_name = str(model_path.resolve())

        model_file = Path(self.model_name)
        file_size = model_file.stat().st_size if model_file.is_file() else 0
        logger.info("YOLO MODEL PATH: %s", self.model_name)
        logger.info("YOLO MODEL EXISTS: %s (size: %d bytes)", model_file.is_file(), file_size)

        self.confidence = confidence

        try:
            try:
                self.model = YOLO(self.model_name, task="detect")
            except TypeError:
                self.model = YOLO(self.model_name)
        except Exception as error:
            logger.error("Could not load YOLO model '%s': %s", self.model_name, error, exc_info=True)
            raise ObjectDetectionError(
                f"Could not load YOLO model '{self.model_name}': {error}"
            ) from error

    def detect(
        self,
        image_path: str | Path,
        confidence: float | None = None,
        image_size: int = 640,
    ) -> DetectionResult:

        path = Path(image_path)

        if not path.is_file():
            candidates = [
                PROJECT_ROOT / path,
                PROJECT_ROOT / "data" / "images" / path.name,
                PROJECT_ROOT / "data" / path.name,
                Path("data") / "images" / path.name,
                Path("data") / path.name,
            ]
            for cand in candidates:
                if cand.is_file():
                    path = cand
                    break

        if not path.is_file():
            logger.error("Image file not found: %s", image_path)
            raise ObjectDetectionError(f"Image file not found: {image_path}")

        # Verify actual image content with PIL
        try:
            with Image.open(path) as img:
                img_format = img.format
                img_size = img.size
                img_mode = img.mode
                logger.info("IMAGE PATH: %s", path)
                logger.info("IMAGE EXISTS: True")
                logger.info("IMAGE FORMAT: %s, SIZE: %s, MODE: %s", img_format, img_size, img_mode)
        except Exception as img_err:
            logger.warning("Could not read PIL image metadata for %s: %s", path, img_err)

        conf = float(
            confidence
            if confidence is not None
            else self.confidence
        )

        logger.info("YOLO TASK: detect")
        logger.info("YOLO CONFIDENCE: %.3f", conf)
        logger.info("YOLO DEVICE: cpu")

        try:
            results = self.model.predict(
                source=str(path),
                conf=conf,
                imgsz=image_size,
                device="cpu",
                workers=0,
                verbose=False,
            )
        except Exception as error:
            logger.error("YOLO inference failed on %s: %s", path, error, exc_info=True)
            raise ObjectDetectionError(
                f"YOLO inference failed: {error}"
            ) from error

        if not results:
            logger.error("YOLO returned empty results list for %s", path)
            raise ObjectDetectionError(
                "YOLO returned no result."
            )

        result = results[0]

        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)

        raw_box_count = len(boxes) if boxes is not None else 0
        logger.info("YOLO CLASSES: %s", list(names.values()) if isinstance(names, dict) else names)
        logger.info("RAW BOX COUNT: %d (at conf=%.2f)", raw_box_count, conf)

        # Diagnostic comparison for confidence threshold
        try:
            if abs(conf - 0.25) < 1e-4:
                diag_results = self.model.predict(
                    source=str(path),
                    conf=0.10,
                    imgsz=image_size,
                    device="cpu",
                    workers=0,
                    verbose=False,
                )
                diag_boxes = getattr(diag_results[0], "boxes", None) if diag_results else None
                diag_count = len(diag_boxes) if diag_boxes is not None else 0
                logger.info("YOLO BOX COUNT @0.25: %d", raw_box_count)
                logger.info("YOLO BOX COUNT @0.10: %d", diag_count)
        except Exception as diag_err:
            logger.debug("Diagnostic confidence check failed: %s", diag_err)

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
            logger.error("Could not render detection plot for %s: %s", path, error)
            raise ObjectDetectionError(
                f"Could not render detection result: {error}"
            ) from error

        return DetectionResult(
            image_path=str(path),
            detections=detections,
            annotated_image=annotated_image,
        )
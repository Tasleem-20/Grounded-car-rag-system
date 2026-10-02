"""OCR extraction service using RapidOCR ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ocr_engine = None


def get_ocr_engine():
    """Lazy initialize RapidOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception as err:
            logger.warning("RapidOCR could not be initialized: %s", err)
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def extract_ocr_text(image_path: str | Path) -> str:
    """Extract all text lines from an image file using OCR."""
    path = Path(image_path)
    if not path.is_file():
        return ""

    engine = get_ocr_engine()
    if engine is None:
        return ""

    try:
        results, _ = engine(str(path))
        if not results:
            return ""

        lines = []
        for item in results:
            # item structure in RapidOCR: [box_coordinates, text_str, confidence_score]
            if len(item) >= 2 and item[1]:
                text = str(item[1]).strip()
                if text:
                    lines.append(text)

        return "\n".join(lines).strip()
    except Exception as error:
        logger.warning("OCR extraction failed for %s: %s", path.name, error)
        return ""

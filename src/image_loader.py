"""Standalone image loading and PDF visual-page extraction."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from src.document_loader import DocumentLoadError, clean_text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_PAGE_TEXT_CHARS = 80
PDF_RENDER_SCALE = 2.0


class ImageLoadError(Exception):
    """Raised when an image cannot be read or saved."""


@dataclass(frozen=True)
class LoadedImage:
    """A preserved raster image plus identity metadata."""

    image_id: str
    source_filename: str
    document_name: str
    path: Path
    page_number: int | None
    source_type: str
    width: int
    height: int


def is_image_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _open_rgb_image(content: bytes, filename: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception as error:
        raise ImageLoadError(f"Unable to read image {filename}: {error}") from error

    if image.mode == "RGB":
        return image
    if image.mode == "L":
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[-1])
    return background


def save_standalone_image(
    filename: str,
    content: bytes,
    destination_dir: Path,
) -> LoadedImage:
    """Validate, convert to RGB, and preserve the uploaded raster file."""

    if not content:
        raise ImageLoadError(f"{filename} is empty.")
    if not is_image_filename(filename):
        raise ImageLoadError(f"Unsupported image type for {filename}.")

    image = _open_rgb_image(content, filename)
    destination_dir.mkdir(parents=True, exist_ok=True)

    original_suffix = Path(filename).suffix.lower()
    image_id = uuid.uuid4().hex[:12]
    stem = Path(filename).stem.replace(" ", "_")[:48]
    stored_name = f"{stem}_{image_id}{original_suffix}"
    path = destination_dir / stored_name

    format_name = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }[original_suffix]
    save_kwargs: dict = {}
    if format_name == "JPEG":
        save_kwargs["quality"] = 92
    image.save(path, format=format_name, **save_kwargs)

    return LoadedImage(
        image_id=image_id,
        source_filename=Path(filename).name,
        document_name=Path(filename).name,
        path=path,
        page_number=None,
        source_type="standalone_image",
        width=image.width,
        height=image.height,
    )


def _page_text(reader: PdfReader, page_index: int) -> str:
    try:
        raw = reader.pages[page_index].extract_text() or ""
    except Exception:
        raw = ""
    return clean_text(raw)


def extract_pdf_visual_pages(
    filename: str,
    content: bytes,
    destination_dir: Path,
    min_text_chars: int = MIN_PAGE_TEXT_CHARS,
) -> list[LoadedImage]:
    """Render PDF pages that have little or no extractable text."""

    if not content:
        return []

    try:
        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
    except Exception:
        return []

    try:
        import pypdfium2 as pdfium
    except Exception as error:
        raise ImageLoadError(
            f"PDF visual extraction requires pypdfium2: {error}"
        ) from error

    try:
        pdf = pdfium.PdfDocument(content)
    except Exception as error:
        raise ImageLoadError(
            f"Unable to render pages from {filename}: {error}"
        ) from error

    destination_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[LoadedImage] = []
    doc_name = Path(filename).name
    stem = Path(filename).stem.replace(" ", "_")[:48]

    try:
        render_count = min(page_count, len(pdf))
        for page_index in range(render_count):
            text = _page_text(reader, page_index)
            if len(text) >= min_text_chars:
                continue

            try:
                page = pdf[page_index]
                bitmap = page.render(scale=PDF_RENDER_SCALE)
                pil_image = bitmap.to_pil().convert("RGB")
            except Exception:
                continue

            image_id = uuid.uuid4().hex[:12]
            stored_name = f"{stem}_p{page_index + 1}_{image_id}.png"
            path = destination_dir / stored_name
            pil_image.save(path, format="PNG")
            loaded.append(
                LoadedImage(
                    image_id=image_id,
                    source_filename=doc_name,
                    document_name=doc_name,
                    path=path,
                    page_number=page_index + 1,
                    source_type="pdf_page_image",
                    width=pil_image.width,
                    height=pil_image.height,
                )
            )
    finally:
        pdf.close()

    return loaded

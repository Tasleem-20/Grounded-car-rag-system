from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

_MIME_TO_TYPE = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
}

_TYPE_TO_EXTENSION = {
    "pdf": ".pdf",
    "txt": ".txt",
    "png": ".png",
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "webp": ".webp",
}


def is_image_filename(filename: str) -> bool:
    """Return True when the filename represents an image."""

    return Path(filename).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }


def detect_file_type(filename: str, content: bytes, mime: str = "") -> str:
    """
    Detect PDF, TXT, PNG, JPG, or WEBP without requiring a visible extension.

    Priority: magic bytes → MIME type → filename extension → UTF-8 text fallback.
    """

    mime = (mime or "").split(";")[0].strip().lower()
    suffix = Path(str(filename)).suffix.lower()

    if content.startswith(b"%PDF"):
        return "pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"

    if mime in _MIME_TO_TYPE:
        return _MIME_TO_TYPE[mime]

    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpg"
    if suffix == ".webp":
        return "webp"

    try:
        decoded = content.decode("utf-8")
        if decoded.strip():
            return "txt"
    except UnicodeDecodeError:
        pass

    return "unknown"


def internal_loader_filename(original_name: str, detected_type: str) -> str:
    """Add an internal extension for loaders without changing the visible filename."""

    original_name = Path(str(original_name)).name.replace("\x00", "")
    if Path(original_name).suffix.lower() in SUPPORTED_EXTENSIONS:
        return original_name

    extension = _TYPE_TO_EXTENSION.get(detected_type.lower())
    if extension:
        return f"{original_name}{extension}"
    return original_name


def clean_text(text: str) -> str:
    """Clean extracted document text."""

    if not text:
        return ""

    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Uses words rather than characters so chunks remain
    reasonably readable for retrieval.
    """

    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks


def _read_uploaded_content(source: Any) -> bytes:
    """Accept bytes, file-like objects, or Streamlit UploadedFile."""

    if isinstance(source, bytes):
        return source

    if isinstance(source, bytearray):
        return bytes(source)

    if hasattr(source, "getvalue"):
        return source.getvalue()

    if hasattr(source, "read"):
        return source.read()

    raise TypeError(
        "Unsupported document input. Expected bytes or a file-like object."
    )


def load_document(filename: str, source: Any) -> str:
    """
    Extract text from PDF or TXT.

    The filename may have an internally-added extension.
    """

    extension = Path(filename).suffix.lower()
    content = _read_uploaded_content(source)

    if extension == ".txt":
        return content.decode("utf-8-sig", errors="replace")

    if extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""

    if extension in {".png", ".jpg", ".jpeg", ".webp"}:
        return ""

    raise ValueError(f"Unsupported document type: {extension or 'unknown'}")


def extract_pdf_visuals(
    source: Any,
    output_dir: str | Path,
    source_filename: str = "document.pdf",
    unique_prefix: str = "",
) -> list[dict]:
    """
    Render PDF pages as images.

    The returned visual records retain the ORIGINAL PDF filename
    so the PDF remains one logical document in the RAG system.
    Extracted image filenames are internal evidence only.
    """

    import pypdfium2 as pdfium

    content = _read_uploaded_content(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(content)
    visuals: list[dict] = []
    filename_stem = Path(source_filename).stem.replace(" ", "_")[:48]
    stored_stem = unique_prefix or filename_stem

    try:
        for page_index in range(len(pdf)):
            try:
                page = pdf[page_index]
                width_pt, height_pt = page.get_size()
                max_pt = max(width_pt, height_pt)
                scale = min(1.8, 2048.0 / max_pt) if max_pt > 0 else 1.8
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB")

                image_name = f"{stored_stem}_page_{page_index + 1}.png"
                image_path = output_dir / image_name
                image.save(image_path, format="PNG")

                visuals.append(
                    {
                        "image_path": str(image_path),
                        "page_number": page_index + 1,
                        "source_filename": source_filename,
                        "width": image.width,
                        "height": image.height,
                    }
                )
                image.close()
            except Exception:
                continue
    finally:
        pdf.close()

    return visuals

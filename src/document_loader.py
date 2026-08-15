"""Document loading and text normalization utilities."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import unicodedata

from pypdf import PdfReader


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass(frozen=True)
class LoadedDocument:
    """Normalized text and identity metadata for one source document."""

    name: str
    text: str


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""

    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _read_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as error:
        raise DocumentLoadError(f"Unable to read the PDF file: {error}") from error
    return clean_text("\n\n".join(pages))


def load_document(filename: str, content: bytes | None = None) -> LoadedDocument:
    """Load a PDF or TXT file from bytes, returning cleaned text."""

    path = Path(filename)
    raw_content = content if content is not None else path.read_bytes()
    extension = path.suffix.lower()

    if extension == ".pdf":
        text = _read_pdf(raw_content)
    elif extension == ".txt":
        try:
            text = clean_text(raw_content.decode("utf-8-sig"))
        except UnicodeDecodeError as error:
            raise DocumentLoadError(
                f"Unable to decode {path.name} as UTF-8 text."
            ) from error
    else:
        raise DocumentLoadError(
            f"Unsupported file type for {path.name}. Upload a PDF or TXT file."
        )

    if not text:
        raise DocumentLoadError(
            f"{path.name} did not contain extractable text. "
            "Scanned PDFs need OCR before they can be indexed."
        )

    return LoadedDocument(name=path.name, text=text)
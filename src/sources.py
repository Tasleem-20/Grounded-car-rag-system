"""Normalize uploaded filenames and restrict evidence to one logical source."""

from __future__ import annotations

from pathlib import Path

from src.evidence import RetrievedEvidence

KNOWN_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

CERTIFICATE_QUESTION_TERMS = (
    "certificate",
    "certification",
    "this document",
    "the document",
    "this cert",
    "the cert",
)


def safe_filename(name: str) -> str:
    """Return a path-safe basename."""

    return Path(str(name)).name.replace("\x00", "")


def normalize_source_name(name: str) -> str:
    """
    Treat these as the same logical source:

    Soft Skills
    Soft Skills.pdf
    SOFT SKILLS.PDF
    """

    cleaned = safe_filename(str(name)).strip().lower()

    for _ in range(3):
        suffix = Path(cleaned).suffix.lower()
        if suffix in KNOWN_EXTENSIONS:
            cleaned = Path(cleaned).stem.lower()
        else:
            break

    return " ".join(cleaned.replace("_", " ").split())


def same_source_name(first: str, second: str) -> bool:
    return normalize_source_name(first) == normalize_source_name(second)


def evidence_source_names(item: RetrievedEvidence) -> list[str]:
    """Collect identity fields that may describe the uploaded file."""

    names = [item.document_name]
    extra = getattr(item, "extra", None) or {}

    for candidate in (
        getattr(item, "source_filename", None),
        extra.get("source_filename"),
    ):
        if candidate:
            names.append(str(candidate))

    return [name for name in names if name]


def filter_evidence_to_source(
    results: list[RetrievedEvidence],
    source_name: str,
) -> list[RetrievedEvidence]:
    """Keep only evidence that belongs to the selected uploaded file."""

    filtered: list[RetrievedEvidence] = []

    for result in results:
        if any(
            same_source_name(candidate, source_name)
            for candidate in evidence_source_names(result)
        ):
            filtered.append(result)

    return filtered


def match_source_from_question(
    question: str,
    source_names: list[str],
) -> str | None:
    """Return a unique uploaded filename mentioned in the question, if any."""

    lowered = question.lower()
    matches: list[str] = []

    for name in source_names:
        variants = {
            name.lower(),
            Path(name).name.lower(),
            normalize_source_name(name),
            Path(name).stem.lower(),
        }
        variants = {item for item in variants if item and len(item) >= 3}

        if any(variant in lowered for variant in variants):
            matches.append(name)

    unique: list[str] = []
    for name in matches:
        if not any(same_source_name(name, existing) for existing in unique):
            unique.append(name)

    if len(unique) == 1:
        return unique[0]

    return None


def retrieval_mode_for_selection(
    file_type: str | None,
    search_all: bool,
    default_mode: str = "both",
    has_text_index: bool = True,
    has_image_index: bool = True,
) -> str:
    """
    Choose text, image, or both retrieval.

    A selected PDF always uses both modalities. Search-all uses every
    available index in the active dataset.
    """

    if search_all:
        if has_text_index and has_image_index:
            return "both"
        if has_image_index and not has_text_index:
            return "image"
        if has_text_index and not has_image_index:
            return "text"
        return "both"

    normalized_type = str(file_type or "").strip().upper()
    if normalized_type == "PDF":
        return "both"
    if normalized_type in {"PNG", "JPG", "JPEG", "WEBP"}:
        return "image"
    if normalized_type == "TXT":
        return "text"
    return default_mode


def is_ambiguous_certificate_question(
    question: str,
    source_names: list[str],
    selected_source: str | None,
    search_all: bool,
) -> bool:
    """True when a certificate-style question does not identify a source."""

    if search_all or selected_source or len(source_names) <= 1:
        return False

    lowered = question.lower()
    mentions_certificate = any(term in lowered for term in CERTIFICATE_QUESTION_TERMS)
    if not mentions_certificate:
        return False

    return match_source_from_question(question, source_names) is None

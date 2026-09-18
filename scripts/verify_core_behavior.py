"""Core correctness checks that do not require Streamlit or Groq."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import EmbeddingService
from src.evidence import RetrievedEvidence
from src.groq_models import TEXT_MODEL, VISION_MODEL, format_groq_error, is_rate_limit_error
from src.image_embeddings import ImageEmbeddingService
from src.ingestion import detect_file_type, internal_loader_filename
from src.sources import (
    filter_evidence_to_source,
    is_ambiguous_certificate_question,
    match_source_from_question,
    normalize_source_name,
    retrieval_mode_for_selection,
    same_source_name,
)


def evidence(
    name: str,
    modality: str = "text",
    source_filename: str | None = None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        modality=modality,  # type: ignore[arg-type]
        document_name=name,
        score=0.9,
        text=f"content from {name}",
        source_type="text_chunk" if modality == "text" else "pdf_visual",
        source_filename=source_filename or name,
        page_number=1 if modality == "image" else None,
    )


def main() -> int:
    failed: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failed.append(name)

    check(
        "extensionless PDF magic bytes",
        detect_file_type("Soft Skills", b"%PDF-1.7 fake") == "pdf",
    )
    check(
        "extensionless JPEG magic bytes",
        detect_file_type("photo", b"\xff\xd8\xff\xe0rest") == "jpg",
    )
    check(
        "PNG magic bytes beat wrong extension",
        detect_file_type("notes.txt", b"\x89PNG\r\n\x1a\nrest") == "png",
    )
    check(
        "UTF-8 fallback for extensionless text",
        detect_file_type("notes", b"Hello certificate holder") == "txt",
    )
    check(
        "MIME image/jpg",
        detect_file_type("file", b"not-magic", mime="image/jpg") == "jpg",
    )
    check(
        "internal loader keeps visible name separate",
        internal_loader_filename("Soft Skills", "pdf") == "Soft Skills.pdf",
    )
    check(
        "source name normalization",
        same_source_name("Soft Skills", "SOFT SKILLS.PDF")
        and normalize_source_name("Soft Skills.pdf") == "soft skills",
    )

    mixed = [
        evidence("Soft Skills", "text"),
        evidence("Soft Skills", "image", "Soft Skills"),
        evidence("Java Full Stack", "text"),
        evidence("Java Full Stack", "image", "Java Full Stack.pdf"),
    ]
    filtered = filter_evidence_to_source(mixed, "Soft Skills.pdf")
    check(
        "selected PDF keeps text + visual evidence",
        len(filtered) == 2
        and all(same_source_name(item.document_name, "Soft Skills") for item in filtered)
        and {item.modality for item in filtered} == {"text", "image"},
    )
    check(
        "selected PDF excludes other certificate",
        not any(same_source_name(item.document_name, "Java Full Stack") for item in filtered),
    )
    check(
        "extracted image is not a selectable source match for another file",
        filter_evidence_to_source(
            [evidence("page_1_image.png", "image", "Soft Skills")],
            "Java Full Stack",
        )
        == [],
    )

    names = ["Soft Skills", "Java Full Stack", "Certificate 3"]
    check(
        "question can match an uploaded filename",
        match_source_from_question("What is on Java Full Stack?", names)
        == "Java Full Stack",
    )
    check(
        "ambiguous certificate question",
        is_ambiguous_certificate_question(
            "What course or certification is shown on the certificate?",
            names,
            selected_source=None,
            search_all=False,
        ),
    )
    check(
        "selector removes ambiguity",
        not is_ambiguous_certificate_question(
            "What course or certification is shown on the certificate?",
            names,
            selected_source="Soft Skills",
            search_all=False,
        ),
    )
    check(
        "PDF retrieval mode is both",
        retrieval_mode_for_selection("PDF", search_all=False) == "both",
    )
    check(
        "standalone image retrieval mode",
        retrieval_mode_for_selection("JPG", search_all=False) == "image",
    )
    check(
        "TXT retrieval mode",
        retrieval_mode_for_selection("TXT", search_all=False) == "text",
    )
    check(
        "search all uses both indexes when present",
        retrieval_mode_for_selection(
            "TXT",
            search_all=True,
            default_mode="text",
            has_text_index=True,
            has_image_index=True,
        )
        == "both",
    )
    check(
        "no deprecated llama model",
        "llama-3.3-70b-versatile" not in TEXT_MODEL
        and "llama-3.3-70b-versatile" not in VISION_MODEL,
        f"{TEXT_MODEL} / {VISION_MODEL}",
    )
    check(
        "EmbeddingService.embed_texts exists",
        hasattr(EmbeddingService, "embed_texts")
        and hasattr(EmbeddingService, "embed_query")
        and not hasattr(EmbeddingService, "embed_documents"),
    )
    check(
        "ImageEmbeddingService.embed_images exists",
        hasattr(ImageEmbeddingService, "embed_images")
        and hasattr(ImageEmbeddingService, "embed_query")
        and not hasattr(ImageEmbeddingService, "embed_image"),
    )
    check(
        "rate-limit 400 message",
        is_rate_limit_error(RuntimeError("Error code: 400 rate_limit_exceeded"))
        and format_groq_error(RuntimeError("rate limit reached"))
        == "Groq rate limit reached. Please retry shortly.",
    )
    check(
        "format_groq_error hides keys",
        "gsk_" not in format_groq_error(RuntimeError("key gsk_test123 failed")),
    )

    print()
    if failed:
        print("Failed:")
        for name in failed:
            print(" -", name)
        return 1
    print("All core behavior checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

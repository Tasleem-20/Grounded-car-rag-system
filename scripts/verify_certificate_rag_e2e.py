"""Comprehensive end-to-end test suite for Certificate CAR-RAG Builder."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import ArrayObject, ByteStringObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

load_dotenv(ROOT / ".env")

from src.answer_checker import AnswerChecker
from src.car_rag import CARRAG
from src.chunker import TextChunk
from src.embeddings import EmbeddingService
from src.evidence import RetrievedEvidence
from src.evidence_checker import EvidenceChecker
from src.generator import Generator
from src.groq_models import TEXT_MODEL, VISION_MODEL, format_groq_error, is_rate_limit_error
from src.image_embeddings import ImageEmbeddingService
from src.image_store import ImageRecord, ImageStore
from src.ingestion import (
    chunk_text,
    clean_text,
    detect_file_type,
    extract_pdf_visuals,
    internal_loader_filename,
    load_document,
)
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.sources import (
    filter_evidence_to_source,
    is_ambiguous_certificate_question,
    match_source_from_question,
    normalize_source_name,
    retrieval_mode_for_selection,
    same_source_name,
)
from src.vector_store import VectorStore


def create_mock_certificate_pdf(path: Path, title: str, recipient: str, course: str, grade: str) -> None:
    """Create a valid PDF with text and visual elements for certificate testing."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]

    # Setup Helvetica font
    font = DictionaryObject()
    font.update({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
        NameObject("/Name"): NameObject("/F1"),
    })
    font_ref = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
    page[NameObject("/Resources")] = resources

    text_stream = (
        f"BT /F1 18 Tf 72 700 Td ({title}) Tj "
        f"/F1 14 Tf 0 -40 Td (Awarded to: {recipient}) Tj "
        f"0 -30 Td (Course: {course}) Tj "
        f"0 -30 Td (Final Grade: {grade}) Tj ET"
    ).encode("latin-1")

    stream = DecodedStreamObject()
    stream.set_data(text_stream)
    stream_ref = writer._add_object(stream)
    page[NameObject("/Contents")] = stream_ref
    writer.write(path)


def create_mock_badge_image(path: Path, title: str, color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (600, 400), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 580, 380), outline=(255, 255, 255), width=5)
    draw.text((50, 180), title, fill=(255, 255, 255))
    img.save(path, format="JPEG")


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    print("=" * 60)
    print("RUNNING CERTIFICATE CAR-RAG END-TO-END VERIFICATION")
    print("=" * 60)

    # 1. Automatic file type detection
    check(
        "PDF magic bytes (no extension)",
        detect_file_type("Soft Skills", b"%PDF-1.4\n...") == "pdf",
    )
    check(
        "JPEG magic bytes (no extension)",
        detect_file_type("Badge", b"\xff\xd8\xff\xe0...") == "jpg",
    )
    check(
        "PNG magic bytes (no extension)",
        detect_file_type("Seal", b"\x89PNG\r\n\x1a\n...") == "png",
    )
    check(
        "WEBP magic bytes (no extension)",
        detect_file_type("Stamp", b"RIFF\x00\x00\x00\x00WEBPVP8 ...") == "webp",
    )
    check(
        "UTF-8 fallback (no extension)",
        detect_file_type("Notes", b"This is a certificate syllabus.") == "txt",
    )

    # 2. Source normalization
    check(
        "Normalize certificate names",
        same_source_name("Soft Skills", "Soft Skills.pdf")
        and same_source_name("Soft Skills", "SOFT SKILLS.PDF")
        and same_source_name("soft_skills", "Soft Skills")
        and normalize_source_name("Soft Skills.pdf") == "soft skills",
    )

    # 3. Create simulated upload batch
    tmp_dir = Path(tempfile.mkdtemp(prefix="cert_rag_e2e_"))
    try:
        # File 1: Extensionless certificate PDF
        cert1_path = tmp_dir / "Soft Skills"
        create_mock_certificate_pdf(
            cert1_path,
            title="Certificate of Professional Achievement",
            recipient="Jane Doe",
            course="Executive Leadership & Soft Skills",
            grade="Grade A+",
        )

        # File 2: Extensionless second certificate PDF
        cert2_path = tmp_dir / "Java Full Stack"
        create_mock_certificate_pdf(
            cert2_path,
            title="Certificate of Competency",
            recipient="John Smith",
            course="Java Full Stack Engineering",
            grade="Grade Distinction",
        )

        # File 3: Standalone image (extensionless)
        badge_path = tmp_dir / "Gold Badge"
        create_mock_badge_image(badge_path, "GOLD STAR EXCELLENCE BADGE", (212, 175, 55))

        # File 4: Text document
        syllabus_path = tmp_dir / "Syllabus.txt"
        syllabus_path.write_text(
            "The Executive Leadership course syllabus covers Conflict Resolution and Team Dynamics.",
            encoding="utf-8",
        )

        # 4. Ingestion & Index Building
        prepared_files = [
            ("Soft Skills", cert1_path.read_bytes(), detect_file_type("Soft Skills", cert1_path.read_bytes())),
            ("Java Full Stack", cert2_path.read_bytes(), detect_file_type("Java Full Stack", cert2_path.read_bytes())),
            ("Gold Badge", badge_path.read_bytes(), detect_file_type("Gold Badge", badge_path.read_bytes())),
            ("Syllabus.txt", syllabus_path.read_bytes(), detect_file_type("Syllabus.txt", syllabus_path.read_bytes())),
        ]

        check("Detected all 4 files correctly", len(prepared_files) == 4 and prepared_files[0][2] == "pdf" and prepared_files[2][2] == "jpg")

        staging_dir = tmp_dir / "staging"
        final_images_dir = tmp_dir / "images"
        staging_dir.mkdir(parents=True, exist_ok=True)
        final_images_dir.mkdir(parents=True, exist_ok=True)

        text_chunks: list[TextChunk] = []
        image_records: list[ImageRecord] = []
        active_files: list[dict] = []

        for file_idx, (name, content, ftype) in enumerate(prepared_files, start=1):
            active_files.append({"name": name, "type": ftype.upper(), "modality": "image" if ftype in {"png", "jpg", "webp"} else "text"})

            if ftype in {"png", "jpg", "webp"}:
                img_dest = staging_dir / f"{file_idx}_{name}.jpg"
                img_dest.write_bytes(content)
                record = ImageRecord(
                    image_id=f"{normalize_source_name(name)}_{file_idx}",
                    modality="image",
                    document_name=name,
                    source_filename=name,
                    image_path=str(final_images_dir / img_dest.name),
                    page_number=None,
                    caption=f"Image of {name}",
                    source_type="standalone_image",
                    width=600,
                    height=400,
                )
                image_records.append(record)
                continue

            loader_name = internal_loader_filename(name, ftype)
            extracted_text = load_document(loader_name, io.BytesIO(content))
            cleaned = clean_text(extracted_text)
            if cleaned:
                chunks = chunk_text(cleaned, chunk_size=800, chunk_overlap=120)
                for c_idx, c_text in enumerate(chunks):
                    text_chunks.append(TextChunk(document_name=name, chunk_index=c_idx, text=c_text, start_char=0, end_char=len(c_text)))

            if ftype == "pdf":
                visuals = extract_pdf_visuals(
                    io.BytesIO(content),
                    output_dir=staging_dir,
                    source_filename=name,
                    unique_prefix=f"{normalize_source_name(name).replace(' ', '_')}_{file_idx}",
                )
                for v in visuals:
                    v_path = Path(v["image_path"])
                    record = ImageRecord(
                        image_id=f"{normalize_source_name(name)}_page_{v['page_number']}_{file_idx}",
                        modality="image",
                        document_name=name,
                        source_filename=name,
                        image_path=str(final_images_dir / v_path.name),
                        page_number=v["page_number"],
                        caption=f"Visual page {v['page_number']} of certificate {name}",
                        source_type="pdf_visual",
                        width=v["width"],
                        height=v["height"],
                    )
                    image_records.append(record)

        # Move staged images
        for f in staging_dir.iterdir():
            shutil.copy(f, final_images_dir / f.name)

        check("PDF text chunks created", any(c.document_name == "Soft Skills" for c in text_chunks))
        check("PDF visual page records created", any(r.document_name == "Soft Skills" and r.source_type == "pdf_visual" for r in image_records))
        check(
            "Extracted visual pages preserve original PDF filename",
            all(r.document_name in ["Soft Skills", "Java Full Stack", "Gold Badge"] for r in image_records),
        )

        # 5. Embeddings & Vector Stores
        text_embedder = EmbeddingService()
        image_embedder = ImageEmbeddingService()

        text_embeddings = text_embedder.embed_texts([c.text for c in text_chunks])
        text_store = VectorStore.from_embeddings(text_chunks, text_embeddings)

        image_embeddings = image_embedder.embed_images([final_images_dir / Path(r.image_path).name for r in image_records])
        image_store = ImageStore.from_embeddings(image_records, image_embeddings)

        multimodal_retriever = MultimodalRetriever(
            text_store=text_store,
            image_store=image_store,
            text_embeddings=text_embedder,
            image_embeddings=image_embedder,
        )

        # 6. Test PDF retrieval (MUST retrieve BOTH text and image)
        pdf_mode = retrieval_mode_for_selection("PDF", search_all=False)
        check("PDF retrieval mode is 'both'", pdf_mode == "both")

        soft_skills_evidence = multimodal_retriever.retrieve(
            "What course or skills are on the certificate?",
            top_k=5,
            mode=pdf_mode,
            source_filter="Soft Skills",
        )
        has_pdf_text = any(e.modality == "text" and same_source_name(e.document_name, "Soft Skills") for e in soft_skills_evidence)
        has_pdf_image = any(e.modality == "image" and same_source_name(e.document_name, "Soft Skills") for e in soft_skills_evidence)
        check("Selected PDF retrieves BOTH text and visual evidence", has_pdf_text and has_pdf_image, f"evidence_count={len(soft_skills_evidence)}")

        # 7. Test Strict Filtering
        has_leak = any(not same_source_name(e.document_name, "Soft Skills") for e in soft_skills_evidence)
        check("Strict source filtering prevents other documents from appearing", not has_leak)

        # 8. Test Ambiguous Question Detection
        source_names = [f["name"] for f in active_files]
        is_ambig = is_ambiguous_certificate_question(
            "What is the grade on the certificate?",
            source_names=source_names,
            selected_source=None,
            search_all=False,
        )
        check("Ambiguous certificate question without source selector detected", is_ambig)

        is_not_ambig = is_ambiguous_certificate_question(
            "What is the grade on the certificate?",
            source_names=source_names,
            selected_source="Soft Skills",
            search_all=False,
        )
        check("Selected source resolves ambiguity", not is_not_ambig)

        # 9. Test Standalone TXT and Image mode selection
        check("TXT retrieval mode is 'text'", retrieval_mode_for_selection("TXT", search_all=False) == "text")
        check("JPG retrieval mode is 'image'", retrieval_mode_for_selection("JPG", search_all=False) == "image")
        check("Search all active files mode uses available indexes", retrieval_mode_for_selection("PDF", search_all=True, has_text_index=True, has_image_index=True) == "both")

        # 10. Test Search all active files
        all_evidence = multimodal_retriever.retrieve(
            "Show me everything about leadership and engineering",
            top_k=8,
            mode="both",
            source_filter=None,
        )
        doc_names_in_all = {e.document_name for e in all_evidence}
        check("Search all active files queries across multiple documents", len(doc_names_in_all) >= 2, f"docs={doc_names_in_all}")

        # 11. Method signature checks
        check(
            "EmbeddingService only has embed_texts and embed_query",
            hasattr(EmbeddingService, "embed_texts")
            and hasattr(EmbeddingService, "embed_query")
            and not hasattr(EmbeddingService, "embed_documents")
            and not hasattr(EmbeddingService, "embed_image"),
        )
        check(
            "ImageEmbeddingService only has embed_images and embed_query",
            hasattr(ImageEmbeddingService, "embed_images")
            and hasattr(ImageEmbeddingService, "embed_query")
            and not hasattr(ImageEmbeddingService, "embed_image")
            and not hasattr(ImageEmbeddingService, "embed_documents"),
        )

        # 12. Deprecated Groq model and error formatting checks
        check("No deprecated llama-3.3-70b-versatile in TEXT_MODEL", "llama-3.3-70b-versatile" not in TEXT_MODEL, TEXT_MODEL)
        check("No deprecated llama-3.3-70b-versatile in VISION_MODEL", "llama-3.3-70b-versatile" not in VISION_MODEL, VISION_MODEL)
        check("Active models match supported Groq IDs", TEXT_MODEL == "openai/gpt-oss-120b" and VISION_MODEL == "qwen/qwen3.8-27b", f"{TEXT_MODEL}, {VISION_MODEL}")
        check("format_groq_error masks API keys", "gsk_" not in format_groq_error(RuntimeError("Invalid key: gsk_12345abcdef67890")))

        # 13. Groq CAR-RAG Pipeline Check (if GROQ_API_KEY is available)
        if os.getenv("GROQ_API_KEY"):
            try:
                analyzer = QueryAnalyzer()
                generator = Generator()
                evidence_checker = EvidenceChecker()
                answer_checker = AnswerChecker()

                car_rag = CARRAG(
                    analyzer=analyzer,
                    retriever=multimodal_retriever,
                    generator=generator,
                    evidence_checker=evidence_checker,
                    answer_checker=answer_checker,
                )

                # Query with specific source
                rag_result = car_rag.run(
                    question="What course was Jane Doe awarded a certificate in?",
                    source_filter="Soft Skills",
                    retrieval_mode="both",
                    top_k=5,
                )
                grounded_ok = (
                    rag_result.grounding_verified
                    and rag_result.answer_supported
                    and ("Leadership" in rag_result.answer or "Soft Skills" in rag_result.answer)
                )
                check("CAR-RAG pipeline generates verified grounded certificate answer", grounded_ok, rag_result.answer[:120])
            except Exception as e:
                check("CAR-RAG pipeline execution", False, str(e)[:250])
        else:
            print("[INFO] GROQ_API_KEY not configured, skipping online Groq CAR-RAG stage call.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=" * 60)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"FAILED ({len(failed)}/{len(results)} checks):")
        for name in failed:
            print(" -", name)
        return 1

    print(f"ALL {len(results)} CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

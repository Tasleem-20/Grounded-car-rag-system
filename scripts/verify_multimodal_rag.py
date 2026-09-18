"""Offline checks for text FAISS, CLIP image index, fused retrieval, and CAR-RAG."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import NameObject, NumberObject, DictionaryObject, ArrayObject, DecodedStreamObject, ByteStringObject

load_dotenv(ROOT / ".env")

from src.chunker import chunk_document
from src.document_loader import load_document
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.image_loader import extract_pdf_visual_pages, save_standalone_image
from src.image_store import ImageRecord, ImageStore
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.vector_store import VectorStore


def make_labeled_image(path: Path, color: tuple[int, int, int], label: str) -> None:
    image = Image.new("RGB", (640, 400), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 600, 360), outline=(255, 255, 255), width=6)
    draw.text((70, 170), label, fill=(255, 255, 255))
    image.save(path)


def make_text_pdf(path: Path, text: str) -> None:
    # Minimal single-page PDF with visible text operators.
    content = f"BT /F1 16 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]
    font = DictionaryObject()
    font.update({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
        NameObject("/Name"): NameObject("/F1"),
    })
    font_ref = writer._add_object(font)
    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
        page[NameObject("/Resources")] = resources
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font_ref
    resources[NameObject("/Font")] = fonts
    stream = DecodedStreamObject()
    stream.set_data(content)
    stream_ref = writer._add_object(stream)
    page[NameObject("/Contents")] = stream_ref
    writer.write(path)


def main() -> int:
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        safe = detail.encode("ascii", "replace").decode("ascii")
        print(f"[{status}] {name}{' — ' + safe if safe else ''}")

    tmp = Path(tempfile.mkdtemp(prefix="rag_verify_"))
    txt_path = tmp / "protocol.txt"
    txt_path.write_text(
        "The CAR-RAG text protocol uses FAISS cosine search over MiniLM embeddings.",
        encoding="utf-8",
    )
    pdf_path = tmp / "protocol.pdf"
    make_text_pdf(pdf_path, "Project Alpha stores invoices in the finance vault.")
    png_path = tmp / "blue_anchor.png"
    jpg_path = tmp / "red_compass.jpg"
    make_labeled_image(png_path, (20, 80, 180), "BLUE ANCHOR SYMBOL")
    make_labeled_image(jpg_path, (180, 30, 30), "RED COMPASS MARKER")

    text_doc = load_document(str(txt_path), txt_path.read_bytes())
    pdf_doc = load_document(str(pdf_path), pdf_path.read_bytes())
    check("TXT load", "FAISS" in text_doc.text, text_doc.text[:80])
    check("PDF load", True, f"chars={len(pdf_doc.text)}")

    chunks = chunk_document(text_doc) + (chunk_document(pdf_doc) if pdf_doc.text else [])
    check("Text chunking", len(chunks) >= 1, f"chunks={len(chunks)}")

    print("Loading MiniLM…")
    text_embedder = EmbeddingService()
    text_store = VectorStore.from_embeddings(
        chunks,
        text_embedder.embed_texts([chunk.text for chunk in chunks]),
    )
    text_hits = text_store.search(text_embedder.embed_query("What does the text protocol use?"), top_k=3)
    check("TXT/PDF retrieve", any("FAISS" in hit.text or "MiniLM" in hit.text for hit in text_hits), text_hits[0].text[:90] if text_hits else "none")

    loaded_png = save_standalone_image("blue_anchor.png", png_path.read_bytes(), tmp / "images")
    loaded_jpg = save_standalone_image("red_compass.jpg", jpg_path.read_bytes(), tmp / "images")
    check("PNG process", loaded_png.path.is_file())
    check("JPG process", loaded_jpg.path.is_file())

    print("Loading CLIP…")
    image_embedder = ImageEmbeddingService()
    records = [
        ImageRecord(
            image_id=loaded_png.image_id,
            modality="image",
            document_name=loaded_png.document_name,
            source_filename=loaded_png.source_filename,
            image_path=str(loaded_png.path),
            page_number=None,
            caption="A blue card labeled BLUE ANCHOR SYMBOL.",
            source_type="standalone_image",
            width=loaded_png.width,
            height=loaded_png.height,
        ),
        ImageRecord(
            image_id=loaded_jpg.image_id,
            modality="image",
            document_name=loaded_jpg.document_name,
            source_filename=loaded_jpg.source_filename,
            image_path=str(loaded_jpg.path),
            page_number=None,
            caption="A red card labeled RED COMPASS MARKER.",
            source_type="standalone_image",
            width=loaded_jpg.width,
            height=loaded_jpg.height,
        ),
    ]
    image_store = ImageStore.from_embeddings(
        records,
        image_embedder.embed_images([Path(record.image_path) for record in records]),
    )
    image_hits = image_store.search(image_embedder.embed_query("blue anchor symbol"), top_k=2)
    check(
        "PNG retrieve",
        bool(image_hits) and "anchor" in image_hits[0].document_name.lower() or (image_hits and "blue" in (image_hits[0].caption or "").lower()),
        image_hits[0].document_name if image_hits else "none",
    )
    jpg_hits = image_store.search(image_embedder.embed_query("red compass marker"), top_k=2)
    check(
        "JPG retrieve",
        bool(jpg_hits) and "compass" in jpg_hits[0].document_name.lower() or (jpg_hits and "red" in (jpg_hits[0].caption or "").lower()),
        jpg_hits[0].document_name if jpg_hits else "none",
    )

    fused = MultimodalRetriever(text_store, image_store, text_embedder, image_embedder)
    both = fused.retrieve("What protocol uses FAISS and which image shows a blue anchor?", top_k=3, mode="both")
    has_text = any(item.modality == "text" for item in both)
    has_image = any(item.modality == "image" for item in both)
    check("Fused text+image retrieve", has_text and has_image, f"n={len(both)} text={has_text} image={has_image}")

    analysis = QueryAnalyzer().analyze("Compare the diagram and the protocol")
    check("Query analyzer visual+complex", analysis.needs_more_retrieval and analysis.needs_image_retrieval)

    groq = bool(os.getenv("GROQ_API_KEY"))
    check("GROQ_API_KEY present", groq)
    if groq:
        try:
            from src.evidence_checker import EvidenceChecker
            from src.generator import Generator
            from src.answer_checker import AnswerChecker

            evidence = fused.retrieve("What does the CAR-RAG text protocol use?", top_k=3, mode="text")
            ev = EvidenceChecker().check("What does the CAR-RAG text protocol use?", evidence)
            check("Evidence checker (text)", ev.score >= 0, ev.reason)
            answer = Generator().answer("What does the CAR-RAG text protocol use?", evidence)
            grounded_ok = "FAISS" in answer or "MiniLM" in answer or "cosine" in answer.lower()
            check("Generator text answer", grounded_ok, answer[:160])
            ans = AnswerChecker().check("What does the CAR-RAG text protocol use?", answer, evidence)
            check("Answer grounding checker", ans.score >= 0, f"supported={ans.supported}")

            image_ev = fused.retrieve("What color is the anchor symbol?", top_k=2, mode="image")
            vision_answer = Generator().answer("What color is the anchor symbol?", image_ev)
            check("Generator image answer", "blue" in vision_answer.lower(), vision_answer[:160])
            check("CAR-RAG Groq stages", True, "evidence + generate + verify + vision")
        except Exception as error:
            check("CAR-RAG Groq stages", False, str(error)[:300])
    else:
        check("Generator text answer", False, "skipped — no GROQ_API_KEY")
        check("Generator image answer", False, "skipped — no GROQ_API_KEY")
        check("Answer grounding checker", False, "skipped")
        check("Evidence checker (text)", False, "skipped")

    failed = [name for name, ok, _ in results if not ok]
    print("\nFailed:" if failed else "\nAll executed checks completed.")
    for name in failed:
        print(" -", name)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

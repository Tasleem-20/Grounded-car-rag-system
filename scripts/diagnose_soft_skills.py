"""Diagnose the exact failure when processing Soft Skills.pdf."""

import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ingestion import (
    detect_file_type,
    internal_loader_filename,
    load_document,
    clean_text,
    chunk_text,
    extract_pdf_visuals,
)
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.chunker import TextChunk
from src.image_store import ImageRecord, ImageStore
from src.vector_store import VectorStore
from src.sources import normalize_source_name, safe_filename

def main():
    pdf_path = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates\Soft Skills.pdf")
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return 1

    content = pdf_path.read_bytes()
    name = pdf_path.name
    print(f"Testing file: {name} (size: {len(content)} bytes)")

    # 1. Detection
    try:
        detected_type = detect_file_type(name, content)
        print(f"Step 1 - File type detected: {detected_type}")
    except Exception:
        traceback.print_exc()
        return 1

    # 2. Text extraction
    loader_name = internal_loader_filename(name, detected_type)
    print(f"Step 2 - Loader name: {loader_name}")
    try:
        text = load_document(loader_name, io.BytesIO(content))
        print(f"Step 2 - Extracted text length: {len(text)} chars")
        print(f"Text preview: {repr(text[:200])}")
    except Exception:
        print("ERROR in Step 2 (load_document):")
        traceback.print_exc()
        return 1

    # 3. Clean and chunk
    try:
        cleaned = clean_text(text)
        print(f"Step 3 - Cleaned text length: {len(cleaned)} chars")
        chunks = chunk_text(cleaned, chunk_size=800, chunk_overlap=120) if cleaned.strip() else []
        print(f"Step 4 - Chunk count: {len(chunks)}")
    except Exception:
        print("ERROR in Step 3/4 (clean/chunk):")
        traceback.print_exc()
        return 1

    # 4. Visual extraction
    tmp_staging = ROOT / "data" / "diag_staging"
    tmp_staging.mkdir(parents=True, exist_ok=True)
    try:
        unique_prefix = f"{normalize_source_name(name).replace(' ', '_')}_1"
        visuals = extract_pdf_visuals(
            io.BytesIO(content),
            output_dir=tmp_staging,
            source_filename=name,
            unique_prefix=unique_prefix,
        )
        print(f"Step 5 - Extracted visuals count: {len(visuals)}")
        for v in visuals:
            print(f"  Visual: {v}")
    except Exception:
        print("ERROR in Step 5 (extract_pdf_visuals):")
        traceback.print_exc()
        return 1

    # 5. Visual captioning
    try:
        from src.vision import caption_image, fallback_caption
        for v in visuals:
            vpath = Path(v["image_path"])
            print(f"Step 6 - Captioning {vpath.name}...")
            caption = caption_image(vpath, source_name=name)
            print(f"Caption result ({len(caption)} chars): {caption[:120]}...")
    except Exception:
        print("ERROR in Step 6 (caption_image):")
        traceback.print_exc()

    # 6. Image embedding
    try:
        image_embedder = ImageEmbeddingService()
        for v in visuals:
            vpath = Path(v["image_path"])
            print(f"Step 7 - Embedding image {vpath.name}...")
            emb = image_embedder.embed_images([vpath])[0]
            print(f"Embedding generated: len={len(emb)}")
    except Exception:
        print("ERROR in Step 7 (image embedding):")
        traceback.print_exc()
        return 1

    # 7. Text embedding
    if chunks:
        try:
            text_embedder = EmbeddingService()
            print(f"Step 8 - Embedding {len(chunks)} text chunks...")
            text_embs = text_embedder.embed_texts(chunks)
            print(f"Text embeddings generated: {len(text_embs)}")
        except Exception:
            print("ERROR in Step 8 (text embedding):")
            traceback.print_exc()
            return 1

    print("\nALL DIAGNOSTIC STEPS COMPLETED!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

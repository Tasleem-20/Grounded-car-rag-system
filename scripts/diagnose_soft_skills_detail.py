"""Detailed diagnostic script to test all 12 ingestion checkpoints."""

import io
import os
import sys
import time
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
from src.vision import encode_image_data_url, caption_image, fallback_caption
from PIL import Image

def test_file_diagnostics(pdf_path: Path):
    print(f"\n{'='*70}\nDIAGNOSTIC FOR: {pdf_path.name}\n{'='*70}", flush=True)
    content = pdf_path.read_bytes()
    name = pdf_path.name
    print(f"File size: {len(content)} bytes", flush=True)

    # 1. Detection
    detected_type = detect_file_type(name, content)
    print(f"[1] Detected type: {detected_type}", flush=True)

    # 2. Text extraction
    loader_name = internal_loader_filename(name, detected_type)
    text = load_document(loader_name, io.BytesIO(content))
    print(f"[2] Extracted text length: {len(text)} chars", flush=True)
    print(f"    Preview: {repr(text[:150])}", flush=True)

    # 3. Clean
    cleaned = clean_text(text)
    print(f"[3] Cleaned text length: {len(cleaned)} chars", flush=True)

    # 4. Chunk
    chunks = chunk_text(cleaned, chunk_size=800, chunk_overlap=120) if cleaned.strip() else []
    print(f"[4] Chunks count: {len(chunks)}", flush=True)

    # 5. Visual extraction
    staging = ROOT / "data" / "diag_staging"
    staging.mkdir(parents=True, exist_ok=True)
    visuals = []
    if detected_type == "pdf":
        unique_prefix = f"{normalize_source_name(name).replace(' ', '_')}_diag"
        visuals = extract_pdf_visuals(
            io.BytesIO(content),
            output_dir=staging,
            source_filename=name,
            unique_prefix=unique_prefix,
        )
        print(f"[5] Extracted PDF visuals count: {len(visuals)}", flush=True)
        for v in visuals:
            print(f"    Visual: path={v['image_path']}, size=({v.get('width')}, {v.get('height')})", flush=True)
    elif detected_type in {"png", "jpg", "webp"}:
        vpath = staging / f"diag_{name}"
        vpath.write_bytes(content)
        with Image.open(vpath) as im:
            w, h = im.size
        visuals = [{"image_path": str(vpath), "page_number": None, "source_filename": name, "width": w, "height": h}]
        print(f"[5] Image visual: {vpath}, size=({w}, {h})", flush=True)

    # 6. Check data_url encoding size
    for v in visuals:
        vpath = Path(v["image_path"])
        durl = encode_image_data_url(vpath)
        print(f"[6] Data URL length for {vpath.name}: {len(durl)} chars ({len(durl)//1024} KB)", flush=True)

    # 7. Groq vision captioning
    for v in visuals:
        vpath = Path(v["image_path"])
        print(f"[7] Calling Groq vision caption for {vpath.name}...", flush=True)
        t0 = time.time()
        try:
            caption = caption_image(vpath, source_name=name)
            elapsed = time.time() - t0
            print(f"    Caption success in {elapsed:.2f}s! ({len(caption)} chars)", flush=True)
            print(f"    Caption preview: {repr(caption[:150])}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    Caption FAILED in {elapsed:.2f}s: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

def main():
    cert_dir = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates")
    for fname in ["Soft Skills.pdf", "Qubit.jpg", "java fullstack(edu skills).pdf"]:
        p = cert_dir / fname
        if p.exists():
            test_file_diagnostics(p)

if __name__ == "__main__":
    main()

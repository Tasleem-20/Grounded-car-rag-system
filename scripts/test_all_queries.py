"""Run and verify all 6 required test queries (A-F) against CARRAG."""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.image_store import ImageStore
from src.vector_store import VectorStore
from src.image_embeddings import ImageEmbeddingService
from src.embeddings import EmbeddingService
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.evidence_checker import EvidenceChecker
from src.generator import Generator
from src.answer_checker import AnswerChecker
from src.car_rag import CARRAG

def test_all():
    api_key = os.getenv("GROQ_API_KEY", "")
    print(f"GROQ_API_KEY present: {bool(api_key)}", flush=True)

    meta_path = ROOT / "data" / "image_metadata.json"
    index_path = ROOT / "data" / "image.index"
    text_meta_path = ROOT / "data" / "text_metadata.json"
    text_index_path = ROOT / "data" / "text.index"

    image_store = ImageStore.load(index_path, meta_path) if ImageStore.exists(index_path, meta_path) else None
    text_store = VectorStore.load(text_index_path, text_meta_path) if VectorStore.exists(text_index_path, text_meta_path) else None

    print(f"ImageStore records: {len(image_store.records) if image_store else 0}", flush=True)
    if image_store:
        for r in image_store.records:
            print(f"  Record: {r.document_name} | counts: {r.detection_counts}", flush=True)

    img_embedder = ImageEmbeddingService()
    text_embedder = EmbeddingService()
    analyzer = QueryAnalyzer()
    ev_checker = EvidenceChecker(api_key=api_key)
    generator = Generator(api_key=api_key)
    ans_checker = AnswerChecker(api_key=api_key)

    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=text_embedder,
        image_embeddings=img_embedder,
    )

    car_rag = CARRAG(
        analyzer=analyzer,
        retriever=retriever,
        evidence_checker=ev_checker,
        generator=generator,
        answer_checker=ans_checker,
    )

    queries = [
        ("A", "How many cars are detected?"),
        ("B", "How many people are detected?"),
        ("C", "What objects are detected?"),
        ("D", "Which vehicles are present?"),
        ("E", "What is present in the image?"),
    ]

    import time

    for tag, q in queries:
        print(f"\n==========================================", flush=True)
        print(f"QUERY {tag}: '{q}'", flush=True)
        analysis = analyzer.analyze(q)
        print(f"QUERY TYPE: {analysis.query_type} | Needs Image: {analysis.needs_image_retrieval}", flush=True)
        
        retrieved = retriever.retrieve(question=q, top_k=5, mode="image", prefer_images=True)
        print(f"RETRIEVED ITEMS COUNT: {len(retrieved)}", flush=True)
        for i, item in enumerate(retrieved, 1):
            print(f"  [{i}] modality={item.modality} type={item.source_type} doc={item.document_name}", flush=True)

        res = car_rag.run(question=q, initial_evidence=retrieved, query_analysis=analysis)
        print(f"[{tag}] FINAL ANSWER:\n{res.answer}", flush=True)
        print(f"[{tag}] EVIDENCE SUFFICIENT: {res.evidence_sufficient}", flush=True)
        print(f"[{tag}] GROUNDING VERIFIED: {res.grounding_verified}", flush=True)
        print(f"[{tag}] ANSWER SUPPORTED: {res.answer_supported}", flush=True)
        print(f"[{tag}] EVIDENCE REASON: {res.evidence_reason}", flush=True)
        print(f"[{tag}] ANSWER REASON: {res.answer_reason}", flush=True)
        time.sleep(3)

if __name__ == "__main__":
    test_all()

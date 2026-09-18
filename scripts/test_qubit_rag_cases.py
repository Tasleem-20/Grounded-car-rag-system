"""Test Qubit and Java Full Stack queries against current active indexes and CARRAG pipeline."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.vector_store import VectorStore
from src.image_store import ImageStore
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.retriever import MultimodalRetriever
from src.query_analyzer import QueryAnalyzer
from src.car_rag import CARRAG
from src.generator import Generator
from src.evidence_checker import EvidenceChecker
from src.answer_checker import AnswerChecker
from src.sources import match_source_from_question, filter_evidence_to_source, retrieval_mode_for_selection

def main():
    text_store = VectorStore.load(Path("indexes/faiss.index"), Path("indexes/metadata.json")) if Path("indexes/faiss.index").exists() else None
    image_store = ImageStore.load(Path("indexes/image.faiss.index"), Path("indexes/image_metadata.json")) if Path("indexes/image.faiss.index").exists() else None

    print(f"Loaded Text Store: {len(text_store.chunks) if text_store else 0} chunks")
    print(f"Loaded Image Store: {len(image_store.records) if image_store else 0} records")
    if image_store:
        for r in image_store.records:
            print(f" - Image Record: doc={r.document_name}, source={r.source_filename}, type={r.source_type}, id={r.image_id}, caption_len={len(r.caption)}")

    text_emb = EmbeddingService()
    image_emb = ImageEmbeddingService()
    retriever = MultimodalRetriever(text_store, image_store, text_emb, image_emb)
    analyzer = QueryAnalyzer()
    generator = Generator() if os.getenv("GROQ_API_KEY") else None
    evidence_checker = EvidenceChecker() if os.getenv("GROQ_API_KEY") else None
    answer_checker = AnswerChecker() if os.getenv("GROQ_API_KEY") else None

    rag = CARRAG(
        analyzer=analyzer,
        retriever=retriever,
        generator=generator,
        evidence_checker=evidence_checker,
        answer_checker=answer_checker,
    )

    source_names = ["Qubit.jpg", "java fullstack(edu skills).pdf"]

    print("\n" + "="*70)
    print("TEST 1: Question: 'What is the name of the person on the Qubit certificate?'")
    print("Selected Source: 'Qubit.jpg'")
    print("="*70)
    q1 = "What is the name of the person on the Qubit certificate?"
    res1 = rag.run(
        question=q1,
        source_filter="Qubit.jpg",
        retrieval_mode="image",
        top_k=5,
    )
    def safe_print(val: str) -> str:
        return str(val).encode("ascii", "replace").decode("ascii")

    print(f"Evidence count: {len(res1.evidence)}")
    for idx, ev in enumerate(res1.evidence, 1):
        print(f"  [{idx}] doc={ev.document_name}, modality={ev.modality}, score={ev.score:.3f}")
    print(f"Evidence sufficient: {res1.evidence_sufficient} (reason: {safe_print(res1.evidence_reason)})")
    print(f"Grounding verified: {res1.grounding_verified}")
    print(f"Answer supported: {res1.answer_supported} (reason: {safe_print(res1.answer_reason)})")
    print(f"Answer: {safe_print(res1.answer)}")

    print("\n" + "="*70)
    print("TEST 2: Question: 'What certification or course is mentioned on the Qubit certificate?'")
    print("Selected Source: 'Qubit.jpg'")
    print("="*70)
    q2 = "What certification or course is mentioned on the Qubit certificate?"
    res2 = rag.run(
        question=q2,
        source_filter="Qubit.jpg",
        retrieval_mode="image",
        top_k=5,
    )
    print(f"Evidence count: {len(res2.evidence)}")
    print(f"Grounding verified: {res2.grounding_verified}")
    print(f"Answer: {safe_print(res2.answer)}")

    print("\n" + "="*70)
    print("TEST 3: Java Full Stack question: 'What is the Certificate ID of the Java Full Stack certificate?'")
    print("Selected Source: 'java fullstack(edu skills).pdf'")
    print("="*70)
    q3 = "What is the Certificate ID of the Java Full Stack certificate?"
    res3 = rag.run(
        question=q3,
        source_filter="java fullstack(edu skills).pdf",
        retrieval_mode="text",
        top_k=5,
    )
    print(f"Evidence count: {len(res3.evidence)}")
    for idx, ev in enumerate(res3.evidence, 1):
        print(f"  [{idx}] doc={ev.document_name}, modality={ev.modality}, score={ev.score:.3f}")
    print(f"Grounding verified: {res3.grounding_verified}")
    print(f"Answer: {safe_print(res3.answer)}")

    print("\n" + "="*70)
    print("TEST 4: Source Isolation: Select Qubit.jpg and ask about Java Full Stack")
    print("="*70)
    q4 = "What topics were covered during the Java Full Stack internship?"
    res4 = rag.run(
        question=q4,
        source_filter="Qubit.jpg",
        retrieval_mode="image",
        top_k=5,
    )
    print(f"Evidence count (strictly filtered to Qubit.jpg): {len(res4.evidence)}")
    for ev in res4.evidence:
        print(f"  doc={ev.document_name}")
    has_java_leak = any("java" in ev.document_name.lower() for ev in res4.evidence)
    print(f"Source isolation maintained (no Java leak): {not has_java_leak}")
    print(f"Answer: {safe_print(res4.answer)}")

    print("\n" + "="*70)
    print("TEST 5: Search All Active Files: Question requiring info from both certificates")
    print("="*70)
    q5 = "What courses or programs are completed by Shaik Tasleem across the uploaded certificates?"
    res5 = rag.run(
        question=q5,
        source_filter=None,
        retrieval_mode="both",
        top_k=6,
    )
    print(f"Evidence count: {len(res5.evidence)}")
    docs_found = {ev.document_name for ev in res5.evidence}
    print(f"Documents found in search-all: {docs_found}")
    print(f"Multi-document evidence retrieved: {len(docs_found) >= 2}")
    print(f"Evidence sufficient: {res5.evidence_sufficient} (reason: {safe_print(res5.evidence_reason)})")
    print(f"Grounding verified: {res5.grounding_verified}")
    print(f"Answer: {safe_print(res5.answer)}")

if __name__ == "__main__":
    main()

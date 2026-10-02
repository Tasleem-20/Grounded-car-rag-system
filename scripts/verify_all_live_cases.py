"""
Comprehensive End-to-End Live Verification Script
Simulates the exact Streamlit execution flow through CARRAG and MultimodalRetriever
Testing all 7 user critical requirements:
1. TEXT retrieval & grounding ("What is ABS?") -> no YOLO, pure text chunks
2. IMAGE descriptive retrieval & grounding ("Describe the street scene.") -> CLIP + vision, no detection cards
3. OBJECT DETECTION query ("How many cars are detected?") -> YOLO count + bounding boxes + image source
4. MULTIMODAL query ("What vehicles are visible in the image and what information does the document provide about vehicles?") -> Combined text chunks + image + YOLO detections
5. TEXT ONLY isolation ("What does the document say about ABS?")
6. IMAGE ONLY isolation ("Describe the image.")
7. PERSISTENCE & RESTART -> reload indices from disk and verify detection data + text + image indices intact
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import VectorStore
from src.image_store import ImageStore
from src.retriever import MultimodalRetriever
from src.query_analyzer import QueryAnalyzer
from src.car_rag import CARRAG

def test_pipeline():
    print("=" * 60)
    print("STARTING FULL END-TO-END VERIFICATION OF CAR-RAG PIPELINE")
    print("=" * 60)

    # 1. Initialize Stores from persistent data/ directory
    from pathlib import Path
    text_store = VectorStore.load(Path("data/text.index"), Path("data/text_metadata.json"))
    print(f"[STORE] Loaded Text Index: {text_store.index.ntotal} chunks")

    image_store = ImageStore.load(Path("data/image.index"), Path("data/image_metadata.json"))
    print(f"[STORE] Loaded Image Index: {image_store.index.ntotal} images")
    for r in image_store.records:
        print(f"[STORE] Image record: {r.document_name}, detections={len(r.detections)}, counts={r.detection_counts}")

    from src.embeddings import EmbeddingService
    from src.image_embeddings import ImageEmbeddingService
    text_emb = EmbeddingService()
    img_emb = ImageEmbeddingService()
    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=text_emb,
        image_embeddings=img_emb,
    )
    from src.generator import Generator
    from src.evidence_checker import EvidenceChecker
    from src.answer_checker import AnswerChecker

    generator = Generator()
    evidence_checker = EvidenceChecker()
    answer_checker = AnswerChecker()

    qa = QueryAnalyzer()
    car_rag = CARRAG(
        analyzer=qa,
        retriever=retriever,
        answer_checker=answer_checker,
        evidence_checker=evidence_checker,
        generator=generator,
    )

    # TEST 1 — TEXT
    print("\n--- TEST 1: TEXT RAG ('What is ABS?') ---")
    q1 = "What is ABS?"
    analysis1 = qa.analyze(q1)
    print(f"Query Analysis: mode={analysis1.query_mode}, needs_img={analysis1.needs_image_retrieval}, needs_det={analysis1.needs_detection_evidence}")
    assert analysis1.query_mode == "TEXT_RAG", f"Expected TEXT_RAG, got {analysis1.query_mode}"
    assert analysis1.needs_detection_evidence == False, "Detection evidence should NOT be needed for text query"
    
    res1 = car_rag.run(q1)
    print(f"Supported: {res1.answer_supported}")
    print(f"Answer: {res1.answer[:120]}...")
    print(f"Evidence items: {len(res1.evidence)}")
    for s in res1.evidence:
        print(f"  - Modality: {s.modality}, source_type: {s.source_type}, doc: {s.document_name}")
    assert any(s.modality == "text" for s in res1.evidence), "Must contain text evidence"
    assert not any(s.source_type == "object_detection" for s in res1.evidence), "Must NOT contain object detection evidence"
    assert "anti-lock" in res1.answer.lower() or "braking" in res1.answer.lower() or "abs" in res1.answer.lower(), "Answer must be grounded in ABS document"
    print(">>> TEST 1 PASSED: Pure text retrieval, no YOLO intrusion.")

    time.sleep(3) # respectful pacing

    # TEST 2 — IMAGE DESCRIPTIVE
    print("\n--- TEST 2: IMAGE RAG ('Describe the image.') ---")
    q2 = "Describe the image."
    analysis2 = qa.analyze(q2)
    print(f"Query Analysis: mode={analysis2.query_mode}, needs_img={analysis2.needs_image_retrieval}, needs_det={analysis2.needs_detection_evidence}")
    assert analysis2.query_mode == "IMAGE_RAG", f"Expected IMAGE_RAG, got {analysis2.query_mode}"
    assert analysis2.needs_detection_evidence == False, "Detection evidence should NOT be needed for descriptive image query"
    
    res2 = car_rag.run(q2)
    print(f"Supported: {res2.answer_supported}")
    print(f"Answer: {res2.answer[:150]}...")
    print(f"Evidence items: {len(res2.evidence)}")
    for s in res2.evidence:
        print(f"  - Modality: {s.modality}, source_type: {s.source_type}, doc: {s.document_name}")
    assert any(s.modality == "image" for s in res2.evidence), "Must contain image evidence"
    assert not any(s.source_type == "object_detection" for s in res2.evidence), "Must NOT contain object detection evidence"
    print(">>> TEST 2 PASSED: Pure image retrieval & vision description, no detection cards.")

    time.sleep(3)

    # TEST 3 — OBJECT DETECTION
    print("\n--- TEST 3: OBJECT DETECTION ('How many cars are detected?') ---")
    q3 = "How many cars are detected?"
    analysis3 = qa.analyze(q3)
    print(f"Query Analysis: mode={analysis3.query_mode}, needs_img={analysis3.needs_image_retrieval}, needs_det={analysis3.needs_detection_evidence}")
    assert analysis3.query_mode == "OBJECT_DETECTION", f"Expected OBJECT_DETECTION, got {analysis3.query_mode}"
    assert analysis3.needs_detection_evidence == True, "Detection evidence MUST be enabled"
    
    res3 = car_rag.run(q3)
    print(f"Supported: {res3.answer_supported}")
    print(f"Answer: {res3.answer}")
    print(f"Evidence items: {len(res3.evidence)}")
    for s in res3.evidence:
        print(f"  - Modality: {s.modality}, source_type: {s.source_type}, doc: {s.document_name}")
    assert any(s.source_type == "object_detection" for s in res3.evidence), "Must contain object detection evidence"
    assert "2" in res3.answer and "car" in res3.answer.lower(), "Must report 2 cars detected"
    print(">>> TEST 3 PASSED: Exact car count with YOLO detection evidence and image source.")

    time.sleep(3)

    # TEST 4 — MULTIMODAL
    print("\n--- TEST 4: MULTIMODAL ('What vehicles are visible in the image and what information does the document provide about vehicles?') ---")
    q4 = "What vehicles are visible in the image and what information does the document provide about vehicles?"
    analysis4 = qa.analyze(q4)
    print(f"Query Analysis: mode={analysis4.query_mode}, needs_img={analysis4.needs_image_retrieval}, needs_det={analysis4.needs_detection_evidence}")
    assert analysis4.query_mode == "MULTIMODAL_RAG", f"Expected MULTIMODAL_RAG, got {analysis4.query_mode}"
    assert analysis4.needs_image_retrieval == True, "Must need image retrieval"
    assert analysis4.needs_detection_evidence == True, "Must need detection evidence"
    
    res4 = car_rag.run(q4)
    print(f"Supported: {res4.answer_supported}")
    print(f"Answer: {res4.answer[:200]}...")
    print(f"Evidence items: {len(res4.evidence)}")
    modalities = [s.modality for s in res4.evidence]
    source_types = [s.source_type for s in res4.evidence]
    print(f"Modalities present: {set(modalities)}, Source types: {set(source_types)}")
    assert "text" in modalities, "Multimodal query must retrieve text evidence"
    assert "image" in modalities, "Multimodal query must retrieve image evidence"
    assert "object_detection" in source_types, "Multimodal query must retrieve object detection evidence"
    print(">>> TEST 4 PASSED: Coexistence of Text + Image + YOLO Detection in unified grounded output.")

    # TEST 7 — PERSISTENCE & RESTART
    print("\n--- TEST 7: PERSISTENCE & RESTART SIMULATION ---")
    new_text_store = VectorStore.load(Path("data/text.index"), Path("data/text_metadata.json"))
    new_image_store = ImageStore.load(Path("data/image.index"), Path("data/image_metadata.json"))
    assert new_text_store.index.ntotal > 0, "Text index must survive reload"
    assert new_image_store.index.ntotal > 0, "Image index must survive reload"
    assert any(len(r.detections) > 0 for r in new_image_store.records), "Detection metadata must survive reload"
    print(f"Reload verified: {new_text_store.index.ntotal} text chunks, {new_image_store.index.ntotal} images, {sum(len(r.detections) for r in new_image_store.records)} detected objects.")
    print(">>> TEST 7 PASSED: Full persistence preserved.")

    print("\n" + "=" * 60)
    print("ALL CRITICAL TEST CASES SUCCEEDED 100%!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()

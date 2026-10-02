"""Trace the exact CAR-RAG live query execution path for object detection questions."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.image_store import ImageStore, ImageRecord
from src.image_embeddings import ImageEmbeddingService
from src.object_detection import ObjectDetector
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.evidence_checker import EvidenceChecker
from src.generator import Generator
from src.answer_checker import AnswerChecker
from src.car_rag import CARRAG


def main():
    print("=== DIAGNOSING LIVE OBJECT DETECTION QUERY ===")
    question = "How many cars are detected?"
    
    # 1. Inspect existing data/image_metadata.json
    meta_path = ROOT / "data" / "image_metadata.json"
    index_path = ROOT / "data" / "image.index"
    
    print(f"\n[1] Checking {meta_path}...")
    if meta_path.is_file():
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f" -> Version: {raw_meta.get('version')}")
        records = raw_meta.get("records", [])
        print(f" -> Number of records: {len(records)}")
        for idx, rec in enumerate(records):
            print(f"   Record {idx}: {rec.get('document_name')}")
            print(f"   Has detections field: {'detections' in rec}")
            print(f"   Detections count: {len(rec.get('detections', []))}")
            print(f"   Detection counts: {rec.get('detection_counts')}")
            print(f"   Detection context length: {len(rec.get('detection_context', ''))}")
    else:
        print(" -> image_metadata.json NOT FOUND!")

    # 2. Test YOLO26n on data/images/1_Busy Hyderabad Street Traffic.png if it exists
    img_file = ROOT / "data" / "images" / "1_Busy Hyderabad Street Traffic.png"
    if not img_file.is_file():
        # Check if any image exists in data/images
        img_dir = ROOT / "data" / "images"
        if img_dir.is_dir():
            files = list(img_dir.iterdir())
            print(f" -> Files in data/images: {files}")
            if files:
                img_file = files[0]

    print(f"\n[2] Testing YOLO26n on {img_file}...")
    detector = ObjectDetector(model_name="yolo26n.pt")
    if img_file.is_file():
        det_res = detector.detect(img_file)
        print(f" -> YOLO detections count: {len(det_res.detections)}")
        print(f" -> YOLO counts: {det_res.counts}")
        print(f" -> YOLO summary: {det_res.summary}")
        print(f" -> YOLO RAG context:\n{det_res.rag_context}")
    else:
        print(" -> Image file not found on disk!")
        return

    # 3. Query Analysis
    print(f"\n[3] Query Analysis for '{question}'...")
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(question)
    print(f" -> Query Type: {analysis.query_type}")
    print(f" -> Needs More Retrieval: {analysis.needs_more_retrieval}")
    print(f" -> Needs Image Retrieval: {analysis.needs_image_retrieval}")

    # 4. Load Image Store
    print(f"\n[4] Loading ImageStore from disk...")
    image_store = ImageStore.load(index_path, meta_path)
    print(f" -> Loaded {len(image_store.records)} records")

    # 5. Multimodal Retriever
    print(f"\n[5] MultimodalRetriever.retrieve()...")
    img_embedder = ImageEmbeddingService()
    retriever = MultimodalRetriever(
        text_store=None,
        image_store=image_store,
        text_embeddings=None,
        image_embeddings=img_embedder,
    )
    
    retrieved = retriever.retrieve(
        question=question,
        top_k=5,
        mode="image",
        prefer_images=True,
    )
    print(f" -> Retrieved items count: {len(retrieved)}")
    for idx, item in enumerate(retrieved, 1):
        print(f"   Item {idx}: Modality={item.modality}, SourceType={item.source_type}, Doc={item.document_name}")
        print(f"   Evidence Block:\n{item.evidence_block(idx)}\n")

    # 6. Evidence Checker
    print(f"\n[6] EvidenceChecker.check()...")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print(" -> GROQ_API_KEY missing! Skipping LLM stages.")
        return

    ev_checker = EvidenceChecker(api_key=api_key)
    ev_check = ev_checker.check(question, retrieved)
    print(f" -> Sufficient: {ev_check.sufficient}")
    print(f" -> Score: {ev_check.score}")
    print(f" -> Reason: {ev_check.reason}")

    # 7. Generator
    print(f"\n[7] Generator.answer()...")
    generator = Generator(api_key=api_key)
    gen_answer = generator.answer(question, retrieved)
    print(f" -> Generated Answer:\n{gen_answer}")

    # 8. Answer Checker
    print(f"\n[8] AnswerChecker.check()...")
    ans_checker = AnswerChecker(api_key=api_key)
    ans_check = ans_checker.check(question=question, answer=gen_answer, results=retrieved)
    print(f" -> Supported: {ans_check.supported}")
    print(f" -> Score: {ans_check.score}")
    print(f" -> Reason: {ans_check.reason}")

    # 9. Full CARRAG.run()
    print(f"\n[9] Full CARRAG.run()...")
    car_rag = CARRAG(
        analyzer=analyzer,
        retriever=retriever,
        evidence_checker=ev_checker,
        generator=generator,
        answer_checker=ans_checker,
    )
    result = car_rag.run(
        question=question,
        initial_evidence=retrieved,
        query_analysis=analysis,
    )
    print(f" -> CARRAG Result Answer: {result.answer}")
    print(f" -> Evidence Sufficient: {result.evidence_sufficient}")
    print(f" -> Grounding Verified: {result.grounding_verified}")
    test_queries = [
        ("A", "How many cars are detected?"),
        ("B", "How many people are detected?"),
        ("C", "What objects are detected?"),
        ("D", "Which vehicles are present?"),
        ("E", "What is present in the image?"),
    ]

    print("\n=== RUNNING BATCH QUERIES A-E ===")
    for tag, q in test_queries:
        print(f"\n--- QUERY {tag}: '{q}' ---")
        q_analysis = analyzer.analyze(q)
        q_retrieved = retriever.retrieve(question=q, top_k=5, mode="image", prefer_images=True)
        q_result = car_rag.run(question=q, initial_evidence=q_retrieved, query_analysis=q_analysis)
        
        has_det = any(e.source_type == "object_detection" for e in q_result.evidence)
        print(f"[{tag}] Answer: {q_result.answer}")
        print(f"[{tag}] Evidence Sufficient: {q_result.evidence_sufficient}")
        print(f"[{tag}] Grounding Verified: {q_result.grounding_verified}")
        print(f"[{tag}] Object Detection Evidence Included: {has_det}")
        print(f"[{tag}] Evidence Reason: {q_result.evidence_reason}")
        print(f"[{tag}] Answer Reason: {q_result.answer_reason}")


if __name__ == "__main__":
    main()

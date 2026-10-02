"""Live test of full CAR-RAG pipeline across all 4 query types."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.vector_store import VectorStore
from src.image_store import ImageStore, ImageRecord
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.object_detection import ObjectDetector
from src.chunker import TextChunk
from src.car_rag import CARRAG
from src.generator import Generator
from src.evidence_checker import EvidenceChecker
from src.answer_checker import AnswerChecker


def main():
    print("=== INITIALIZING STORES AND SERVICES ===", flush=True)
    embed_svc = EmbeddingService()
    img_embed_svc = ImageEmbeddingService()
    detector = ObjectDetector(model_name="yolo26n.pt")

    # 1. Text Chunks for Car Safety Manual
    doc_text = (
        "ABS (Anti-lock Braking System) is an automobile safety system that allows the wheels on a motor vehicle "
        "to maintain tractive contact with the road surface according to driver inputs while braking, preventing the wheels "
        "from locking up (ceasing rotation) and avoiding uncontrolled skidding. "
        "Electronic Stability Control (ESC) compares the driver's intended direction to the vehicle's actual direction and applies "
        "brakes to individual wheels to maintain vehicle control. "
        "Vehicles operating on municipal roads must comply with standard vehicle safety codes and maintenance guidelines."
    )
    chunks = [
        TextChunk(
            document_name="Car_Safety_Manual.pdf",
            chunk_index=0,
            text=doc_text,
            start_char=0,
            end_char=len(doc_text),
        )
    ]
    text_embs = embed_svc.embed_texts([c.text for c in chunks])
    text_store = VectorStore.from_embeddings(chunks, text_embs)
    text_store.save(Path("data/text.index"), Path("data/text_metadata.json"))
    print("Text store built and saved.", flush=True)

    # 2. Image Record for Busy Hyderabad Street Traffic.png
    img_path = Path("data/images/1_Busy Hyderabad Street Traffic.png")
    det_res = detector.detect(img_path)
    print(f"YOLO detections: {det_res.summary}", flush=True)

    img_record = ImageRecord(
        image_id="hyderabad_street_1",
        modality="image",
        document_name="Busy Hyderabad Street Traffic.png",
        source_filename="Busy Hyderabad Street Traffic.png",
        image_path=str(img_path),
        page_number=None,
        caption="A bustling street scene in Hyderabad with cars, a bus, motorcycles, and pedestrians in traffic.",
        source_type="standalone_image",
        width=1280,
        height=720,
        detections=[d.to_dict() for d in det_res.detections],
        detection_summary=det_res.summary,
        detection_counts=det_res.counts,
        detection_context=det_res.rag_context,
    )
    img_embs = img_embed_svc.embed_images([img_path])
    image_store = ImageStore.from_embeddings([img_record], img_embs)
    image_store.save(Path("data/image.index"), Path("data/image_metadata.json"))
    print("Image store built and saved.", flush=True)

    # Save active dataset
    import json
    dataset = {
        "files": [
            {"name": "Car_Safety_Manual.pdf", "type": "PDF", "modality": "text"},
            {"name": "Busy Hyderabad Street Traffic.png", "type": "PNG", "modality": "image"},
        ],
        "dataset_name": "Car Safety & Street Traffic",
        "types": ["PDF", "PNG"],
    }
    Path("data/active_dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    qa = QueryAnalyzer()
    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=embed_svc,
        image_embeddings=img_embed_svc,
    )

    api_key = os.getenv("GROQ_API_KEY")
    generator = Generator(api_key=api_key) if api_key else None
    evidence_checker = EvidenceChecker(api_key=api_key) if api_key else None
    answer_checker = AnswerChecker(api_key=api_key) if api_key else None

    rag = CARRAG(
        analyzer=qa,
        retriever=retriever,
        answer_checker=answer_checker,
        evidence_checker=evidence_checker,
        generator=generator,
    )

    # Query 1: Text Query
    print("\n" + "=" * 60, flush=True)
    print("TEST 1: TEXT QUERY — 'What is ABS?'", flush=True)
    print("=" * 60, flush=True)
    an1 = qa.analyze("What is ABS?")
    print(f"Analysis: {an1}", flush=True)
    res1 = rag.run("What is ABS?", query_analysis=an1, retrieval_mode="both")
    print(f"ANSWER:\n{res1.answer}", flush=True)
    print(f"Grounding: {res1.grounding_verified}, Reason: {res1.evidence_reason}", flush=True)
    has_det = any(item.source_type == "object_detection" for item in res1.evidence)
    print(f"Object detection in evidence? {has_det} (Must be False)", flush=True)
    assert not has_det, "Object detection leaked into text query!"
    assert "anti-lock" in res1.answer.lower() or "braking" in res1.answer.lower(), "Expected ABS definition from text!"

    # Query 2: Image Query
    print("\n" + "=" * 60, flush=True)
    print("TEST 2: IMAGE QUERY — 'Describe the street scene.'", flush=True)
    print("=" * 60, flush=True)
    an2 = qa.analyze("Describe the street scene.")
    print(f"Analysis: {an2}", flush=True)
    res2 = rag.run("Describe the street scene.", query_analysis=an2, retrieval_mode="both")
    print(f"ANSWER:\n{res2.answer}", flush=True)
    print(f"Grounding: {res2.grounding_verified}, Reason: {res2.evidence_reason}", flush=True)
    has_det2 = any(item.source_type == "object_detection" for item in res2.evidence)
    print(f"Object detection in evidence? {has_det2} (Must be False)", flush=True)
    assert not has_det2, "Object detection leaked into image description query!"

    # Query 3: Object Detection Count Query
    print("\n" + "=" * 60, flush=True)
    print("TEST 3: OBJECT DETECTION QUERY — 'How many cars are detected?'", flush=True)
    print("=" * 60, flush=True)
    an3 = qa.analyze("How many cars are detected?")
    print(f"Analysis: {an3}", flush=True)
    res3 = rag.run("How many cars are detected?", query_analysis=an3, retrieval_mode="both")
    print(f"ANSWER:\n{res3.answer}", flush=True)
    print(f"Grounding: {res3.grounding_verified}, Reason: {res3.evidence_reason}", flush=True)
    has_det3 = any(item.source_type == "object_detection" for item in res3.evidence)
    print(f"Object detection in evidence? {has_det3} (Must be True)", flush=True)
    assert has_det3, "Object detection missing for count query!"
    assert "2 cars" in res3.answer or "2 car" in res3.answer, f"Expected 2 cars in answer, got: {res3.answer}"

    # Query 4: Multimodal Query
    print("\n" + "=" * 60, flush=True)
    print("TEST 4: MULTIMODAL QUERY — 'What vehicles are visible in the image and what information does the document provide about vehicles?'", flush=True)
    print("=" * 60, flush=True)
    an4 = qa.analyze("What vehicles are visible in the image and what information does the document provide about vehicles?")
    print(f"Analysis: {an4}", flush=True)
    res4 = rag.run("What vehicles are visible in the image and what information does the document provide about vehicles?", query_analysis=an4, retrieval_mode="both")
    print(f"ANSWER:\n{res4.answer}", flush=True)
    print(f"Grounding: {res4.grounding_verified}, Reason: {res4.evidence_reason}", flush=True)
    modalities = {item.modality for item in res4.evidence}
    source_types = {item.source_type for item in res4.evidence}
    print(f"Modalities: {modalities}, Source types: {source_types}", flush=True)
    assert "text" in modalities and "image" in modalities and "object_detection" in source_types, "Multimodal evidence missing one of the modalities!"

    print("\n" + "=" * 60, flush=True)
    print("ALL 4 QUERY FLOWS PASSED VERIFICATION WITH FLYING COLORS!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

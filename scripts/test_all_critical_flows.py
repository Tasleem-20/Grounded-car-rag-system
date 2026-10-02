"""Comprehensive test verifying all CAR-RAG and Object Detection coexistence flows."""

import io
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.vector_store import VectorStore
from src.image_store import ImageStore, ImageRecord
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.chunker import TextChunk
from src.car_rag import CARRAG
from src.deterministic_detection import is_direct_object_detection_query, answer_detection_query


def run_tests():
    print("=" * 60)
    print("1. QUERY ANALYZER ROUTING TESTS")
    print("=" * 60)
    qa = QueryAnalyzer()
    
    test_queries = [
        ("What is the main topic of this document?", "TEXT_RAG", False, False),
        ("What does the document say about ABS?", "TEXT_RAG", False, False),
        ("What is ABS?", "TEXT_RAG", False, False),
        ("Describe the image.", "IMAGE_RAG", True, False),
        ("Describe the street scene.", "IMAGE_RAG", True, False),
        ("How many cars are detected?", "OBJECT_DETECTION", True, True),
        ("How many cars are there?", "OBJECT_DETECTION", True, True),
        ("What vehicles are visible in the image and what information does the document provide about vehicles?", "MULTIMODAL_RAG", True, True),
        ("What vehicles are shown in the image and what does the document say about them?", "MULTIMODAL_RAG", True, True),
    ]

    for q, expected_mode, expected_img, expected_det in test_queries:
        res = qa.analyze(q)
        print(f"Query: '{q}'")
        print(f"  -> mode: {res.query_mode} (expected: {expected_mode})")
        print(f"  -> needs_image: {res.needs_image_retrieval} (expected: {expected_img})")
        print(f"  -> needs_det: {res.needs_detection_evidence} (expected: {expected_det})")
        assert res.query_mode == expected_mode, f"Mode mismatch for '{q}': got {res.query_mode}, expected {expected_mode}"
        assert res.needs_image_retrieval == expected_img, f"needs_image mismatch for '{q}'"
        assert res.needs_detection_evidence == expected_det, f"needs_det mismatch for '{q}'"
        print("  [PASS]")

    print("\n" + "=" * 60)
    print("2. DETERMINISTIC DETECTION EXCLUSION TESTS")
    print("=" * 60)
    direct_checks = [
        ("How many cars are detected?", True),
        ("How many cars are there?", True),
        ("Are there any buses?", True),
        ("What objects are detected?", True),
        ("What is ABS?", False),
        ("What does the document say about ABS?", False),
        ("Describe the image.", False),
        ("Describe the street scene.", False),
        ("What vehicles are visible in the image and what information does the document provide about vehicles?", False),
    ]
    for q, expected_direct in direct_checks:
        direct = is_direct_object_detection_query(q)
        print(f"Query: '{q}' -> is_direct: {direct} (expected: {expected_direct})")
        assert direct == expected_direct, f"Direct mismatch for '{q}'"
        print("  [PASS]")

    print("\n" + "=" * 60)
    print("3. MULTIMODAL RETRIEVER MODALITY ISOLATION TESTS")
    print("=" * 60)
    embed_svc = EmbeddingService()
    img_embed_svc = ImageEmbeddingService()

    # Create synthetic text and image store
    text_chunks = [
        TextChunk(
            document_name="car_manual.pdf",
            chunk_index=0,
            text="ABS is an anti-lock braking system that prevents the wheels from locking up during braking, thereby maintaining tractive contact with the road surface.",
            start_char=0,
            end_char=150,
        ),
        TextChunk(
            document_name="car_manual.pdf",
            chunk_index=1,
            text="Vehicles equipped with electronic stability control provide superior handling under adverse weather conditions.",
            start_char=151,
            end_char=260,
        ),
    ]
    text_embs = embed_svc.embed_texts([c.text for c in text_chunks])
    text_store = VectorStore.from_embeddings(text_chunks, text_embs)

    # Synthetic image record
    image_record = ImageRecord(
        image_id="street_1",
        modality="image",
        document_name="Busy Hyderabad Street Traffic.png",
        source_filename="Busy Hyderabad Street Traffic.png",
        image_path="data/images/1_Busy Hyderabad Street Traffic.png",
        page_number=None,
        caption="A bustling street in Hyderabad with cars, auto rickshaws, and buses in traffic.",
        source_type="standalone_image",
        width=1280,
        height=720,
        detections=[
            {"label": "car", "confidence": 0.92, "x1": 100, "y1": 200, "x2": 300, "y2": 400},
            {"label": "car", "confidence": 0.88, "x1": 350, "y1": 220, "x2": 500, "y2": 420},
            {"label": "bus", "confidence": 0.95, "x1": 550, "y1": 150, "x2": 800, "y2": 450},
        ],
        detection_summary="Detected: 2 cars, 1 bus.",
        detection_counts={"car": 2, "bus": 1},
        detection_context="OBJECT DETECTION EVIDENCE\nDetected: 2 cars, 1 bus.\n1. car | confidence=0.920\n2. car | confidence=0.880\n3. bus | confidence=0.950",
    )
    # We create a dummy 1x1 image in memory if path not present for embedding
    dummy_img_path = Path("data/images/test_dummy.png")
    dummy_img_path.parent.mkdir(parents=True, exist_ok=True)
    if not dummy_img_path.exists():
        Image.new("RGB", (100, 100), color="blue").save(dummy_img_path)
    
    img_embs = img_embed_svc.embed_images([dummy_img_path])
    image_store = ImageStore.from_embeddings([image_record], img_embs)

    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=embed_svc,
        image_embeddings=img_embed_svc,
    )

    # Test A: Text Query (Mode TEXT_RAG, include_detection=False)
    q_text = "What is ABS?"
    an_text = qa.analyze(q_text)
    ev_text = retriever.retrieve(
        q_text,
        top_k=3,
        mode="both",
        prefer_images=an_text.needs_image_retrieval,
        include_detection=an_text.needs_detection_evidence,
    )
    print(f"\nQuery: '{q_text}'")
    for item in ev_text:
        print(f"  Item: modality={item.modality}, source_type={item.source_type}, doc={item.document_name}")
    assert any(item.modality == "text" and "anti-lock" in item.text for item in ev_text), "Text chunk missing!"
    assert not any(item.source_type == "object_detection" for item in ev_text), "Object detection leaked into pure text query!"
    print("  [PASS - Text query returned text chunks with ZERO object detection leakage]")

    # Test B: Image Query (Mode IMAGE_RAG, include_detection=False)
    q_img = "Describe the image."
    an_img = qa.analyze(q_img)
    ev_img = retriever.retrieve(
        q_img,
        top_k=3,
        mode="both",
        prefer_images=an_img.needs_image_retrieval,
        include_detection=an_img.needs_detection_evidence,
    )
    print(f"\nQuery: '{q_img}'")
    for item in ev_img:
        print(f"  Item: modality={item.modality}, source_type={item.source_type}, doc={item.document_name}")
    assert any(item.modality == "image" for item in ev_img), "Image evidence missing!"
    assert not any(item.source_type == "object_detection" for item in ev_img), "Object detection leaked into pure image description query!"
    print("  [PASS - Image query returned image evidence with ZERO object detection leakage]")

    # Test C: Object Detection Query (Mode OBJECT_DETECTION, include_detection=True)
    q_det = "How many cars are detected?"
    an_det = qa.analyze(q_det)
    ev_det = retriever.retrieve(
        q_det,
        top_k=3,
        mode="both",
        prefer_images=an_det.needs_image_retrieval,
        include_detection=an_det.needs_detection_evidence,
    )
    print(f"\nQuery: '{q_det}'")
    for item in ev_det:
        print(f"  Item: modality={item.modality}, source_type={item.source_type}, doc={item.document_name}")
    assert any(item.source_type == "object_detection" for item in ev_det), "Object detection evidence missing for count query!"
    ans, valid = answer_detection_query(q_det, ev_det)
    print(f"  Detection Answer: '{ans}' (valid={valid})")
    assert valid and "2 cars" in ans, f"Unexpected count answer: {ans}"
    print("  [PASS - Object detection query returned exact count from evidence]")

    # Test D: Multimodal Query (Mode MULTIMODAL_RAG, include_detection=True)
    q_multi = "What vehicles are visible in the image and what information does the document provide about vehicles?"
    an_multi = qa.analyze(q_multi)
    ev_multi = retriever.retrieve(
        q_multi,
        top_k=3,
        mode="both",
        prefer_images=an_multi.needs_image_retrieval,
        include_detection=an_multi.needs_detection_evidence,
    )
    print(f"\nQuery: '{q_multi}'")
    modalities = {item.modality for item in ev_multi}
    source_types = {item.source_type for item in ev_multi}
    print(f"  Modalities present: {modalities}")
    print(f"  Source types present: {source_types}")
    assert "text" in modalities, "Text modality missing from multimodal retrieval!"
    assert "image" in modalities, "Image modality missing from multimodal retrieval!"
    assert "object_detection" in source_types, "Object detection evidence missing from multimodal retrieval!"
    print("  [PASS - Multimodal query returned text + image + object detection coexisting]")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

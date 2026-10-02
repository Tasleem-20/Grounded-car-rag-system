"""Test deterministic object detection routing and answers without Groq."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_store import ImageStore
from src.query_analyzer import QueryAnalyzer
from src.retriever import MultimodalRetriever
from src.image_embeddings import ImageEmbeddingService
from src.car_rag import CARRAG

def main():
    meta_path = ROOT / "data" / "image_metadata.json"
    index_path = ROOT / "data" / "image.index"
    image_store = ImageStore.load(index_path, meta_path)

    analyzer = QueryAnalyzer()
    img_embedder = ImageEmbeddingService()
    retriever = MultimodalRetriever(
        text_store=None,
        image_store=image_store,
        text_embeddings=None,
        image_embeddings=img_embedder,
    )

    # Initialise CARRAG with NO groq_service / generators to strictly guarantee NO Groq is called
    car_rag = CARRAG(
        analyzer=analyzer,
        retriever=retriever,
        groq_service=None,
        generator=None,
        evidence_checker=None,
        answer_checker=None,
    )

    test_queries = [
        ("0", "how many cars here"),
        ("1", "how many cars are there"),
        ("2", "how many buses are there"),
        ("3", "what objects are detected?"),
        ("4", "is there a car?"),
        ("5", "is there a dog?"),
        ("6", "how many cars and buses are there?"),
        ("7", "which vehicles are present?"),
        ("8", "list the detected objects"),
    ]

    print("=== TESTING DETERMINISTIC OBJECT DETECTION (ZERO GROQ CALLS) ===")
    for tag, q in test_queries:
        analysis = analyzer.analyze(q)
        print(f"\n[Query {tag}]: '{q}'")
        print(f"  Query Mode: {analysis.query_mode}")
        
        retrieved = retriever.retrieve(q, mode="image", prefer_images=True)
        res = car_rag.run(q, initial_evidence=retrieved, query_analysis=analysis)
        
        print(f"  Grounded Answer: {res.answer}")
        print(f"  Evidence Sufficient: {res.evidence_sufficient}")
        print(f"  Grounding Verified: {res.grounding_verified}")
        print(f"  Evidence Reason: {res.evidence_reason}")

    # Test routing on semantic / gender questions to ensure they are NOT routed to deterministic YOLO
    semantic_queries = [
        "describe the image",
        "is there a girl or boy in the picture?",
        "what is the person doing?",
        "explain the vehicle safety policy in the documents",
    ]
    print("\n=== TESTING ROUTING FOR SEMANTIC / GENDER / TEXT QUERIES ===")
    for q in semantic_queries:
        analysis = analyzer.analyze(q)
        print(f"Query: '{q}' -> Query Mode: {analysis.query_mode} (Needs Image: {analysis.needs_image_retrieval})")

if __name__ == "__main__":
    main()

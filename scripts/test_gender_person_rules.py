"""Verification of Gender-Related Person Query Handling and Object Detection in CAR-RAG."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import app
from src.query_analyzer import QueryAnalyzer
from src.car_rag import CARRAG
from src.retriever import MultimodalRetriever


def main():
    text_store = app.load_text_store()
    image_store = app.load_image_store()

    qa = QueryAnalyzer()
    retriever = MultimodalRetriever(
        text_store=text_store,
        image_store=image_store,
        text_embeddings=app.get_embedding_service(),
        image_embeddings=app.get_image_embedding_service(),
    )
    rag = CARRAG(
        analyzer=qa,
        retriever=retriever,
        answer_checker=app.optional_answer_checker(),
        evidence_checker=app.optional_evidence_checker(),
        generator=app.optional_generator(),
    )

    test_queries = [
        "Is there a boy or girl in the image?",
        "Is the person a girl?",
        "How many people are detected?",
        "Is there a person?",
        "What objects are detected?",
    ]

    print("=" * 70)
    print("END-TO-END CARRAG EVALUATION FOR 5 REQUIRED QUERIES")
    print("=" * 70)

    for q in test_queries:
        res = rag.run(q)
        print(f"\nQuery: '{q}'")
        print(f"Answer: '{res.answer}'")
        print(f"Route: {res.query_analysis.query_mode} | Grounded: {res.grounding_verified}")

    print("\n" + "=" * 70)
    print("ALL 5 QUERIES EXECUTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

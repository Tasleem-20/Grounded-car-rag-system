"""End-to-end verification of Image and Document Question Answering."""

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

class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.type = "application/pdf" if name.endswith(".pdf") else "image/jpeg"

    def getvalue(self) -> bytes:
        return self._data


def main():
    qubit_path = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates\Qubit.jpg")
    pdf_path = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates\Soft Skills.pdf")

    print(f"Indexing {[qubit_path.name, pdf_path.name]}...")
    uploads = [
        MockUploadedFile(qubit_path.name, qubit_path.read_bytes()),
        MockUploadedFile(pdf_path.name, pdf_path.read_bytes()),
    ]
    app.process_uploads(uploads)
    print("Indexing Complete!")

    text_store = app.load_text_store()
    image_store = app.load_image_store()

    assert text_store is not None, "Text store must be present"
    assert image_store is not None, "Image store must be present"

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

    print("\n" + "="*60)
    print("TESTING IMAGE QUESTION ANSWERING (Qubit.jpg)")
    print("="*60)

    test_queries = [
        ("What is the title of this certificate?", "QUANTUM FUNDAMENTALS 2025-2026"),
        ("What is the name of the recipient?", "Shaik Tasleem"),
        ("What is the Certificate ID?", "3A228880"),
        ("When was the certificate issued?", "07-02-2026"),
        ("What program is mentioned on the certificate?", "Quantum Fundamentals Program"),
    ]

    for q, expected in test_queries:
        res = rag.run(q, source_filter="Qubit.jpg")
        print(f"\nQ: '{q}'")
        print(f"A: {res.answer}")
        print(f"Route: {res.query_analysis.query_mode}, Grounded: {res.grounding_verified}")
        print(f"Evidence count: {len(res.evidence)}")
        assert any(term.lower() in res.answer.lower() for term in expected.split() if len(term) > 3), f"Failed for {q}: got {res.answer}, expected {expected}"
        assert res.grounding_verified is True

    print("\n" + "="*60)
    print("TESTING PDF TEXT QUESTION ANSWERING (Soft Skills.pdf)")
    print("="*60)

    pdf_queries = [
        "Who is the recipient of this credential?",
        "What is the domain of this internship?",
    ]
    for q in pdf_queries:
        res = rag.run(q, source_filter="Soft Skills.pdf")
        print(f"\nQ: '{q}'")
        print(f"A: {res.answer}")
        print(f"Route: {res.query_analysis.query_mode}, Grounded: {res.grounding_verified}")
        assert len(res.evidence) > 0
        assert res.grounding_verified is True

    print("\n" + "="*60)
    print("ALL IMAGE AND PDF TESTS PASSED PERFECTLY!")
    print("="*60)


if __name__ == "__main__":
    main()

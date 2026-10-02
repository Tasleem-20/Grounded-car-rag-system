"""Test calling process_uploads with Soft Skills.pdf."""

import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.type = "application/pdf" if name.endswith(".pdf") else "image/jpeg"

    def getvalue(self) -> bytes:
        return self._data

def main():
    import app
    import streamlit as st
    if not hasattr(st, "session_state"):
        st.session_state = {}

    st.session_state.settings = {"top_k": 5, "retrieval_mode": "both", "show_debug": False}
    st.session_state.active_dataset = {"files": [], "dataset_name": "No active dataset"}
    st.session_state.selected_source = None
    st.session_state.last_result = None

    pdf_files = list(Path("data/documents").glob("*.pdf"))
    img_files = list(Path("data/images").glob("*.png")) + list(Path("data/images").glob("*.jpg"))

    pdf_path = pdf_files[0] if pdf_files else Path(r"C:\Users\Arif\Documents\tasleeem\Certificates\Soft Skills.pdf")
    img_path = next((p for p in img_files if "Hyderabad" in p.name), img_files[0] if img_files else None)

    uploads = [MockUploadedFile(pdf_path.name, pdf_path.read_bytes())]
    if img_path:
        uploads.append(MockUploadedFile(img_path.name.split("_", 1)[-1], img_path.read_bytes()))

    print(f"Indexing {[u.name for u in uploads]}...")
    app.process_uploads(uploads)
    print("Indexing Complete!")

    from src.query_analyzer import QueryAnalyzer
    from src.car_rag import CARRAG
    from src.retriever import MultimodalRetriever

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

    print("\n" + "="*60)
    print("LIVE FACTUAL, ROUTING, DETECTION & RESILIENCE TESTS (A - J)")
    print("="*60)

    # TEST A — TEXT RETRIEVAL (Domain)
    print("\n--- TEST A: DOMAIN ---")
    resA = rag.run("What is the domain of this internship?")
    print(f"Answer: {resA.answer}")
    print(f"Mode: {resA.query_analysis.query_mode}, Grounded: {resA.grounding_verified}")
    assert resA.query_analysis.query_mode == "TEXT_RAG", "Must route to TEXT_RAG"
    assert "java" in resA.answer.lower() or "full stack" in resA.answer.lower(), "Must identify Java Full Stack"
    assert not any(x.source_type == "object_detection" for x in resA.evidence), "Object detection must be hidden"

    # TEST B — TEXT RETRIEVAL (Duration)
    print("\n--- TEST B: DURATION ---")
    resB = rag.run("How long was the internship?")
    print(f"Answer: {resB.answer}")
    print(f"Mode: {resB.query_analysis.query_mode}, Grounded: {resB.grounding_verified}")
    assert resB.query_analysis.query_mode == "TEXT_RAG", "Must route to TEXT_RAG"
    assert any(w in resB.answer.lower() for w in ["10", "week", "duration"]), "Must identify 10 weeks"

    # TEST C — TEXT RETRIEVAL (Technologies)
    print("\n--- TEST C: TECHNOLOGIES & TOPICS ---")
    resC = rag.run("What technologies and topics are mentioned in the certificate?")
    print(f"Answer: {resC.answer}")
    print(f"Mode: {resC.query_analysis.query_mode}, Grounded: {resC.grounding_verified}")
    assert resC.query_analysis.query_mode == "TEXT_RAG", "Must route to TEXT_RAG"
    assert not any(x.source_type == "object_detection" for x in resC.evidence), "Object detection must be hidden"

    # TEST D — TEXT RETRIEVAL (PDF Summary)
    print("\n--- TEST D: COMPLETE SUMMARY ---")
    resD = rag.run("Give me a complete summary of this certificate.")
    print(f"Answer: {resD.answer[:120]}...")
    print(f"Mode: {resD.query_analysis.query_mode}, Grounded: {resD.grounding_verified}")
    assert resD.query_analysis.query_mode == "TEXT_RAG", "Must route to TEXT_RAG"
    assert not any(x.source_type == "object_detection" for x in resD.evidence), "Object detection must be hidden"

    # TEST E — IMAGE RETRIEVAL (Visual Design)
    print("\n--- TEST E: VISUAL DESIGN OF CERTIFICATE ---")
    resE = rag.run("Describe the visual design of this certificate.")
    print(f"Answer: {resE.answer[:120]}...")
    print(f"Mode: {resE.query_analysis.query_mode}, Grounded: {resE.grounding_verified}")
    assert resE.query_analysis.query_mode == "IMAGE_RAG", "Must route to IMAGE_RAG"
    assert not any(x.source_type == "object_detection" for x in resE.evidence), "Object detection must be hidden"

    # TEST F — IMAGE RETRIEVAL (Street Scene)
    print("\n--- TEST F: STREET SCENE DESCRIPTION ---")
    resF = rag.run("Describe the street scene.")
    print(f"Answer: {resF.answer[:120]}...")
    print(f"Mode: {resF.query_analysis.query_mode}, Grounded: {resF.grounding_verified}")
    assert resF.query_analysis.query_mode == "IMAGE_RAG", "Must route to IMAGE_RAG"
    assert not any(x.source_type == "object_detection" for x in resF.evidence), "Object detection must be hidden"

    # TEST G — OBJECT DETECTION (Cars count)
    print("\n--- TEST G: CARS COUNT ---")
    resG = rag.run("How many cars are detected?")
    print(f"Answer: {resG.answer}")
    print(f"Mode: {resG.query_analysis.query_mode}, Grounded: {resG.grounding_verified}")
    assert resG.query_analysis.query_mode == "OBJECT_DETECTION", "Must route to OBJECT_DETECTION"
    assert "car" in resG.answer.lower() and "2" in resG.answer, "Must report 2 cars detected from YOLO metadata"
    assert any(x.source_type == "object_detection" for x in resG.evidence), "Object detection evidence must be present"

    # TEST H — OBJECT DETECTION (Objects detected)
    print("\n--- TEST H: OBJECTS DETECTED ---")
    resH = rag.run("What objects are detected in the image?")
    print(f"Answer: {resH.answer}")
    print(f"Mode: {resH.query_analysis.query_mode}, Grounded: {resH.grounding_verified}")
    assert resH.query_analysis.query_mode == "OBJECT_DETECTION", "Must route to OBJECT_DETECTION"
    assert any(x.source_type == "object_detection" for x in resH.evidence), "Object detection evidence must be present"

    # TEST I — OBJECT DETECTION (Vehicles detected)
    print("\n--- TEST I: VEHICLES DETECTED ---")
    resI = rag.run("Which vehicles are detected?")
    print(f"Answer: {resI.answer}")
    print(f"Mode: {resI.query_analysis.query_mode}, Grounded: {resI.grounding_verified}")
    assert resI.query_analysis.query_mode == "OBJECT_DETECTION", "Must route to OBJECT_DETECTION"
    assert any(x.source_type == "object_detection" for x in resI.evidence), "Object detection evidence must be present"

    # TEST J — OBJECT DETECTION (Person detection)
    print("\n--- TEST J: PERSON DETECTION ---")
    resJ = rag.run("Is there a person in the image?")
    print(f"Answer: {resJ.answer}")
    print(f"Mode: {resJ.query_analysis.query_mode}, Grounded: {resJ.grounding_verified}")
    assert resJ.query_analysis.query_mode == "OBJECT_DETECTION", "Must route to OBJECT_DETECTION"
    assert any(x.source_type == "object_detection" for x in resJ.evidence), "Object detection evidence must be present"

    # PERSISTENCE TEST
    print("\n--- TEST: PERSISTENCE (RELOAD FROM DISK) ---")
    reloaded_text = app.load_text_store()
    reloaded_image = app.load_image_store()
    assert reloaded_text is not None, "Text store must persist on disk"
    assert reloaded_image is not None, "Image store must persist on disk"
    assert len(reloaded_image.records) > 0, "Image records and YOLO metadata must persist"
    
    reloaded_retriever = MultimodalRetriever(
        text_store=reloaded_text,
        image_store=reloaded_image,
        text_embeddings=app.get_embedding_service(),
        image_embeddings=app.get_image_embedding_service(),
    )
    reloaded_rag = CARRAG(
        analyzer=qa,
        retriever=reloaded_retriever,
        answer_checker=None,
        evidence_checker=None,
        generator=None,
    )
    res_persist_obj = reloaded_rag.run("How many cars are detected?")
    assert "car" in res_persist_obj.answer.lower() and "2" in res_persist_obj.answer, "Must answer from reloaded YOLO metadata"
    
    res_persist_txt = reloaded_rag.run("What is the domain of this internship?")
    assert "java" in res_persist_txt.answer.lower() or "full stack" in res_persist_txt.answer.lower(), "Must answer from reloaded text store"
    print("Persistence test successful!")

    print("\n" + "="*60)
    print("ALL TESTS (A - J) + PERSISTENCE PASSED WITH 100% SUCCESS!")
    print("="*60)

if __name__ == "__main__":
    main()

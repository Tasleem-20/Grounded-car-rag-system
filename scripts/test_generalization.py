"""Test general unseen queries and varied phrasings across Text, Image, Object, and Multimodal modalities."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import app
import streamlit as st
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

    print("="*60)
    print("GENERALIZATION VALIDATION (UNSEEN VARIED PHRASINGS)")
    print("="*60)

    # 1. UNSEEN TEXT QUESTIONS
    print("\n--- 1. UNSEEN TEXT QUESTIONS ---")
    
    q_text_1 = "Who is the recipient of this credential?"
    res1 = rag.run(q_text_1)
    print(f"Q: '{q_text_1}'")
    print(f"A: {res1.answer}")
    print(f"Route: {res1.query_analysis.query_mode}, Grounded: {res1.grounding_verified}")
    assert res1.query_analysis.query_mode == "TEXT_RAG"
    assert "tasleem" in res1.answer.lower() or "shaik" in res1.answer.lower()
    assert not any(x.source_type == "object_detection" for x in res1.evidence)

    q_text_2 = "What web development topics did the learner study?"
    res2 = rag.run(q_text_2)
    print(f"\nQ: '{q_text_2}'")
    print(f"A: {res2.answer}")
    print(f"Route: {res2.query_analysis.query_mode}, Grounded: {res2.grounding_verified}")
    assert res2.query_analysis.query_mode == "TEXT_RAG"
    assert any(w in res2.answer.lower() for w in ["html", "css", "javascript", "bootstrap"])
    assert not any(x.source_type == "object_detection" for x in res2.evidence)

    q_text_3 = "Tell me about the backend frameworks mentioned in the file."
    res3 = rag.run(q_text_3)
    print(f"\nQ: '{q_text_3}'")
    print(f"A: {res3.answer}")
    print(f"Route: {res3.query_analysis.query_mode}, Grounded: {res3.grounding_verified}")
    assert res3.query_analysis.query_mode == "TEXT_RAG"
    assert any(w in res3.answer.lower() for w in ["spring", "hibernate", "java"])
    assert not any(x.source_type == "object_detection" for x in res3.evidence)

    # 2. UNSEEN VISUAL QUESTIONS
    print("\n--- 2. UNSEEN VISUAL QUESTIONS ---")
    
    q_vis_1 = "Can you describe what the scene in Hyderabad looks like?"
    res_v1 = rag.run(q_vis_1)
    print(f"Q: '{q_vis_1}'")
    print(f"A: {res_v1.answer[:120]}...")
    print(f"Route: {res_v1.query_analysis.query_mode}, Grounded: {res_v1.grounding_verified}")
    assert res_v1.query_analysis.query_mode == "IMAGE_RAG"
    assert not any(x.source_type == "object_detection" for x in res_v1.evidence)

    q_vis_2 = "Describe the overall layout and appearance of this certificate."
    res_v2 = rag.run(q_vis_2)
    print(f"\nQ: '{q_vis_2}'")
    print(f"A: {res_v2.answer[:120]}...")
    print(f"Route: {res_v2.query_analysis.query_mode}, Grounded: {res_v2.grounding_verified}")
    assert res_v2.query_analysis.query_mode == "IMAGE_RAG"
    assert not any(x.source_type == "object_detection" for x in res_v2.evidence)

    # 3. UNSEEN OBJECT DETECTION QUESTIONS
    print("\n--- 3. UNSEEN OBJECT DETECTION QUESTIONS ---")

    q_obj_1 = "Are there any pedestrians walking around?"
    res_o1 = rag.run(q_obj_1)
    print(f"Q: '{q_obj_1}'")
    print(f"A: {res_o1.answer}")
    print(f"Route: {res_o1.query_analysis.query_mode}, Grounded: {res_o1.grounding_verified}")
    assert res_o1.query_analysis.query_mode == "OBJECT_DETECTION"
    assert "people" in res_o1.answer.lower() or "person" in res_o1.answer.lower() or "3" in res_o1.answer
    assert any(x.source_type == "object_detection" for x in res_o1.evidence)

    q_obj_2 = "How many backpacks are visible?"
    res_o2 = rag.run(q_obj_2)
    print(f"\nQ: '{q_obj_2}'")
    print(f"A: {res_o2.answer}")
    print(f"Route: {res_o2.query_analysis.query_mode}, Grounded: {res_o2.grounding_verified}")
    assert res_o2.query_analysis.query_mode == "OBJECT_DETECTION"
    assert "2 backpack" in res_o2.answer.lower()
    assert any(x.source_type == "object_detection" for x in res_o2.evidence)

    q_obj_3 = "Are there any airplanes in the image?"
    res_o3 = rag.run(q_obj_3)
    print(f"\nQ: '{q_obj_3}'")
    print(f"A: {res_o3.answer}")
    print(f"Route: {res_o3.query_analysis.query_mode}, Grounded: {res_o3.grounding_verified}")
    assert res_o3.query_analysis.query_mode == "OBJECT_DETECTION"
    assert "no airplane" in res_o3.answer.lower() or "0 airplane" in res_o3.answer.lower()
    assert any(x.source_type == "object_detection" for x in res_o3.evidence)

    q_obj_4 = "Can you list every item detected by the model?"
    res_o4 = rag.run(q_obj_4)
    print(f"\nQ: '{q_obj_4}'")
    print(f"A: {res_o4.answer}")
    print(f"Route: {res_o4.query_analysis.query_mode}, Grounded: {res_o4.grounding_verified}")
    assert res_o4.query_analysis.query_mode == "OBJECT_DETECTION"
    assert any(w in res_o4.answer.lower() for w in ["car", "bus", "people", "backpack"])
    assert any(x.source_type == "object_detection" for x in res_o4.evidence)

    print("\n" + "="*60)
    print("ALL GENERALIZATION TESTS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()

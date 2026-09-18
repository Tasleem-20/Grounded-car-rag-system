"""Test all 4 batch upload configurations requested by user:
A: Soft Skills.pdf alone
B: Soft Skills.pdf + Qubit.jpg
C: Soft Skills.pdf + java fullstack(edu skills).pdf
D: Soft Skills.pdf + Qubit.jpg + java fullstack(edu skills).pdf
"""

import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import app
import streamlit as st

class MockUploadedFile:
    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()
        self.type = "application/pdf" if path.suffix.lower() == ".pdf" else "image/jpeg"

    def getvalue(self) -> bytes:
        return self._data

def run_batch_test(test_name: str, file_paths: list[Path]):
    print(f"\n{'='*70}", flush=True)
    print(f"RUNNING {test_name}: {[p.name for p in file_paths]}", flush=True)
    print(f"{'='*70}", flush=True)

    if not hasattr(st, "session_state"):
        st.session_state = {}

    st.session_state.settings = {"top_k": 5, "retrieval_mode": "both", "show_debug": True}
    st.session_state.active_dataset = {"files": [], "dataset_name": "No active dataset"}
    st.session_state.selected_source = None
    st.session_state.last_result = None

    uploads = [MockUploadedFile(p) for p in file_paths]

    try:
        app.process_uploads(uploads)
        print(f"[{test_name}] Process Uploads: SUCCESS", flush=True)
        active = st.session_state.active_dataset
        active_filenames = [f["name"] for f in active["files"]]
        print(f"[{test_name}] Active Dataset Files ({len(active['files'])}): {active_filenames}", flush=True)

        expected_filenames = [p.name for p in file_paths]
        if set(active_filenames) != set(expected_filenames):
            print(f"[{test_name}] ERROR: Active dataset files mismatch! Expected {expected_filenames}, got {active_filenames}", flush=True)
            return False

        # Check indexes exist and load
        text_store = app.load_text_store()
        image_store = app.load_image_store()
        print(f"[{test_name}] Text Store chunks: {len(text_store.chunks) if text_store else 0}", flush=True)
        print(f"[{test_name}] Image Store records: {len(image_store.records) if image_store else 0}", flush=True)

        # Run test query on Soft Skills
        res_ss = app.run_car_rag(
            question="What is the candidate name and course name on the Soft Skills certificate?",
            selected_source="Soft Skills.pdf",
            search_all=False,
        )
        print(f"[{test_name}] Soft Skills Query Answer: {str(res_ss.answer).encode('ascii', 'replace').decode('ascii')}", flush=True)
        print(f"[{test_name}] Soft Skills Grounding Verified: {res_ss.grounding_verified}", flush=True)

        # If Qubit is in batch, test Qubit query
        if "Qubit.jpg" in expected_filenames:
            res_qb = app.run_car_rag(
                question="What workshop did Tasleem attend at Qubit?",
                selected_source="Qubit.jpg",
                search_all=False,
            )
            print(f"[{test_name}] Qubit Query Answer: {str(res_qb.answer).encode('ascii', 'replace').decode('ascii')}", flush=True)
            print(f"[{test_name}] Qubit Grounding Verified: {res_qb.grounding_verified}", flush=True)

        # If Java Full Stack is in batch, test Java FS query
        if "java fullstack(edu skills).pdf" in expected_filenames:
            res_java = app.run_car_rag(
                question="What internship program did Shaik Tasleem complete for Java Full Stack?",
                selected_source="java fullstack(edu skills).pdf",
                search_all=False,
            )
            print(f"[{test_name}] Java FS Query Answer: {str(res_java.answer).encode('ascii', 'replace').decode('ascii')}", flush=True)
            print(f"[{test_name}] Java FS Grounding Verified: {res_java.grounding_verified}", flush=True)

        return True

    except Exception as e:
        print(f"[{test_name}] FAILED WITH EXCEPTION:", flush=True)
        traceback.print_exc()
        return False

def main():
    cert_dir = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates")
    soft_skills = cert_dir / "Soft Skills.pdf"
    qubit = cert_dir / "Qubit.jpg"
    java_fs = cert_dir / "java fullstack(edu skills).pdf"

    for p in [soft_skills, qubit, java_fs]:
        if not p.exists():
            print(f"Required test file not found: {p}", flush=True)
            return 1

    ok_a = run_batch_test("TEST A", [soft_skills])
    ok_b = run_batch_test("TEST B", [soft_skills, qubit])
    ok_c = run_batch_test("TEST C", [soft_skills, java_fs])
    ok_d = run_batch_test("TEST D", [soft_skills, qubit, java_fs])

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY: TEST A={ok_a}, TEST B={ok_b}, TEST C={ok_c}, TEST D={ok_d}", flush=True)
    print(f"{'='*70}", flush=True)

    if all([ok_a, ok_b, ok_c, ok_d]):
        print("ALL 4 BATCH TESTS PASSED SUCCESSFULLY!", flush=True)
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())

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

    # Setup dummy session state
    import streamlit as st
    if not hasattr(st, "session_state"):
        st.session_state = {}

    st.session_state.settings = {"top_k": 5, "retrieval_mode": "both", "show_debug": True}
    st.session_state.active_dataset = {"files": [], "dataset_name": "No active dataset"}
    st.session_state.selected_source = None
    st.session_state.last_result = None

    pdf_path = Path(r"C:\Users\Arif\Documents\tasleeem\Certificates\Soft Skills.pdf")
    content = pdf_path.read_bytes()
    upload = MockUploadedFile(pdf_path.name, content)

    print("Running process_uploads([upload])...")
    try:
        app.process_uploads([upload])
        print("process_uploads SUCCESS!")
        print("Active Dataset:", st.session_state.active_dataset)
    except Exception as e:
        print("EXACT ERROR in process_uploads:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

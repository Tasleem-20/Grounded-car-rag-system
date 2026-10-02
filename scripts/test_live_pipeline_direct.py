"""Test the live run_car_rag_pipeline directly from app.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.image_store import ImageStore
from src.deterministic_detection import is_direct_object_detection_query, answer_detection_query
from app import run_car_rag_pipeline

def test_live_pipeline():
    meta_path = ROOT / "data" / "image_metadata.json"
    index_path = ROOT / "data" / "image.index"
    image_store = ImageStore.load(index_path, meta_path)
    print(f"Loaded {len(image_store.records)} image record(s) from disk.")
    rec = image_store.records[0]
    print(f"  Source: {rec.document_name}")
    print(f"  Stored counts: {rec.detection_counts}")

    queries = [
        ("A", "how many cars here"),
        ("B", "how many buses are there"),
        ("C", "what objects are detected"),
        ("D", "is there a car"),
        ("E", "how many people in the image"),
        ("F", "is there a dog"),
    ]

    print("\n=== TESTING APP.PY LIVE PIPELINE (ZERO GROQ FOR OBJECT DETECTION) ===")
    for tag, q in queries:
        print(f"\n[{tag}] QUERY: '{q}'")
        res = run_car_rag_pipeline(question=q, search_all=True)
        print(f"[{tag}] QUERY ROUTE: {res.query_analysis.query_mode}")
        print(f"[{tag}] GROUNDED ANSWER:\n{res.answer}")
        print(f"[{tag}] EVIDENCE SUFFICIENT: {res.evidence_sufficient}")
        print(f"[{tag}] GROUNDING VERIFIED: {res.grounding_verified}")
        print(f"[{tag}] EVIDENCE COUNT: {len(res.evidence)}")
        det_ev = [e for e in res.evidence if e.source_type == "object_detection"]
        print(f"[{tag}] OBJECT DETECTION EVIDENCE PRESENT: {bool(det_ev)}")

if __name__ == "__main__":
    test_live_pipeline()

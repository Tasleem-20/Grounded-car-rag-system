import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv()

from app import run_car_rag_pipeline

def main():
    q = "Who is this certificate issued to?"
    print(f"Testing query: '{q}'")
    res = run_car_rag_pipeline(q)
    print("Answer:", res.answer)
    print("Grounding Verified:", res.grounding_verified)
    print("Evidence Sufficient:", res.evidence_sufficient)
    print("Evidence Count:", len(res.evidence or []))

if __name__ == "__main__":
    main()

"""End-to-end verification for YOLO26n ingestion, persistence, and Ask RAG retrieval."""

import json
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.object_detection import ObjectDetector
from src.image_embeddings import ImageEmbeddingService
from src.image_store import ImageRecord, ImageStore
from src.retriever import MultimodalRetriever
from src.evidence import RetrievedEvidence


def make_test_image(path: Path) -> None:
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 300, 300], fill=(200, 50, 50), outline=(255, 255, 255), width=3)
    draw.rectangle([350, 150, 600, 400], fill=(50, 200, 50), outline=(255, 255, 255), width=3)
    img.save(path)


def test_yolo_detector():
    print("[1/5] Testing ObjectDetector loading & inference...")
    detector = ObjectDetector(model_name="yolo26n.pt")
    test_img_path = ROOT / "test_traffic.jpg"
    make_test_image(test_img_path)
    
    result = detector.detect(test_img_path)
    print(f" -> YOLO detection completed. Detections count: {len(result.detections)}")
    print(f" -> Summary: {result.summary}")
    print(f" -> RAG Context preview:\n{result.rag_context[:150]}...")
    assert isinstance(result.counts, dict)
    assert isinstance(result.rag_context, str)
    return detector, result, test_img_path


def test_persistence_and_reload(det_result, test_img_path):
    print("\n[2/5] Testing ImageRecord metadata serialization and persistence...")
    data_dir = ROOT / "data_test_e2e"
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path = data_dir / "image.index"
    meta_path = data_dir / "image_metadata.json"

    embedder = ImageEmbeddingService()
    emb = embedder.embed_images([test_img_path])[0]

    record = ImageRecord(
        image_id="traffic_1",
        modality="image",
        document_name="Traffic Scene.jpg",
        source_filename="Traffic Scene.jpg",
        image_path=str(test_img_path),
        page_number=None,
        caption="Traffic street scene with vehicles and pedestrians",
        source_type="standalone_image",
        width=640,
        height=480,
        detections=[d.to_dict() for d in det_result.detections],
        detection_summary=det_result.summary,
        detection_counts=det_result.counts,
        detection_context=det_result.rag_context,
    )

    store = ImageStore.from_embeddings(records=[record], embeddings=[emb])
    store.save(index_path, meta_path)

    print(" -> Saved to disk. Checking image_metadata.json content...")
    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    saved_record = raw_meta["records"][0]
    assert "detections" in saved_record
    assert "detection_counts" in saved_record
    assert "detection_context" in saved_record
    print(" -> Metadata JSON format verified with full detection fields.")

    print("\n[3/5] Testing restart simulation (ImageStore.load from disk)...")
    reloaded_store = ImageStore.load(index_path, meta_path)
    assert len(reloaded_store.records) == 1
    rel_rec = reloaded_store.records[0]
    assert rel_rec.document_name == "Traffic Scene.jpg"
    assert rel_rec.detection_summary == det_result.summary
    assert rel_rec.detection_counts == det_result.counts
    assert rel_rec.detection_context == det_result.rag_context
    print(" -> ImageStore successfully reloaded persisted detection data.")
    return reloaded_store


def test_retrieval(reloaded_store):
    print("\n[4/5] Testing MultimodalRetriever with object-count and vehicle queries...")
    embedder = ImageEmbeddingService()
    retriever = MultimodalRetriever(
        text_store=None,
        image_store=reloaded_store,
        text_embeddings=None,
        image_embeddings=embedder,
    )

    # Test 1: Object count question
    evidence_items = retriever.retrieve("How many cars are detected in the image?", top_k=5)
    print(f" -> Query 'How many cars are detected...': retrieved {len(evidence_items)} evidence item(s).")
    det_items = [e for e in evidence_items if e.source_type == "object_detection"]
    assert len(det_items) >= 1, "Expected object_detection evidence in retrieved items"
    print(f" -> Detected evidence text snippet:\n{det_items[0].text[:120]}...")

    # Test 2: General image query
    img_evidence = retriever.retrieve("Show me the traffic street photo", top_k=5, prefer_images=True)
    print(f" -> Query 'Show me the traffic...': retrieved {len(img_evidence)} item(s).")
    assert any(e.modality == "image" for e in img_evidence)
    print(" -> Image evidence retrieved properly.")


def cleanup():
    print("\n[5/5] Cleaning up test artifacts...")
    for p in [ROOT / "test_traffic.jpg", ROOT / "data_test_e2e"]:
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    print(" -> Cleaned up.")


def main():
    try:
        detector, det_result, test_img_path = test_yolo_detector()
        store = test_persistence_and_reload(det_result, test_img_path)
        test_retrieval(store)
        cleanup()
        print("\n=== ALL E2E INGESTION & RETRIEVAL VERIFICATION TESTS PASSED ===")
    except Exception as e:
        cleanup()
        print(f"\n[FAILED]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

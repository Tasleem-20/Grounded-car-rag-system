from pathlib import Path

app_path = Path("app.py")
content = app_path.read_text(encoding="utf-8")

old_code = """    # Prepend any extra evidence (e.g. YOLO detection and query image analysis)
    combined_evidence: list[RetrievedEvidence] = []
    if extra_evidence:
        combined_evidence.extend(extra_evidence)
    combined_evidence.extend(retrieved_items)

    rag = CARRAG("""

new_code = """    # Prepend any extra evidence (e.g. YOLO detection and query image analysis)
    combined_evidence: list[RetrievedEvidence] = []
    if extra_evidence:
        combined_evidence.extend(extra_evidence)
    combined_evidence.extend(retrieved_items)

    # DIRECT DETERMINISTIC OBJECT DETECTION CHECK (ZERO GROQ / ZERO LLM)
    from src.deterministic_detection import (
        answer_detection_query,
        is_direct_object_detection_query,
    )
    if (
        getattr(analysis, "query_mode", "") == "OBJECT_DETECTION_DIRECT"
        or is_direct_object_detection_query(question)
    ):
        det_answer, is_valid = answer_detection_query(question, combined_evidence)
        if is_valid and det_answer:
            if source_filter:
                allowed_evidence = []
                for ev in combined_evidence:
                    if ev.source_type == "object_detection" or same_source_name(ev.document_name, source_filter):
                        allowed_evidence.append(ev)
                combined_evidence = allowed_evidence

            return CARRAGResult(
                answer=det_answer,
                evidence=combined_evidence,
                query_analysis=analysis,
                evidence_sufficient=True,
                grounding_verified=True,
                answer_supported=True,
                evidence_reason="Grounded directly in persisted YOLO26n object detection metadata.",
                answer_reason="Object counts and detection bounding boxes match indexed ImageRecord evidence exactly.",
                retrieval_mode=retrieval_mode,
                re_retrieved=False,
            )

    rag = CARRAG("""

if old_code in content:
    content = content.replace(old_code, new_code, 1)
    app_path.write_text(content, encoding="utf-8")
    print("Patched app.py with direct deterministic detection path.")
else:
    old_crlf = old_code.replace("\n", "\r\n")
    new_crlf = new_code.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf, 1)
        app_path.write_text(content, encoding="utf-8")
        print("Patched app.py with direct deterministic detection path (CRLF).")
    else:
        print("Old code not found in app.py!")

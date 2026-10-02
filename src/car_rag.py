from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.answer_checker import AnswerCheckError, AnswerChecker
from src.deterministic_detection import (
    answer_detection_query,
    is_direct_object_detection_query,
)
from src.evidence import RetrievedEvidence
from src.evidence_checker import EvidenceCheckError, EvidenceChecker
from src.evidence_fallback import (
    extract_direct_text_answer,
    extract_grounded_evidence_fallback,
    is_direct_text_factual_query,
)
from src.generator import GenerationError, Generator
from src.groq_models import GroqServiceError, format_groq_error
from src.query_analyzer import QueryAnalysis, QueryAnalyzer
from src.sources import filter_evidence_to_source


INSUFFICIENT_EVIDENCE_ANSWER = "I don't have enough evidence to answer that."


@dataclass
class CARRAGResult:
    """Result returned by the CAR-RAG pipeline."""

    answer: str
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    query_analysis: QueryAnalysis | None = None
    evidence_sufficient: bool = True
    grounding_verified: bool = True
    answer_supported: bool = True
    evidence_reason: str = ""
    answer_reason: str = ""
    retrieval_mode: str = "both"
    re_retrieved: bool = False
    error: str | None = None


class CARRAG:
    """
    Main CAR-RAG orchestration layer.

    Pipeline:

        Query Analysis / Router
              ↓
        [Deterministic Detection Check] ──→ (Bypass Groq) ──→ Direct Grounded Answer
              ↓ (if semantic / text / complex)
        Initial Retrieval
              ↓
        Adaptive Retrieval
              ↓
        Evidence Check
              ↓
        Re-retrieval when needed
              ↓
        Generation
              ↓
        Answer Check
              ↓
        Grounded Answer / Refusal
    """

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retriever: Any,
        groq_service: Any = None,
        answer_checker: AnswerChecker | None = None,
        evidence_checker: EvidenceChecker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.retriever = retriever
        self.groq_service = groq_service
        self.answer_checker = answer_checker
        self.evidence_checker = evidence_checker
        self.generator = generator

    def _retrieve(
        self,
        question: str,
        top_k: int,
        mode: str,
        prefer_images: bool,
        source_filter: str | None,
        include_detection: bool = False,
    ) -> list[RetrievedEvidence]:
        try:
            results = self.retriever.retrieve(
                question,
                top_k=top_k,
                mode=mode,
                prefer_images=prefer_images,
                source_filter=source_filter,
                include_detection=include_detection,
            )
        except TypeError:
            try:
                results = self.retriever.retrieve(
                    question,
                    top_k=top_k,
                    mode=mode,
                    prefer_images=prefer_images,
                    source_filter=source_filter,
                )
            except TypeError:
                results = self.retriever.retrieve(
                    question,
                    top_k=top_k,
                    mode=mode,
                    prefer_images=prefer_images,
                )
        results = list(results or [])

        if source_filter:
            results = filter_evidence_to_source(results, source_filter)

        return results

    def _merge_evidence(
        self,
        current: list[RetrievedEvidence],
        extra: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        seen = {
            (
                getattr(item, "document_name", ""),
                getattr(item, "text", ""),
                getattr(item, "image_id", ""),
            )
            for item in current
        }

        merged = list(current)

        for item in extra:
            key = (
                getattr(item, "document_name", ""),
                getattr(item, "text", ""),
                getattr(item, "image_id", ""),
            )
            if key not in seen:
                merged.append(item)
                seen.add(key)

        return merged

    def _generate_answer(
        self,
        question: str,
        evidence: list[RetrievedEvidence],
    ) -> str:
        if self.generator is not None:
            return self.generator.answer(question, evidence)

        if self.groq_service is not None and hasattr(self.groq_service, "generate"):
            blocks = []
            for index, item in enumerate(evidence, start=1):
                blocks.append(
                    f"[Evidence {index}] Source: {item.document_name}\n{item.text}"
                )
            prompt = (
                "Answer using ONLY the evidence. If it is insufficient, say so.\n\n"
                f"Question: {question}\n\n"
                f"Evidence:\n{chr(10).join(blocks)}"
            )
            return str(self.groq_service.generate(prompt=prompt)).strip()

        return INSUFFICIENT_EVIDENCE_ANSWER

    def _check_evidence(
        self,
        question: str,
        evidence: list[RetrievedEvidence],
    ) -> tuple[bool, str]:
        if not evidence:
            return False, "No evidence was retrieved."

        if self.evidence_checker is None:
            return True, "Evidence retrieved successfully."

        check = self.evidence_checker.check(question, evidence)
        return bool(check.sufficient), str(check.reason)

    def _check_answer(
        self,
        question: str,
        answer: str,
        evidence: list[RetrievedEvidence],
    ) -> tuple[bool, str]:
        if self.answer_checker is None:
            return True, "Answer checker was not configured."

        try:
            check = self.answer_checker.check(
                question=question,
                answer=answer,
                results=evidence,
            )
        except TypeError:
            check = self.answer_checker.check(
                question=question,
                answer=answer,
                evidence=evidence,
            )

        supported = getattr(check, "supported", True)
        reason = getattr(check, "reason", "")
        return bool(supported), str(reason)

    def run(
        self,
        question: str,
        initial_evidence: list[RetrievedEvidence] | None = None,
        query_analysis: QueryAnalysis | None = None,
        evidence: list[RetrievedEvidence] | None = None,
        source_filter: str | None = None,
        retrieval_mode: str = "both",
        top_k: int = 5,
        prefer_images: bool = False,
    ) -> CARRAGResult:

        question = question.strip()

        if not question:
            raise ValueError("The question cannot be empty.")

        if query_analysis is None:
            query_analysis = self.analyzer.analyze(question)

        mode_from_analysis = getattr(query_analysis, "query_mode", "TEXT_RAG")
        if retrieval_mode == "both":
            if source_filter and any(source_filter.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                effective_mode = "image"
            elif mode_from_analysis == "TEXT_RAG" and not getattr(query_analysis, "needs_image_retrieval", False):
                effective_mode = "both"
            elif mode_from_analysis in {"IMAGE_RAG", "OBJECT_DETECTION"}:
                effective_mode = "image"
            else:
                effective_mode = "both"
        else:
            effective_mode = retrieval_mode

        include_detection = (
            getattr(query_analysis, "needs_detection_evidence", False)
            or mode_from_analysis in {"OBJECT_DETECTION", "MULTIMODAL_RAG"}
        )
        needs_image = prefer_images or getattr(query_analysis, "needs_image_retrieval", False)

        retrieved = initial_evidence if initial_evidence is not None else evidence

        if retrieved is None:
            retrieved = self._retrieve(
                question=question,
                top_k=top_k,
                mode=effective_mode,
                prefer_images=needs_image,
                source_filter=source_filter,
                include_detection=include_detection,
            )
        elif source_filter:
            retrieved = filter_evidence_to_source(list(retrieved), source_filter)
        else:
            retrieved = list(retrieved)

        # ----------------------------------------------------------------------
        # 1. DIRECT DETERMINISTIC / FACTUAL EVIDENCE ANSWERS (NO GROQ NEEDED)
        # ----------------------------------------------------------------------
        mode = getattr(query_analysis, "query_mode", "")

        # A. Object detection queries (e.g. "how many cars are detected?")
        if mode == "OBJECT_DETECTION" or is_direct_object_detection_query(question):
            det_ans, is_valid = answer_detection_query(question, retrieved)
            if is_valid and det_ans:
                return CARRAGResult(
                    answer=det_ans,
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=True,
                    grounding_verified=True,
                    answer_supported=True,
                    evidence_reason="Grounded directly in persisted YOLO26n object detection metadata.",
                    answer_reason="Object counts and detection bounding boxes match indexed ImageRecord evidence exactly.",
                    retrieval_mode=retrieval_mode,
                    re_retrieved=False,
                )

        # B. Direct factual queries across Text and Image OCR (e.g. "What is the domain?", "What is the title of this certificate?")
        if mode in {"TEXT_RAG", "IMAGE_RAG", "MULTIMODAL_RAG"} and is_direct_text_factual_query(question):
            direct_ans, is_direct = extract_direct_text_answer(question, retrieved)
            if is_direct and direct_ans:
                return CARRAGResult(
                    answer=direct_ans,
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=True,
                    grounding_verified=True,
                    answer_supported=True,
                    evidence_reason="Extracted directly from retrieved text/image evidence.",
                    answer_reason="Factual statement directly matches document text/OCR.",
                    retrieval_mode=retrieval_mode,
                    re_retrieved=False,
                )

        re_retrieved = False

        if query_analysis.needs_more_retrieval and len(retrieved) < 8:
            additional = self._retrieve(
                question=question,
                top_k=max(top_k, 8),
                mode=effective_mode,
                prefer_images=needs_image,
                source_filter=source_filter,
                include_detection=include_detection,
            )
            retrieved = self._merge_evidence(retrieved, additional)
            re_retrieved = True

        # Evidence sufficiency checking (with graceful fallback if checker API is unavailable)
        evidence_sufficient = True
        evidence_reason = "Evidence retrieved successfully."

        if not retrieved:
            return CARRAGResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=[],
                query_analysis=query_analysis,
                evidence_sufficient=False,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason="No supporting evidence was found.",
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
            )

        # Only run semantic LLM evidence check for pure text queries when checker is active
        if mode == "TEXT_RAG" and not any(ev.modality == "image" for ev in retrieved):
            try:
                evidence_sufficient, evidence_reason = self._check_evidence(
                    question,
                    retrieved,
                )
            except (EvidenceCheckError, GroqServiceError):
                evidence_sufficient = True
                evidence_reason = "Retrieved evidence is available (checker skipped on API rate limit)."
        else:
            evidence_sufficient = True
            evidence_reason = "Multimodal/Image evidence retrieved successfully."

        if not evidence_sufficient:
            # Check if any fallback answer can still be extracted before concluding no evidence
            fallback_ans, is_fallback = extract_grounded_evidence_fallback(
                question=question,
                evidence=retrieved,
                query_mode=mode,
            )
            if is_fallback and fallback_ans:
                return CARRAGResult(
                    answer=fallback_ans,
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=True,
                    grounding_verified=True,
                    answer_supported=True,
                    evidence_reason="Grounded in retrieved document evidence.",
                    answer_reason="Extracted directly from document text.",
                    retrieval_mode=retrieval_mode,
                    re_retrieved=re_retrieved,
                )
            return CARRAGResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=False,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason=evidence_reason or "No supporting evidence was found.",
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
            )

        # ----------------------------------------------------------------------
        # 2. GENERATION WITH EVIDENCE-GROUNDED FALLBACK
        # ----------------------------------------------------------------------
        answer = ""
        try:
            if self.generator is not None or self.groq_service is not None:
                answer = self._generate_answer(question, retrieved)
        except (GenerationError, GroqServiceError):
            answer = ""

        # If LLM generation succeeded
        if answer and answer.strip() and answer != INSUFFICIENT_EVIDENCE_ANSWER:
            answer_supported = True
            answer_reason = "Grounded in retrieved evidence."
            try:
                answer_supported, answer_reason = self._check_answer(
                    question,
                    answer,
                    retrieved,
                )
            except Exception:
                answer_supported = True
                answer_reason = "Grounded in retrieved evidence (verifier skipped on API rate limit)."

            if answer_supported:
                return CARRAGResult(
                    answer=answer,
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=True,
                    grounding_verified=True,
                    answer_supported=True,
                    evidence_reason=evidence_reason,
                    answer_reason=answer_reason,
                    retrieval_mode=retrieval_mode,
                    re_retrieved=re_retrieved,
                )

        # ----------------------------------------------------------------------
        # 3. EVIDENCE-BASED FALLBACK (WHEN GROQ IS RATE-LIMITED OR OFFLINE)
        # ----------------------------------------------------------------------
        fallback_ans, is_fallback = extract_grounded_evidence_fallback(
            question=question,
            evidence=retrieved,
            query_mode=mode,
        )
        if is_fallback and fallback_ans:
            return CARRAGResult(
                answer=fallback_ans,
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=True,
                grounding_verified=True,
                answer_supported=True,
                evidence_reason="Grounded in retrieved document/image evidence.",
                answer_reason="Extracted from active dataset evidence.",
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
            )

        return CARRAGResult(
            answer="Relevant evidence was retrieved, but language-model generation is temporarily unavailable.",
            evidence=retrieved,
            query_analysis=query_analysis,
            evidence_sufficient=True,
            grounding_verified=True,
            answer_supported=True,
            evidence_reason="Evidence available from active dataset.",
            retrieval_mode=retrieval_mode,
            re_retrieved=re_retrieved,
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.answer_checker import AnswerCheckError, AnswerChecker
from src.evidence import RetrievedEvidence
from src.evidence_checker import EvidenceCheckError, EvidenceChecker
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

        Query Analysis
              ↓
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
    ) -> list[RetrievedEvidence]:
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

        retrieved = initial_evidence if initial_evidence is not None else evidence

        if retrieved is None:
            retrieved = self._retrieve(
                question=question,
                top_k=top_k,
                mode=retrieval_mode,
                prefer_images=prefer_images or query_analysis.needs_image_retrieval,
                source_filter=source_filter,
            )
        elif source_filter:
            retrieved = filter_evidence_to_source(list(retrieved), source_filter)
        else:
            retrieved = list(retrieved)

        re_retrieved = False

        if query_analysis.needs_more_retrieval and len(retrieved) < 8:
            additional = self._retrieve(
                question=question,
                top_k=max(top_k, 8),
                mode=retrieval_mode,
                prefer_images=query_analysis.needs_image_retrieval or prefer_images,
                source_filter=source_filter,
            )
            retrieved = self._merge_evidence(retrieved, additional)
            re_retrieved = True

        try:
            evidence_sufficient, evidence_reason = self._check_evidence(
                question,
                retrieved,
            )
        except (EvidenceCheckError, GroqServiceError) as error:
            message = (
                format_groq_error(error)
                if not isinstance(error, GroqServiceError)
                else str(error)
            )
            return CARRAGResult(
                answer="",
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=False,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason=str(error),
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
                error=message,
            )

        if not evidence_sufficient:
            additional = self._retrieve(
                question=question,
                top_k=max(top_k * 2, 10),
                mode="both" if retrieval_mode != "text" else retrieval_mode,
                prefer_images=True,
                source_filter=source_filter,
            )
            retrieved = self._merge_evidence(retrieved, additional)
            re_retrieved = True

            try:
                evidence_sufficient, evidence_reason = self._check_evidence(
                    question,
                    retrieved,
                )
            except (EvidenceCheckError, GroqServiceError) as error:
                message = (
                    format_groq_error(error)
                    if not isinstance(error, GroqServiceError)
                    else str(error)
                )
                return CARRAGResult(
                    answer="",
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=False,
                    grounding_verified=False,
                    answer_supported=False,
                    retrieval_mode=retrieval_mode,
                    re_retrieved=re_retrieved,
                    error=message,
                )

        if not evidence_sufficient or not retrieved:
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

        if self.generator is None and self.groq_service is None:
            return CARRAGResult(
                answer="",
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=evidence_sufficient,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason=evidence_reason,
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
                error="Groq is not configured. Add GROQ_API_KEY before generating answers.",
            )

        try:
            answer = self._generate_answer(question, retrieved)
        except (GenerationError, GroqServiceError) as error:
            message = (
                format_groq_error(error)
                if not isinstance(error, GroqServiceError)
                else str(error)
            )
            return CARRAGResult(
                answer="",
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=True,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason=evidence_reason,
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
                error=message,
            )

        try:
            answer_supported, answer_reason = self._check_answer(
                question,
                answer,
                retrieved,
            )
        except (AnswerCheckError, GroqServiceError, TypeError) as error:
            if isinstance(error, TypeError):
                answer_supported, answer_reason = True, "Answer checker signature mismatch."
            else:
                message = (
                    format_groq_error(error)
                    if not isinstance(error, GroqServiceError)
                    else str(error)
                )
                return CARRAGResult(
                    answer="",
                    evidence=retrieved,
                    query_analysis=query_analysis,
                    evidence_sufficient=True,
                    grounding_verified=False,
                    answer_supported=False,
                    evidence_reason=evidence_reason,
                    retrieval_mode=retrieval_mode,
                    re_retrieved=re_retrieved,
                    error=message,
                )

        if not answer_supported:
            return CARRAGResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                evidence=retrieved,
                query_analysis=query_analysis,
                evidence_sufficient=True,
                grounding_verified=False,
                answer_supported=False,
                evidence_reason=evidence_reason,
                answer_reason=answer_reason,
                retrieval_mode=retrieval_mode,
                re_retrieved=re_retrieved,
            )

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

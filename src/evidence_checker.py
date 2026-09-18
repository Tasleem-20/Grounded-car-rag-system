"""Semantic evidence checker for self-checking CAR-RAG."""

from __future__ import annotations

from dataclasses import dataclass

from src.evidence import RetrievedEvidence
from src.groq_models import (
    VISION_MODEL,
    format_groq_error,
    groq_client,
    is_rate_limit_error,
    parse_json_object,
)


class EvidenceCheckError(Exception):
    """Raised when evidence checking fails."""


@dataclass
class EvidenceCheck:
    """Result of semantic evidence checking."""

    sufficient: bool
    score: float
    reason: str


class EvidenceChecker:
    """Use Groq to determine whether retrieved evidence supports a question."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            self.client = groq_client(api_key)
        except RuntimeError as error:
            raise EvidenceCheckError(str(error)) from error

        # Vision/judge model — not used for general text reasoning.
        self.model = model or VISION_MODEL

    def check(
        self,
        question: str,
        results: list[RetrievedEvidence],
    ) -> EvidenceCheck:

        if not question.strip():
            raise EvidenceCheckError(
                "The question cannot be empty."
            )

        if not results:
            return EvidenceCheck(
                sufficient=False,
                score=0.0,
                reason="No evidence was retrieved.",
            )

        context = "\n\n".join(
            result.evidence_block(position)
            for position, result in enumerate(results, start=1)
        )

        prompt = f"""
You are the evidence verification component of a CAR-RAG system.

Your job is ONLY to determine whether the retrieved evidence is sufficient
to answer the user's question.

USER QUESTION:
{question.strip()}

RETRIEVED EVIDENCE:
{context}

RULES:
- Use ONLY the retrieved evidence.
- Do not use outside knowledge.
- Text evidence and image evidence are equally valid.
- If the evidence directly supports the answer, sufficient must be true.
- If important information is missing, sufficient must be false.
- If the requested fact does not appear in the evidence, sufficient must be false.
- score must be a number from 0.0 to 1.0.
- reason must briefly explain the decision.
- Return ONLY one JSON object.
- Do NOT return markdown.
- Do NOT return explanations outside the JSON.

The required JSON format is:

{{
  "sufficient": true,
  "score": 0.95,
  "reason": "The retrieved evidence directly supports the question."
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evidence verification "
                            "component. Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_completion_tokens=500,
                response_format={"type": "json_object"},
                reasoning_effort="none",
            )

            message = response.choices[0].message

            content = message.content

            if not content:
                raise EvidenceCheckError(
                    "The evidence checker returned an empty response."
                )

            data = parse_json_object(content)

            sufficient_value = data.get("sufficient", False)

            if isinstance(sufficient_value, str):
                sufficient = sufficient_value.strip().lower() in {
                    "true",
                    "yes",
                    "1",
                }
            else:
                sufficient = bool(sufficient_value)

            try:
                score = float(data.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            score = max(0.0, min(1.0, score))

            reason = str(
                data.get(
                    "reason",
                    "The evidence could not be evaluated.",
                )
            ).strip()

            if not reason:
                reason = "The evidence could not be evaluated."

            return EvidenceCheck(
                sufficient=sufficient,
                score=score,
                reason=reason,
            )

        except EvidenceCheckError:
            raise

        except Exception as error:
            if is_rate_limit_error(error):
                raise EvidenceCheckError(format_groq_error(error)) from error
            raise EvidenceCheckError(format_groq_error(error)) from error
"""Semantic answer checker for self-checking CAR-RAG."""

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


class AnswerCheckError(Exception):
    """Raised when answer verification fails."""


@dataclass
class AnswerCheck:
    """Result of semantic answer verification."""

    supported: bool
    score: float
    reason: str


class AnswerChecker:
    """Check whether a generated answer is supported by retrieved evidence."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            self.client = groq_client(api_key)
        except RuntimeError as error:
            raise AnswerCheckError(str(error)) from error

        # Vision/judge model — not used for general text reasoning.
        self.model = model or VISION_MODEL

    def check(
        self,
        question: str,
        answer: str,
        results: list[RetrievedEvidence] | None = None,
        evidence: list[RetrievedEvidence] | None = None,
    ) -> AnswerCheck:

        if not question.strip():
            raise AnswerCheckError(
                "The question cannot be empty."
            )

        if not answer.strip():
            raise AnswerCheckError(
                "The answer cannot be empty."
            )

        results = results if results is not None else evidence

        if not results:
            return AnswerCheck(
                supported=False,
                score=0.0,
                reason="No retrieved evidence is available.",
            )

        context = "\n\n".join(
            result.evidence_block(position)
            for position, result in enumerate(
                results,
                start=1,
            )
        )

        prompt = f"""
You are the final answer verification component of a CAR-RAG system.

QUESTION:
{question.strip()}

GENERATED ANSWER:
{answer.strip()}

RETRIEVED EVIDENCE:
{context}

Determine whether the generated answer is supported by the retrieved
evidence.

RULES:
1. Use ONLY the retrieved evidence.
2. Do not use outside knowledge.
3. Every factual claim in the answer must be supported by the evidence.
4. If the answer contains unsupported or invented information,
   supported must be false.
5. A refusal caused by insufficient evidence is valid and may be marked true.
6. Text evidence and image evidence are both valid.
7. Visual claims must be supported by the retrieved image evidence
   or its caption/OCR.
8. score must be between 0.0 and 1.0.
9. reason must briefly explain the decision.
10. Return ONLY one JSON object.
11. Do NOT return markdown.
12. Do NOT return reasoning outside the JSON.

Return exactly this structure:

{{
    "supported": true,
    "score": 0.95,
    "reason": "The generated answer is supported by the retrieved evidence."
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=500,
                reasoning_effort="none",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict answer verification "
                            "component. Return only the requested "
                            "JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            message = response.choices[0].message
            content = message.content

            if not content:
                raise AnswerCheckError(
                    "The answer checker returned an empty response."
                )

            data = parse_json_object(content)

            supported_value = data.get(
                "supported",
                False,
            )

            if isinstance(supported_value, str):
                supported = (
                    supported_value.strip().lower()
                    in {"true", "yes", "1"}
                )
            else:
                supported = bool(supported_value)

            try:
                score = float(
                    data.get(
                        "score",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                score = 0.0

            score = max(
                0.0,
                min(1.0, score),
            )

            reason = str(
                data.get(
                    "reason",
                    "The answer could not be verified.",
                )
            ).strip()

            if not reason:
                reason = "The answer could not be verified."

            return AnswerCheck(
                supported=supported,
                score=score,
                reason=reason,
            )

        except AnswerCheckError:
            raise

        except Exception as error:
            if is_rate_limit_error(error):
                raise AnswerCheckError(format_groq_error(error)) from error
            raise AnswerCheckError(format_groq_error(error)) from error
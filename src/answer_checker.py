"""Semantic answer checker for self-checking RAG."""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from src.vector_store import RetrievedChunk

load_dotenv()


class AnswerCheckError(Exception):
    """Raised when answer verification fails."""


@dataclass
class AnswerCheck:
    """Result of semantic answer verification."""

    supported: bool
    score: float
    reason: str


class AnswerChecker:
    """Checks whether a generated answer is supported by retrieved evidence."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise AnswerCheckError(
                "GROQ_API_KEY is not configured. Add it to .env."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def check(
        self,
        question: str,
        answer: str,
        results: list[RetrievedChunk],
    ) -> AnswerCheck:

        if not question.strip():
            raise AnswerCheckError("The question cannot be empty.")

        if not answer.strip():
            raise AnswerCheckError("The answer cannot be empty.")

        if not results:
            return AnswerCheck(
                supported=False,
                score=0.0,
                reason="No retrieved evidence is available.",
            )

        context = "\n\n".join(
            f"[Evidence {position}]\n{result.text}"
            for position, result in enumerate(results, start=1)
        )

        prompt = f"""
Question:
{question.strip()}

Generated answer:
{answer.strip()}

Retrieved evidence:
{context}

Check whether the generated answer is fully supported by the
retrieved evidence.

Rules:
1. Use ONLY the retrieved evidence.
2. Do not use outside knowledge.
3. Every factual claim in the answer must be supported by the evidence.
4. If the answer contains unsupported or invented information,
   mark supported as false.
5. If the answer is a clear refusal because the retrieved evidence
   does not contain enough information to answer the question,
   mark supported as true.
6. A grounded refusal such as "The available documents do not
   contain enough information to answer this question" is valid
   when the evidence is insufficient.
7. Return ONLY valid JSON.

Return exactly:

{{
    "supported": true,
    "score": 0.95,
    "reason": "The generated answer is supported by the retrieved evidence."
}}

The score must be between 0.0 and 1.0.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict answer verification component "
                            "inside a Retrieval-Augmented Generation system. "
                            "Verify answers only against the provided evidence. "
                            "A refusal to answer due to insufficient evidence is valid "
                            "when the evidence checker has determined that the evidence "
                            "does not support the question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise AnswerCheckError(
                    "The answer checker returned an empty response."
                )

            data = json.loads(content)

            supported = bool(data.get("supported", False))
            score = float(data.get("score", 0.0))
            reason = str(
                data.get(
                    "reason",
                    "The answer could not be verified.",
                )
            )

            score = max(0.0, min(1.0, score))

            return AnswerCheck(
                supported=supported,
                score=score,
                reason=reason,
            )

        except AnswerCheckError:
            raise

        except Exception as error:
            raise AnswerCheckError(
                f"Answer verification failed: {error}"
            ) from error
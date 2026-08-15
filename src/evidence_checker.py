"""Semantic evidence checker for self-checking RAG."""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from src.vector_store import RetrievedChunk

load_dotenv()


class EvidenceCheckError(Exception):
    """Raised when evidence checking fails."""


@dataclass
class EvidenceCheck:
    """Result of semantic evidence checking."""

    sufficient: bool
    score: float
    reason: str


class EvidenceChecker:
    """Uses an LLM to determine whether retrieved evidence supports a question."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise EvidenceCheckError(
                "GROQ_API_KEY is not configured. Add it to .env."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def check(
        self,
        question: str,
        results: list[RetrievedChunk],
    ) -> EvidenceCheck:

        if not question.strip():
            raise EvidenceCheckError("The question cannot be empty.")

        if not results:
            return EvidenceCheck(
                sufficient=False,
                score=0.0,
                reason="No evidence was retrieved.",
            )

        context = "\n\n".join(
            f"[Evidence {position}]\n{result.text}"
            for position, result in enumerate(results, start=1)
        )

        prompt = f"""
Question:
{question.strip()}

Retrieved evidence:
{context}

Determine whether the retrieved evidence contains enough information
to answer the question accurately.

Important rules:
1. Use ONLY the retrieved evidence.
2. Do not use outside knowledge.
3. If the evidence directly supports the answer, mark sufficient as true.
4. If the evidence is missing important information, mark sufficient as false.
5. If the question asks for a fact that does not appear in the evidence,
   mark sufficient as false.
6. Return ONLY valid JSON.

Return exactly this structure:

{{
    "sufficient": true,
    "score": 0.95,
    "reason": "The evidence directly supports the question."
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
                            "You are an evidence verification component "
                            "inside a Retrieval-Augmented Generation system. "
                            "You must judge evidence strictly and never use "
                            "outside knowledge."
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
                raise EvidenceCheckError(
                    "The evidence checker returned an empty response."
                )

            data = json.loads(content)

            sufficient = bool(data.get("sufficient", False))
            score = float(data.get("score", 0.0))
            reason = str(
                data.get(
                    "reason",
                    "The evidence could not be evaluated.",
                )
            )

            score = max(0.0, min(1.0, score))

            return EvidenceCheck(
                sufficient=sufficient,
                score=score,
                reason=reason,
            )

        except EvidenceCheckError:
            raise

        except Exception as error:
            raise EvidenceCheckError(
                f"Semantic evidence checking failed: {error}"
            ) from error
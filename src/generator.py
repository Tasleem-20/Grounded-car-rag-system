"""Grounded answer generation using Groq."""

from __future__ import annotations

from dotenv import load_dotenv

from src.evidence import RetrievedEvidence
from src.groq_models import TEXT_MODEL, format_groq_error, groq_client, strip_thinking
from src.vector_store import RetrievedChunk

load_dotenv()


class GenerationError(Exception):
    """Raised when a grounded answer cannot be generated."""


SYSTEM_PROMPT = """You are the answer-generation component of a CAR-RAG system.

Answer the user's question using ONLY the retrieved evidence.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent citations.
- Do not add information that is not supported by the retrieved evidence.
- If the evidence is insufficient, say so clearly.
- Give a concise, direct answer.
- Return the actual answer only.
- Do not return reasoning or internal analysis.
"""


class Generator:
    """Generate grounded answers from retrieved text or image evidence."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        try:
            self.client = groq_client(api_key)
        except RuntimeError as error:
            raise GenerationError(str(error)) from error

        self.model = model or TEXT_MODEL

    def answer(
        self,
        question: str,
        results: list[RetrievedEvidence] | list[RetrievedChunk],
    ) -> str:

        if not question.strip():
            raise GenerationError(
                "The question cannot be empty."
            )

        if not results:
            return (
                "The available documents do not contain enough "
                "information to answer this question."
            )

        # Normalize retrieved chunks into RetrievedEvidence.
        evidence: list[RetrievedEvidence] = []

        for result in results:
            if isinstance(result, RetrievedEvidence):
                evidence.append(result)
            else:
                evidence.append(result.to_evidence())

        # If image evidence exists, try the multimodal vision pipeline first.
        if any(item.modality == "image" for item in evidence):
            try:
                from src.vision import answer_with_images

                answer = answer_with_images(
                    question,
                    evidence,
                )

                if answer and answer.strip():
                    return strip_thinking(answer).strip()

            except Exception:
                # If the image file is missing or vision model fails, gracefully fall back
                # to generating from the evidence blocks (which contain full OCR / captions).
                pass

        # Build grounded text context.
        context = "\n\n".join(
            item.evidence_block(position)
            for position, item in enumerate(
                evidence,
                start=1,
            )
        )

        user_prompt = f"""USER QUESTION:
{question.strip()}

RETRIEVED EVIDENCE:
{context}

Using only the retrieved evidence above, answer the user's question.

Return only the final answer."""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=700,
                    reasoning_effort="none",
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=700,
                )

        except Exception as error:
            raise GenerationError(format_groq_error(error)) from error

        try:
            message = response.choices[0].message

            answer = message.content

            # Some Groq models may expose reasoning separately.
            # We only want the final answer.
            if not answer:
                reasoning = getattr(
                    message,
                    "reasoning",
                    None,
                )

                if reasoning:
                    raise GenerationError(
                        "The model returned reasoning but no final answer."
                    )

                raise GenerationError(
                    "Groq returned an empty answer."
                )

            answer = strip_thinking(answer).strip()

        except GenerationError:
            raise

        except Exception as error:
            raise GenerationError(
                f"Could not read the Groq answer: {error}"
            ) from error

        if not answer:
            raise GenerationError(
                "Groq returned an empty answer after processing."
            )

        return answer
"""Grounded answer generation using Groq."""

import os

from dotenv import load_dotenv
from groq import Groq

from src.vector_store import RetrievedChunk

load_dotenv()


class GenerationError(Exception):
    """Raised when a grounded answer cannot be generated."""


SYSTEM_PROMPT = """You answer questions using only the provided retrieved context.
Do not invent facts, citations, or details that are not supported by the context.
If the context is insufficient, clearly say that the available documents do not
contain enough information to answer the question.
Keep the answer concise and directly address the question.
Do not include a separate sources list; the application will display the retrieved
evidence below your answer."""


class Generator:
    """Wrapper around Groq for grounded responses."""

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise GenerationError(
                "GROQ_API_KEY is not configured. Add it to .env."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def answer(self, question: str, results: list[RetrievedChunk]) -> str:
        if not question.strip():
            raise GenerationError("The question cannot be empty.")

        if not results:
            return (
                "The available documents do not contain enough information "
                "to answer this question."
            )

        context = "\n\n".join(
            f"[Source {position}: {result.document_name}, "
            f"chunk {result.chunk_index + 1}]\n"
            f"{result.text}"
            for position, result in enumerate(results, start=1)
        )

        user_prompt = f"""Question:
{question.strip()}

Retrieved context:
{context}

Write a concise answer grounded only in the retrieved context."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            answer = response.choices[0].message.content

        except Exception as error:
            if "401" in str(error) or "authentication" in str(error).lower():
                raise GenerationError(
                    "Groq rejected the configured API credential. "
                    "Check GROQ_API_KEY in .env."
                ) from error

            raise GenerationError(
                "Groq answer generation failed. Check your network, "
                "model access, and API limits."
            ) from error

        if not answer or not answer.strip():
            raise GenerationError("Groq returned an empty answer.")

        return answer.strip()
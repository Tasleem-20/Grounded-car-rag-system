"""Groq vision helpers for captions, OCR, and image-grounded generation."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

from src.evidence import RetrievedEvidence
from src.generator import GenerationError
from src.groq_models import format_groq_error, groq_client, strip_thinking


load_dotenv()


# Current multimodal model used for image understanding.
VISION_MODEL = "qwen/qwen3.8-27b"

MAX_VISION_IMAGES = 5
MAX_IMAGE_EDGE = 1280


class VisionError(Exception):
    """Raised when Groq vision processing fails."""


def _client() -> Groq:
    """Create the Groq client."""
    try:
        return groq_client()
    except RuntimeError as error:
        raise VisionError(str(error)) from error


def encode_image_data_url(image_path: str | Path) -> str:
    """Resize and JPEG-encode an image for Groq vision requests."""

    path = Path(image_path)

    if not path.is_file():
        raise VisionError(
            f"Image file is missing: {path}"
        )

    try:
        with Image.open(path) as raw_image:
            image = raw_image.convert("RGB")
        image.thumbnail(
            (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE)
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=85,
            optimize=True,
        )
        image.close()

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"

    except Exception as error:
        raise VisionError(
            f"Could not prepare image for Groq: {error}"
        ) from error


def caption_image(
    image_path: str | Path,
    source_name: str = "",
) -> str:
    """Describe an image and transcribe visible text."""

    data_url = encode_image_data_url(
        image_path
    )

    prompt = """
You are an image-document indexing component for a
Retrieval-Augmented Generation system.

Analyze the attached image carefully.

Provide:
1. A factual description of what is visible.
2. All important visible text.
3. Names, dates, IDs, titles, organizations,
   numbers, labels, and other useful document details.
4. Important visual structure such as tables,
   certificates, charts, diagrams, or sections.

Do not speculate.
Do not invent information.
Only describe information that is actually visible.
Return the result as plain text.
"""

    if source_name:
        prompt += f"\nSource file: {source_name}"

    try:
        response = _client().chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            max_completion_tokens=700,
            reasoning_effort="none",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document image analysis "
                        "component. Analyze only the supplied image."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                },
            ],
        )

        message = response.choices[0].message
        content = message.content

    except VisionError:
        raise

    except Exception as error:
        raise VisionError(format_groq_error(error)) from error

    if not content:
        raise VisionError(
            "Groq vision returned no final content."
        )

    caption = strip_thinking(
        content
    ).strip()

    if not caption:
        raise VisionError(
            "Groq vision returned an empty caption."
        )

    return caption


def fallback_caption(
    source_name: str,
    page_number: int | None = None,
) -> str:
    """Fallback metadata when visual captioning is unavailable."""

    page = (
        f", page {page_number}"
        if page_number
        else ""
    )

    return (
        f"Image from {source_name}{page}. "
        "Caption unavailable; retrieve by visual similarity."
    )


def answer_with_images(
    question: str,
    evidence: list[RetrievedEvidence],
) -> str:
    """Generate an answer using retrieved text and image evidence."""

    if not question.strip():
        raise GenerationError(
            "The question cannot be empty."
        )

    image_items = [
        item
        for item in evidence
        if (
            item.modality == "image"
            and item.image_path
        )
    ][:MAX_VISION_IMAGES]

    if not image_items:
        raise GenerationError(
            "No usable image evidence was retrieved."
        )

    text_context = "\n\n".join(
        item.evidence_block(position)
        for position, item in enumerate(
            evidence,
            start=1,
        )
    )

    user_content: list[dict] = [
        {
            "type": "text",
            "text": f"""
Answer the user's question using ONLY the
retrieved evidence and attached document images.

Do not use outside knowledge.
Do not invent facts.
Read the visible text in the images carefully.
Pay particular attention to names, dates,
certificate IDs, titles, organizations,
numbers, and other exact values.

If the evidence does not contain enough information,
say so clearly.

USER QUESTION:
{question.strip()}

RETRIEVED EVIDENCE:
{text_context}

Return ONLY the final answer.
Do not return reasoning.
""",
        }
    ]

    for item in image_items:
        try:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image_data_url(
                            item.image_path
                        ),
                    },
                }
            )
        except VisionError:
            continue

    if len(user_content) == 1:
        raise GenerationError(
            "The retrieved image evidence could not "
            "be prepared for the vision model."
        )

    try:
        response = _client().chat.completions.create(
            model=VISION_MODEL,
            temperature=0.1,
            max_completion_tokens=700,
            reasoning_effort="none",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document image question-answering "
                        "component. Answer only from the supplied "
                        "retrieved evidence and images. "
                        "Return the final answer only."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        message = response.choices[0].message
        answer = message.content

    except Exception as error:
        raise GenerationError(format_groq_error(error)) from error

    if not answer:
        raise GenerationError(
            "Groq vision returned no final answer."
        )

    answer = strip_thinking(
        answer
    ).strip()

    if not answer:
        raise GenerationError(
            "Groq vision returned an empty answer."
        )

    return answer
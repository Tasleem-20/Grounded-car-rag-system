from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx
from groq import Groq

load_dotenv()


TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"


class GroqServiceError(Exception):
    """Raised when a Groq call fails in a user-facing way."""


def _get_api_key() -> str:
    return (
        os.getenv("GROQ_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
    ).strip()


def groq_client(api_key: str | None = None) -> Groq:
    """Return a Groq client. Raises if no API key is configured."""

    key = (api_key or _get_api_key()).strip()

    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    timeout = httpx.Timeout(60.0, connect=15.0, read=60.0, write=60.0)
    return Groq(api_key=key, timeout=timeout, max_retries=2)


def is_rate_limit_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "rate_limit",
            "rate_limit_exceeded",
            "too many requests",
            "tokens per minute",
            "token quota",
            "tpm",
            "429",
        )
    )


def format_groq_error(error: BaseException) -> str:
    """Professional message for the UI. Never includes API keys."""

    text = str(error).lower()

    if is_rate_limit_error(error):
        return "Groq rate limit reached. Please retry shortly."

    if "not configured" in text or "groq_api_key is not" in text:
        return "Groq is not configured. Add GROQ_API_KEY before generating answers."

    if any(
        token in text
        for token in ("401", "403", "authentication", "invalid api", "api key")
    ):
        return "Groq authentication failed. Check that GROQ_API_KEY is configured."

    if "model" in text and any(
        token in text for token in ("not found", "decommissioned", "does not exist")
    ):
        return "The configured Groq model is not available for this account."

    return "The language model is currently unavailable. Please try again."


def strip_thinking(text: str) -> str:
    """Remove model thinking blocks, leaving the final answer."""

    if not text:
        return ""

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<thinking>.*?</thinking>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def parse_json_object(value: Any) -> dict:
    """
    Safely parse a JSON object returned by an LLM.

    Handles:
    - normal JSON
    - JSON inside ```json blocks
    - surrounding text
    """

    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    text = str(value).strip()

    if not text:
        return {}

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start : end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return {}


class GroqModelService:
    """Groq text and vision model service."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:

        self.api_key = (api_key or _get_api_key()).strip()

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        model: str = TEXT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> str:

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            raise GroqServiceError(format_groq_error(error)) from error

        if not response.choices:
            return ""

        content = response.choices[0].message.content

        return strip_thinking(str(content or "")).strip()

    def generate_answer(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:

        return self.generate(prompt=prompt, **kwargs)

    def generate_json(
        self,
        prompt: str,
        model: str = TEXT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> dict:

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            raise GroqServiceError(format_groq_error(error)) from error

        if not response.choices:
            return {}

        content = response.choices[0].message.content

        return parse_json_object(content)

    def caption_image(
        self,
        image_path: str | Path,
        prompt: str | None = None,
    ) -> str:

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = image_path.read_bytes()

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }

        mime_type = mime_types.get(
            image_path.suffix.lower(),
            "image/jpeg",
        )

        if prompt is None:
            prompt = (
                "Analyze this certificate or document image carefully. "
                "Extract visible names, certificate titles, course names, "
                "marks, dates, organizations, logos, signatures, and "
                "other important information. "
                "Return a concise factual description. "
                "Do not invent information."
            )

        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
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
                                    "url": (
                                        f"data:{mime_type};base64,"
                                        f"{encoded_image}"
                                    )
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        except Exception as error:
            raise GroqServiceError(format_groq_error(error)) from error

        if not response.choices:
            return ""

        content = response.choices[0].message.content

        return strip_thinking(str(content or "")).strip()


def generate(
    prompt: str,
    model: str = TEXT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> str:

    service = GroqModelService()

    return service.generate(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def caption_image(
    image_path: str | Path,
    prompt: str | None = None,
) -> str:

    service = GroqModelService()

    return service.caption_image(
        image_path=image_path,
        prompt=prompt,
    )

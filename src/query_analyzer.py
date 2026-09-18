"""Query analysis for adaptive RAG."""

from dataclasses import dataclass
from typing import Literal


QueryType = Literal["simple", "complex"]


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""

    query: str
    query_type: QueryType
    needs_more_retrieval: bool
    needs_image_retrieval: bool = False


class QueryAnalyzer:
    """Analyze a question before retrieval."""

    def analyze(self, question: str) -> QueryAnalysis:
        question = question.strip()

        if not question:
            raise ValueError("The question cannot be empty.")

        # Simple heuristic for the first version.
        # We will replace this with an LLM-based analyzer later.
        complex_indicators = [
            "and",
            "compare",
            "difference",
            "relationship",
            "why",
            "how does",
            "how do",
            "explain",
            "both",
            "multiple",
        ]
        visual_indicators = [
            "image",
            "picture",
            "photo",
            "diagram",
            "figure",
            "chart",
            "logo",
            "screenshot",
            "visual",
            "shown",
            "looks like",
            "color",
            "icon",
            "certificate",
            "certification",
            "signature",
            "stamp",
            "seal",
            "person on",
            "name of the person",
            "shown on the",
        ]

        question_lower = question.lower()

        is_complex = any(
            indicator in question_lower
            for indicator in complex_indicators
        )
        needs_image_retrieval = any(
            indicator in question_lower
            for indicator in visual_indicators
        )

        if is_complex:
            return QueryAnalysis(
                query=question,
                query_type="complex",
                needs_more_retrieval=True,
                needs_image_retrieval=needs_image_retrieval,
            )

        return QueryAnalysis(
            query=question,
            query_type="simple",
            needs_more_retrieval=False,
            needs_image_retrieval=needs_image_retrieval,
        )
"""Query analysis and semantic routing for multimodal CAR-RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


QueryType = Literal["simple", "complex"]
QueryMode = Literal["TEXT_RAG", "IMAGE_RAG", "OBJECT_DETECTION", "MULTIMODAL_RAG"]


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""

    query: str
    query_type: QueryType
    needs_more_retrieval: bool
    needs_image_retrieval: bool = False
    needs_detection_evidence: bool = False
    query_mode: QueryMode = "TEXT_RAG"


class QueryAnalyzer:
    """Analyze a question before retrieval and route to appropriate pipeline."""

    def analyze(self, question: str) -> QueryAnalysis:
        question = question.strip()

        if not question:
            raise ValueError("The question cannot be empty.")

        q_lower = question.lower()
        norm = " ".join(re.sub(r"[^\w\s]", " ", q_lower).split())

        # 1. Document / text indicators
        text_indicators = [
            "policy", "rules", "guidelines",
            "report", "manual", "statement", "terms", "text", "pdf", "file",
            "document", "section", "paragraph", "article",
            "say about", "says about", "according to", "written in", "mentioned in",
            "information does the document", "information does the pdf", "provide about",
            "information provide", "information does", "what is abs", "who is",
            "roll number", "marks", "score", "grade", "main topic", "summarize", "summary",
            "domain", "duration", "internship", "technologies", "technology", "skills",
            "tools", "database", "certificate id", "certificate number", "issued", "awarded", "credential",
            "mentioned in", "written in", "stated in", "what does the certificate say",
            "topics", "complete summary",
        ]
        is_text_query = any(ind in norm for ind in text_indicators)

        # 2. Visual / Image descriptive indicators
        vision_indicators = [
            "image", "picture", "photo", "diagram", "figure", "chart", "logo",
            "screenshot", "visual", "visual design", "looks like", "color", "icon",
            "describe", "what is happening", "what is the person doing",
            "scene", "street scene", "wearing", "background", "view",
            "signature", "stamp", "seal", "badge",
        ]
        is_vision_query = any(re.search(rf"\b{re.escape(ind)}\b", norm) for ind in vision_indicators)

        # 3. Object Detection / Count indicators
        from src.deterministic_detection import is_direct_object_detection_query
        is_object_query = is_direct_object_detection_query(question)

        # 4. Complexity check
        complex_indicators = [
            "compare", "difference", "relationship", "why", "how does", "how do",
            "explain", "both", "multiple", "and what", "and how",
        ]
        is_complex = any(ind in norm for ind in complex_indicators)

        # 5. Semantic Routing
        # A. OBJECT_DETECTION: Pure count/presence/detection question on objects
        if is_object_query:
            if is_text_query and any(w in norm for w in ["compare", "document", "pdf", "both"]):
                mode: QueryMode = "MULTIMODAL_RAG"
                needs_image = True
                needs_det = True
            else:
                mode = "OBJECT_DETECTION"
                needs_image = True
                needs_det = True

        # B. Pure visual/scene description/appearance question
        elif is_vision_query and not is_text_query:
            mode = "IMAGE_RAG"
            needs_image = True
            needs_det = False

        # C. Both visual and text cues
        elif is_vision_query and is_text_query:
            # Check if primarily visual request (e.g. "describe the visual design of this certificate")
            if any(w in norm for w in ["visual design", "visual", "look like", "looks like", "color", "appearance", "design of", "describe the certificate", "describe the image", "describe the"]):
                mode = "IMAGE_RAG"
                needs_image = True
                needs_det = False
            elif any(w in norm for w in ["compare", "both", "document and image", "text and image"]):
                mode = "MULTIMODAL_RAG"
                needs_image = True
                needs_det = False
            else:
                # Text/factual query about a certificate/document
                mode = "TEXT_RAG"
                needs_image = False
                needs_det = False

        # D. Pure text/document question
        elif is_text_query:
            mode = "TEXT_RAG"
            needs_image = False
            needs_det = False

        # E. Fallback
        elif is_vision_query:
            mode = "IMAGE_RAG"
            needs_image = True
            needs_det = False
        else:
            mode = "TEXT_RAG"
            needs_image = False
            needs_det = False

        return QueryAnalysis(
            query=question,
            query_type="complex" if is_complex else "simple",
            needs_more_retrieval=is_complex,
            needs_image_retrieval=needs_image,
            needs_detection_evidence=needs_det,
            query_mode=mode,
        )
"""Meaningful, metadata-preserving text chunking."""

from dataclasses import dataclass

from src.document_loader import LoadedDocument


@dataclass(frozen=True)
class TextChunk:
    """A searchable portion of a document with traceable source offsets."""

    document_name: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int

    def to_metadata(self) -> dict:
        return {
            "document_name": self.document_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "TextChunk":
        return cls(
            document_name=str(metadata["document_name"]),
            chunk_index=int(metadata["chunk_index"]),
            text=str(metadata["text"]),
            start_char=int(metadata["start_char"]),
            end_char=int(metadata["end_char"]),
        )


def _find_boundary(text: str, start: int, proposed_end: int) -> int:
    """Prefer paragraph, sentence, or word boundaries over hard cuts."""

    candidates = [
        text.rfind("\n\n", start + 1, proposed_end),
        text.rfind(". ", start + 1, proposed_end),
        text.rfind(" ", start + 1, proposed_end),
    ]
    boundary = max(candidates)
    return boundary + 1 if boundary > start else proposed_end


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split a document into overlapping chunks with source character offsets."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = document.text
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end
        if proposed_end < len(text):
            end = _find_boundary(text, start, proposed_end)

        chunk_text = text[start:end].strip()
        if chunk_text:
            leading_whitespace = len(text[start:end]) - len(text[start:end].lstrip())
            actual_start = start + leading_whitespace
            actual_end = actual_start + len(chunk_text)
            chunks.append(
                TextChunk(
                    document_name=document.name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=actual_start,
                    end_char=actual_end,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
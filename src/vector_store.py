"""Local FAISS vector index and persisted chunk metadata."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.chunker import TextChunk


class VectorStoreError(Exception):
    """Raised when a vector index cannot be created, loaded, or searched."""


class RetrievedChunk:
    """A source chunk plus its cosine-similarity score."""

    def __init__(self, chunk: TextChunk, score: float) -> None:
        self.document_name = chunk.document_name
        self.chunk_index = chunk.chunk_index
        self.text = chunk.text
        self.start_char = chunk.start_char
        self.end_char = chunk.end_char
        self.score = float(score)


class VectorStore:
    """FAISS inner-product index over normalized embedding vectors."""

    def __init__(self, index: faiss.Index, chunks: list[TextChunk]) -> None:
        self.index = index
        self.chunks = chunks

    @property
    def document_names(self) -> list[str]:
        return list(dict.fromkeys(chunk.document_name for chunk in self.chunks))

    @classmethod
    def from_embeddings(
        cls,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> "VectorStore":
        if not chunks or not embeddings:
            raise VectorStoreError("Cannot build an index without chunks and embeddings.")
        if len(chunks) != len(embeddings):
            raise VectorStoreError("The chunk and embedding counts do not match.")

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise VectorStoreError("Embeddings must be a two-dimensional matrix.")
        if not np.isfinite(matrix).all():
            raise VectorStoreError("Embeddings contain invalid numeric values.")

        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, chunks=chunks)

    @staticmethod
    def exists(index_path: Path, metadata_path: Path) -> bool:
        return index_path.is_file() and metadata_path.is_file()

    def save(self, index_path: Path, metadata_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(index_path))
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "chunks": [chunk.to_metadata() for chunk in self.chunks],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to save the FAISS index: {error}") from error

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "VectorStore":
        try:
            index = faiss.read_index(str(index_path))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            chunks = [TextChunk.from_metadata(item) for item in payload["chunks"]]
        except Exception as error:
            raise VectorStoreError(f"Unable to load the saved index: {error}") from error

        if index.ntotal != len(chunks):
            raise VectorStoreError("The saved index and metadata are out of sync.")
        return cls(index=index, chunks=chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not self.chunks or self.index.ntotal == 0:
            raise VectorStoreError("No vector index is available. Process documents first.")
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")

        query = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                results.append(RetrievedChunk(self.chunks[int(index)], float(score)))
        return results
"""Question embedding and fused text + image evidence retrieval."""

from typing import Literal

from src.embeddings import EmbeddingService
from src.evidence import RetrievedEvidence
from src.image_embeddings import ImageEmbeddingService
from src.image_store import ImageStore
from src.sources import filter_evidence_to_source
from src.vector_store import RetrievedChunk, VectorStore


RetrievalMode = Literal["both", "text", "image"]


class Retriever:
    """Connects the embedding service to the local vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_query(question)
        return self.vector_store.search(query_embedding, top_k=top_k)


class MultimodalRetriever:
    """Retrieve text (MiniLM/FAISS) and images (CLIP/FAISS) independently, then fuse."""

    def __init__(
        self,
        text_store: VectorStore | None,
        image_store: ImageStore | None,
        text_embeddings: EmbeddingService | None,
        image_embeddings: ImageEmbeddingService | None,
    ) -> None:
        self.text_store = text_store
        self.image_store = image_store
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    @staticmethod
    def _index_size(store: VectorStore | ImageStore | None) -> int:
        if store is None:
            return 0
        index = getattr(store, "index", None)
        return int(getattr(index, "ntotal", 0) or 0)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        mode: RetrievalMode = "both",
        prefer_images: bool = False,
        source_filter: str | None = None,
    ) -> list[RetrievedEvidence]:
        text_results: list[RetrievedEvidence] = []
        image_results: list[RetrievedEvidence] = []
        text_fetch_k = top_k
        image_fetch_k = top_k

        if source_filter:
            if self._index_size(self.text_store):
                text_fetch_k = max(top_k, self._index_size(self.text_store))
            if self._index_size(self.image_store):
                image_fetch_k = max(top_k, self._index_size(self.image_store))

        if mode in {"both", "text"} and self.text_store and self.text_embeddings:
            text_retriever = Retriever(self.text_store, self.text_embeddings)
            text_results = [
                item.to_evidence()
                for item in text_retriever.retrieve(question, top_k=text_fetch_k)
            ]
            if source_filter:
                text_results = filter_evidence_to_source(text_results, source_filter)[
                    :top_k
                ]

        if mode in {"both", "image"} and self.image_store and self.image_embeddings:
            query_embedding = self.image_embeddings.embed_query(question)
            image_results = self.image_store.search(
                query_embedding, top_k=image_fetch_k
            )
            if source_filter:
                image_results = filter_evidence_to_source(image_results, source_filter)[
                    :top_k
                ]

        if prefer_images:
            return image_results + text_results
        return text_results + image_results

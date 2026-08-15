"""Question embedding and top-k evidence retrieval."""

from src.embeddings import EmbeddingService
from src.vector_store import RetrievedChunk, VectorStore


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
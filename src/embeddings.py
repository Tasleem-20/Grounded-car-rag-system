"""Local embedding service used by indexing and retrieval."""

from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingService:
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            self.model = SentenceTransformer(model)
        except Exception as error:
            raise EmbeddingError(
                f"Could not load the local embedding model: {error}"
            ) from error

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise EmbeddingError("There is no text to embed.")

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return embeddings.tolist()

        except Exception as error:
            raise EmbeddingError(
                f"Could not generate embeddings: {error}"
            ) from error

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("The question cannot be empty.")

        return self.embed_texts([query])[0]
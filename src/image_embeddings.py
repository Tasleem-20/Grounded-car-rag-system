"""CLIP embeddings for the separate image retrieval index."""

from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer

from src.embeddings import EmbeddingError


DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"


class ImageEmbeddingService:
    """Embed images and text queries in a shared CLIP vector space."""

    def __init__(self, model: str = DEFAULT_IMAGE_MODEL) -> None:
        try:
            self.model = SentenceTransformer(model)
            self.model_name = model
        except Exception as error:
            raise EmbeddingError(
                f"Could not load the image embedding model: {error}"
            ) from error

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            raise EmbeddingError("There are no images to embed.")

        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            embeddings = self.model.encode(
                images,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for image in images:
                image.close()
            return embeddings.tolist()
        except Exception as error:
            raise EmbeddingError(
                f"Could not generate image embeddings: {error}"
            ) from error

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise EmbeddingError("The question cannot be empty.")

        try:
            embeddings = self.model.encode(
                [query.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()[0]
        except Exception as error:
            raise EmbeddingError(
                f"Could not embed the image-search query: {error}"
            ) from error

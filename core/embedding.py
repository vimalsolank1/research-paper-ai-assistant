"""
embedding.py — HuggingFace embedding model manager.

Wraps sentence-transformers to generate text vectors for FAISS indexing.
Uses a singleton pattern so the model loads only once across the entire app.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingManager:
    """
    Loads and manages the HuggingFace embedding model.

    Singleton: model is loaded once into memory and reused by all instances.
    Default model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
    """

    # Shared across all instances — None until first instantiation
    _model: HuggingFaceEmbeddings | None = None

    def __init__(self, model_name: str = None):
        """
        Load the embedding model if not already loaded.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.EMBEDDING_MODEL.
        """
        self.model_name = (
            model_name
            or settings.EMBEDDING_MODEL
            or "sentence-transformers/all-mpnet-base-v2"
        )

        # Only load once — subsequent instantiations reuse the cached model
        if EmbeddingManager._model is None:
            EmbeddingManager._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},  # change to "cuda" if GPU available
                encode_kwargs={
                    "normalize_embeddings": True,  # required for cosine similarity
                    "batch_size": 32
                }
            )

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Return the loaded model. Used by VectorStoreManager to pass into FAISS."""
        return EmbeddingManager._model

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace before embedding."""
        return " ".join(text.split())

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string (used for query-side encoding).

        Args:
            text: Query or short text to embed.

        Returns:
            List of floats — 768-dimensional vector.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self.model.embed_query(self._clean_text(text))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts (used for document-side encoding).

        More efficient than calling embed_text() in a loop.

        Args:
            texts: List of strings to embed. Empty strings are filtered out.

        Returns:
            List of vectors, one per input text.

        Raises:
            ValueError: If list is empty or all texts are empty.
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        texts = [self._clean_text(t) for t in texts if t.strip()]

        if not texts:
            raise ValueError("All texts were empty after cleaning")

        return self.model.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """
        Return the vector size of the current model.

        Returns:
            int: 768 for all-mpnet-base-v2, 384 for all-MiniLM-L6-v2.
        """
        return len(self.embed_text("dimension probe"))
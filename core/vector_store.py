"""
vector_store.py — FAISS vector store manager.

Handles creating, searching, saving, and loading the FAISS index.
All paper chunks are stored here as vectors for semantic search.
"""

import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from core.embedding import EmbeddingManager


class VectorStoreManager:
    """
    Wraps FAISS to manage document indexing and semantic search.

    Supports:
    - Adding documents to the index
    - Similarity search with or without scores
    - Saving/loading index to disk for fast restarts
    - LangChain retriever interface for the RAG chain
    """

    def __init__(self, embedding_manager: EmbeddingManager = None):
        """
        Args:
            embedding_manager: Handles text-to-vector conversion.
                Creates a default EmbeddingManager if not provided.
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self._vector_store: Optional[FAISS] = None
        self.index_path = settings.FAISS_INDEX_PATH

    @property
    def vector_store(self) -> Optional[FAISS]:
        """Return the underlying FAISS instance (None if not initialized)."""
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """True if the index has been created or loaded."""
        return self._vector_store is not None

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        """Remove documents with empty content before indexing."""
        return [doc for doc in documents if doc.page_content.strip()]

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Build a new FAISS index from a list of documents.

        Args:
            documents: Chunks to embed and index.

        Returns:
            The created FAISS instance.

        Raises:
            ValueError: If all documents are empty after filtering.
        """
        docs = self._filter_documents(documents)

        if not docs:
            raise ValueError("No valid documents to index")

        self._vector_store = FAISS.from_documents(docs, self.embedding_manager.model)
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the existing index, or create one if none exists.

        Args:
            documents: New chunks to add.
        """
        docs = self._filter_documents(documents)

        if not docs:
            return

        if not self.is_initialized:
            self.create_from_documents(docs)
        else:
            self._vector_store.add_documents(docs)

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Find the top-k most similar documents to the query.

        Args:
            query: User's search query.
            k: Number of results to return. Defaults to settings.TOP_K_RESULTS.

        Returns:
            List of matching Documents ordered by similarity.

        Raises:
            ValueError: If the index has not been initialized.
        """
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Same as search() but also returns a similarity score per document.

        Lower score = more similar (L2 distance).

        Args:
            query: User's search query.
            k: Number of results. Defaults to settings.TOP_K_RESULTS.

        Returns:
            List of (Document, score) tuples.

        Raises:
            ValueError: If the index has not been initialized.
        """
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.similarity_search_with_score(query, k=k)

    def save(self, path: str = None) -> None:
        """
        Save the FAISS index to disk so it can be reloaded on restart.

        Args:
            path: Directory to save to. Defaults to settings.FAISS_INDEX_PATH.

        Raises:
            ValueError: If no index exists to save.
        """
        if not self.is_initialized:
            raise ValueError("Nothing to save — index is empty")

        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)
        self._vector_store.save_local(save_path)

    def load(self, path: str = None) -> FAISS:
        """
        Load a previously saved FAISS index from disk.

        Args:
            path: Directory to load from. Defaults to settings.FAISS_INDEX_PATH.

        Returns:
            The loaded FAISS instance.

        Raises:
            FileNotFoundError: If no index exists at the given path.
        """
        load_path = path or self.index_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No FAISS index found at {load_path}")

        self._vector_store = FAISS.load_local(
            load_path,
            self.embedding_manager.model,
            allow_dangerous_deserialization=True  # required by FAISS for local loading
        )
        return self._vector_store

    def get_retriever(self, k: int = None):
        """
        Return a LangChain-compatible retriever for use in the RAG chain.

        Args:
            k: Number of documents to retrieve per query.

        Returns:
            LangChain VectorStoreRetriever.

        Raises:
            ValueError: If the index has not been initialized.
        """
        if not self.is_initialized:
            raise ValueError("Vector store not initialized")

        k = k or settings.TOP_K_RESULTS
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def clear(self) -> None:
        """Reset the vector store (drops all indexed documents from memory)."""
        self._vector_store = None
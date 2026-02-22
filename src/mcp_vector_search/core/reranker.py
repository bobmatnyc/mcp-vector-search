"""Cross-encoder reranking for higher precision search results.

Cross-encoders process (query, document) pairs jointly, producing more accurate
relevance scores than bi-encoder (embedding-based) similarity. This module provides
reranking capabilities to improve search precision after initial retrieval.

Architecture:
    1. Initial retrieval: Bi-encoder (fast, broad recall)
    2. Reranking: Cross-encoder (slow, high precision)
    3. Diversity: MMR (if enabled)

The cross-encoder model scores each (query, document) pair directly, allowing it to
capture semantic interactions that bi-encoders miss.
"""

from typing import Any

from loguru import logger


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model for higher precision.

    Cross-encoders process (query, document) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity.

    This reranker uses the ms-marco-MiniLM cross-encoder model, which is:
    - Small (22MB) and fast enough for interactive search
    - Trained on MS MARCO passage ranking dataset
    - Optimized for ranking relevance (not just semantic similarity)

    Usage:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(
            query="parse file into chunks",
            documents=["def parse_file(path):", "class Config:", "import os"],
            top_k=5
        )
        # Returns: [(0, 0.95), (2, 0.45), (1, 0.12)] - (index, score) tuples
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22MB, fast

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: Model name/path (default: ms-marco-MiniLM-L-6-v2)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: Any = None  # Lazy loading (type: CrossEncoder)

    def _ensure_model(self) -> Any:
        """Lazy-load the cross-encoder model.

        Returns:
            Loaded CrossEncoder model

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                ) from e

            logger.debug(f"Loading cross-encoder model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

            # Try to use MPS/CUDA if available for acceleration
            try:
                import torch

                if torch.backends.mps.is_available():
                    # MPS (Apple Silicon GPU) support
                    logger.debug("Cross-encoder using MPS device (Apple Silicon)")
                    self._model.model.to("mps")
                elif torch.cuda.is_available():
                    # CUDA (NVIDIA GPU) support
                    logger.debug("Cross-encoder using CUDA device")
                    self._model.model.to("cuda")
                else:
                    logger.debug("Cross-encoder using CPU (no GPU available)")
            except Exception as e:
                # Non-fatal: fall back to CPU if device detection fails
                logger.debug(f"GPU detection failed, using CPU: {e}")

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by cross-encoder relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return only top-k results (None = return all)

        Returns:
            List of (original_index, score) tuples, sorted by score descending

        Example:
            >>> reranker = CrossEncoderReranker()
            >>> docs = ["def parse_file(path):", "class Config:", "import os"]
            >>> results = reranker.rerank("parse file", docs, top_k=2)
            >>> # Returns: [(0, 0.95), (2, 0.45)] - indices into docs
        """
        if not documents:
            logger.debug("No documents to rerank")
            return []

        # Ensure model is loaded
        model = self._ensure_model()

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Score all pairs
        # scores[i] = relevance score for documents[i]
        scores = model.predict(pairs)

        # Create (index, score) pairs and sort by score descending
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top-k limit if specified
        if top_k:
            indexed_scores = indexed_scores[:top_k]

        logger.debug(
            f"Cross-encoder reranked {len(documents)} documents, "
            f"top score: {indexed_scores[0][1]:.3f}, "
            f"bottom score: {indexed_scores[-1][1]:.3f}"
        )

        return indexed_scores

    @property
    def is_available(self) -> bool:
        """Check if cross-encoder can be loaded.

        Returns:
            True if sentence-transformers is available, False otherwise
        """
        try:
            # Import check (don't actually load model yet)
            from sentence_transformers import CrossEncoder  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def model_name(self) -> str:
        """Get the model name used by this reranker."""
        return self._model_name

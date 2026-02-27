"""BM25 backend for keyword-based search using rank_bm25.

This module provides BM25 (Best Matching 25) keyword search as a complement
to vector similarity search. BM25 is a probabilistic retrieval function that
ranks documents based on term frequency and inverse document frequency.

Key features:
- Fast keyword-based search (O(n) per query)
- Indexes both code content and metadata (function names, class names, file paths)
- Persistent storage with pickle/orjson serialization
- Graceful fallback if index is missing or corrupted

Use cases:
- Exact keyword matching (e.g., "DatabaseConnection")
- API/function name searches (e.g., "search_code")
- Hybrid search combined with vector similarity
"""

import pickle  # nosec B403 - BM25 index is local-only, not from untrusted sources
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from .exceptions import DatabaseError


class BM25Backend:
    """BM25 keyword search backend for code chunks.

    Provides fast keyword-based search using the BM25Okapi algorithm.
    Index includes code content and metadata for comprehensive matching.

    Example:
        backend = BM25Backend()
        chunks = load_chunks_from_database()
        backend.build_index(chunks)
        backend.save(index_path)

        # Later
        backend.load(index_path)
        results = backend.search("parse file chunks", limit=10)
        # Returns: [(chunk_id, score), ...]
    """

    def __init__(self) -> None:
        """Initialize BM25 backend."""
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_texts: list[str] = []  # For debugging/inspection

    def build_index(
        self,
        chunks: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build BM25 index from code chunks.

        Tokenizes each chunk by combining content and metadata (function name,
        class name, file path) to enable matching on both code and structure.

        Args:
            chunks: List of dicts with chunk data:
                - chunk_id (str): Unique identifier
                - content (str): Code content
                - name (str): Function/class name (optional)
                - file_path (str): Source file path
                - chunk_type (str): function, class, method, etc.
            progress_callback: Optional callable invoked periodically with
                (current, total) during tokenization.  Called every 500 chunks
                and once more at completion.

        Raises:
            DatabaseError: If index building fails
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            self._bm25 = None
            self._chunk_ids = []
            self._corpus_texts = []
            return

        try:
            # Build tokenized corpus for BM25
            corpus = []
            chunk_ids = []

            for idx, chunk in enumerate(chunks):
                # Combine content and metadata for comprehensive search
                # This allows matching on function names, file paths, and code content
                text_parts = []

                # Add code content (primary)
                content = chunk.get("content", "")
                if content:
                    text_parts.append(content)

                # Add metadata (secondary) - helps with keyword searches
                name = chunk.get("name", "")
                if name:
                    # Repeat name multiple times to boost its weight in BM25 scoring
                    text_parts.append(name)
                    text_parts.append(name)

                file_path = chunk.get("file_path", "")
                if file_path:
                    # Include file path components for path-based searches
                    text_parts.append(str(file_path))

                chunk_type = chunk.get("chunk_type", "")
                if chunk_type:
                    text_parts.append(chunk_type)

                # Combine all text parts
                combined_text = " ".join(text_parts)

                # Tokenize: lowercase + split on whitespace and special chars
                # BM25Okapi expects pre-tokenized corpus (list of token lists)
                tokens = self._tokenize(combined_text)

                corpus.append(tokens)
                chunk_ids.append(chunk["chunk_id"])

                # Progress callback for large indices (every 500 chunks)
                if progress_callback and (idx + 1) % 500 == 0:
                    progress_callback(idx + 1, len(chunks))

            # Final progress update (ensure 100% is reported)
            if progress_callback:
                progress_callback(len(chunks), len(chunks))

            # Build BM25 index
            self._bm25 = BM25Okapi(corpus)
            self._chunk_ids = chunk_ids
            self._corpus_texts = [" ".join(tokens) for tokens in corpus]

            logger.info(
                f"Built BM25 index with {len(corpus)} chunks "
                f"(avg {sum(len(c) for c in corpus) // len(corpus)} tokens per chunk)"
            )

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise DatabaseError(f"BM25 index building failed: {e}") from e

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search using BM25 keyword matching.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending.
            Scores are BM25 relevance scores (higher = better match).

        Raises:
            DatabaseError: If search fails
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built or loaded, returning empty results")
            return []

        if not query.strip():
            return []

        try:
            # Tokenize query using same tokenization as corpus
            query_tokens = self._tokenize(query)

            # Get BM25 scores for all documents
            scores = self._bm25.get_scores(query_tokens)

            # Get top N results
            # BM25 scores are not normalized, but higher = better
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:limit]

            # Return (chunk_id, score) tuples
            results = [
                (self._chunk_ids[idx], float(scores[idx])) for idx in top_indices
            ]

            # Filter out zero scores (no match)
            results = [(cid, score) for cid, score in results if score > 0.0]

            logger.debug(
                f"BM25 search for '{query}' returned {len(results)} results "
                f"(top score: {results[0][1]:.3f})"
                if results
                else "BM25 search returned no results"
            )

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise DatabaseError(f"BM25 search failed: {e}") from e

    def save(self, path: Path) -> None:
        """Save BM25 index to disk using pickle.

        Args:
            path: Path to save the index (e.g., .mcp-vector-search/bm25_index.pkl)

        Raises:
            DatabaseError: If saving fails
        """
        if self._bm25 is None:
            logger.warning("No BM25 index to save")
            return

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save index data as dict
            index_data = {
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "corpus_texts": self._corpus_texts,
            }

            with open(path, "wb") as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved BM25 index to {path} ({len(self._chunk_ids)} chunks)")

        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise DatabaseError(f"BM25 index save failed: {e}") from e

    def load(self, path: Path) -> None:
        """Load BM25 index from disk.

        Args:
            path: Path to the saved index

        Raises:
            DatabaseError: If loading fails
        """
        if not path.exists():
            logger.warning(f"BM25 index not found at {path}")
            self._bm25 = None
            self._chunk_ids = []
            self._corpus_texts = []
            return

        try:
            with open(path, "rb") as f:
                index_data = pickle.load(f)  # nosec B301 - local BM25 index only

            self._bm25 = index_data["bm25"]
            self._chunk_ids = index_data["chunk_ids"]
            self._corpus_texts = index_data.get("corpus_texts", [])

            logger.info(
                f"Loaded BM25 index from {path} ({len(self._chunk_ids)} chunks)"
            )

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            # Non-fatal: set to None and continue (allows graceful fallback)
            self._bm25 = None
            self._chunk_ids = []
            self._corpus_texts = []
            logger.warning("BM25 search will be unavailable (index load failed)")

    def is_built(self) -> bool:
        """Check if BM25 index is built and ready.

        Returns:
            True if index is ready, False otherwise
        """
        return self._bm25 is not None and len(self._chunk_ids) > 0

    def get_stats(self) -> dict[str, Any]:
        """Get BM25 index statistics.

        Returns:
            Dictionary with index stats
        """
        if not self.is_built():
            return {"built": False, "chunk_count": 0}

        # Calculate average document length
        avg_doc_len = 0.0
        if self._bm25 and hasattr(self._bm25, "avgdl"):
            avg_doc_len = float(self._bm25.avgdl)

        return {
            "built": True,
            "chunk_count": len(self._chunk_ids),
            "avg_doc_length": avg_doc_len,
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Simple tokenization strategy:
        - Lowercase
        - Split on whitespace and common separators
        - Remove empty tokens

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import re

        # Lowercase
        text = text.lower()

        # Split on whitespace, punctuation, and special chars
        # Keep alphanumeric sequences and underscores (common in code)
        tokens = re.findall(r"\w+", text)

        # Filter empty tokens
        tokens = [t for t in tokens if t]

        return tokens

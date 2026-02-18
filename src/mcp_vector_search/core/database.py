"""Database abstraction for MCP Vector Search.

This module defines the abstract VectorDatabase interface. All vector database
backends (e.g., LanceDB) must implement this interface.

The ChromaDB implementation has been removed in favor of LanceDB as the primary
and only backend. See lancedb_backend.py for the LanceDB implementation.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import CodeChunk, IndexStats, SearchResult


def _detect_optimal_cache_size() -> int:
    """Detect optimal cache size based on available RAM.

    Returns:
        Optimal cache size for embedding/search results:
        - 10000 for 64GB+ RAM (M4 Max/Ultra, high-end workstations)
        - 5000 for 32GB RAM (M4 Pro, mid-tier systems)
        - 1000 for 16GB RAM (M4 base, standard systems)
        - 100 for <16GB RAM or detection failure (safe default)

    Environment Variables:
        MCP_VECTOR_SEARCH_CACHE_SIZE: Override auto-detection
    """
    env_size = os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE")
    if env_size:
        return int(env_size)

    try:
        import subprocess

        result = subprocess.run(  # nosec B607 - safe system call for RAM detection
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            total_ram_gb = int(result.stdout.strip()) / (1024**3)

            if total_ram_gb >= 64:
                return 10000  # 64GB+ RAM
            elif total_ram_gb >= 32:
                return 5000  # 32GB RAM
            elif total_ram_gb >= 16:
                return 1000  # 16GB RAM
            else:
                return 100  # <16GB RAM
    except Exception:
        pass

    return 100  # Safe default


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts."""
        ...


class VectorDatabase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection and collections."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        ...

    @abstractmethod
    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters to apply
            similarity_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        ...

    @abstractmethod
    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Number of deleted chunks
        """
        ...

    @abstractmethod
    async def get_stats(self) -> IndexStats:
        """Get database statistics.

        Returns:
            Index statistics
        """
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset the database (delete all data)."""
        ...

    @abstractmethod
    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all code chunks with metadata
        """
        ...

    def iter_chunks_batched(
        self,
        batch_size: int = 10000,
        file_path: str | None = None,
        language: str | None = None,
    ) -> Any:
        """Stream chunks from database in batches to avoid memory explosion.

        Optional method for databases that support memory-efficient iteration.
        Default implementation raises NotImplementedError.

        Args:
            batch_size: Number of chunks per batch (default 10000)
            file_path: Optional filter by file path
            language: Optional filter by language

        Yields:
            List of CodeChunk objects per batch

        Raises:
            NotImplementedError: If database backend doesn't support batch iteration
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch iteration"
        )

    def get_chunk_count(
        self, file_path: str | None = None, language: str | None = None
    ) -> int:
        """Get total chunk count without loading all data.

        Optional method for databases that support efficient counting.
        Default implementation raises NotImplementedError.

        Args:
            file_path: Optional filter by file path
            language: Optional filter by language

        Returns:
            Total number of chunks matching the filter criteria

        Raises:
            NotImplementedError: If database backend doesn't support counting
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chunk counting"
        )

    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health and integrity.

        Returns:
            True if database is healthy, False otherwise
        """
        ...

    async def __aenter__(self) -> "VectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

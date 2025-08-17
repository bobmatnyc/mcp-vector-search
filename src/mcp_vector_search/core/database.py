"""Database abstraction and ChromaDB implementation for MCP Vector Search."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
    SearchError,
)
from .models import CodeChunk, IndexStats, SearchResult


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: List[str]) -> List[List[float]]:
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
    async def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add code chunks to the database.
        
        Args:
            chunks: List of code chunks to add
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7,
    ) -> List[SearchResult]:
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

    async def __aenter__(self) -> "VectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation of vector database."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "code_search",
    ) -> None:
        """Initialize ChromaDB vector database.
        
        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Create client with new API
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )

            # Create or get collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Semantic code search collection",
                },
            )

            logger.info(f"ChromaDB initialized at {self.persist_directory}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise DatabaseInitializationError(f"ChromaDB initialization failed: {e}") from e

    async def remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks for a specific file.

        Args:
            file_path: Relative path to the file

        Returns:
            Number of chunks removed
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all chunks for this file
            results = self._collection.get(
                where={"file_path": file_path}
            )

            if not results["ids"]:
                return 0

            # Delete the chunks
            self._collection.delete(ids=results["ids"])

            removed_count = len(results["ids"])
            logger.debug(f"Removed {removed_count} chunks for file: {file_path}")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to remove chunks for file {file_path}: {e}")
            return 0

    async def close(self) -> None:
        """Close database connections."""
        if self._client:
            # ChromaDB doesn't require explicit closing
            self._client = None
            self._collection = None
            logger.debug("ChromaDB connections closed")

    async def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Add code chunks to the database."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        if not chunks:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for chunk in chunks:
                # Create searchable text
                searchable_text = self._create_searchable_text(chunk)
                documents.append(searchable_text)

                # Create metadata
                metadata = {
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "function_name": chunk.function_name or "",
                    "class_name": chunk.class_name or "",
                    "docstring": chunk.docstring or "",
                    "complexity_score": chunk.complexity_score,
                }
                metadatas.append(metadata)

                # Use chunk ID
                ids.append(chunk.id)

            # Add to collection
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.debug(f"Added {len(chunks)} chunks to database")

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7,
    ) -> List[SearchResult]:
        """Search for similar code chunks."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Build where clause
            where_clause = self._build_where_clause(filters) if filters else None

            # Perform search
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            search_results = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    similarity = 1.0 - distance

                    if similarity >= similarity_threshold:
                        result = SearchResult(
                            content=doc,
                            file_path=Path(metadata["file_path"]),
                            start_line=metadata["start_line"],
                            end_line=metadata["end_line"],
                            language=metadata["language"],
                            similarity_score=similarity,
                            rank=i + 1,
                            chunk_type=metadata.get("chunk_type", "code"),
                            function_name=metadata.get("function_name") or None,
                            class_name=metadata.get("class_name") or None,
                        )
                        search_results.append(result)

            logger.debug(f"Found {len(search_results)} results for query: {query}")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}") from e

    async def delete_by_file(self, file_path: Path) -> int:
        """Delete all chunks for a specific file."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all chunks for this file
            results = self._collection.get(
                where={"file_path": str(file_path)},
                include=["metadatas"],
            )

            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                count = len(results["ids"])
                logger.debug(f"Deleted {count} chunks for {file_path}")
                return count

            return 0

        except Exception as e:
            logger.error(f"Failed to delete chunks for {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self) -> IndexStats:
        """Get database statistics."""
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get total count
            count = self._collection.count()

            # Get sample for language distribution
            sample_results = self._collection.get(
                limit=min(1000, count) if count > 0 else 0,
                include=["metadatas"],
            )

            languages = {}
            file_types = {}

            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    # Count languages
                    lang = metadata.get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1

                    # Count file types
                    file_path = metadata.get("file_path", "")
                    ext = Path(file_path).suffix or "no_extension"
                    file_types[ext] = file_types.get(ext, 0) + 1

            # Estimate index size (rough approximation)
            index_size_mb = count * 0.001  # Rough estimate

            return IndexStats(
                total_files=len(set(m.get("file_path", "") for m in sample_results.get("metadatas", []))),
                total_chunks=count,
                languages=languages,
                file_types=file_types,
                index_size_mb=index_size_mb,
                last_updated="unknown",  # TODO: Track this
                embedding_model="unknown",  # TODO: Track this
            )

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return IndexStats(
                total_files=0,
                total_chunks=0,
                languages={},
                file_types={},
                index_size_mb=0.0,
                last_updated="error",
                embedding_model="unknown",
            )

    async def reset(self) -> None:
        """Reset the database."""
        if self._client:
            try:
                self._client.reset()
                # Recreate collection
                await self.initialize()
                logger.info("Database reset successfully")
            except Exception as e:
                logger.error(f"Failed to reset database: {e}")
                raise DatabaseError(f"Failed to reset database: {e}") from e

    def _create_searchable_text(self, chunk: CodeChunk) -> str:
        """Create optimized searchable text from code chunk."""
        parts = [chunk.content]

        # Add contextual information
        if chunk.function_name:
            parts.append(f"Function: {chunk.function_name}")

        if chunk.class_name:
            parts.append(f"Class: {chunk.class_name}")

        if chunk.docstring:
            parts.append(f"Documentation: {chunk.docstring}")

        # Add language and file context
        parts.append(f"Language: {chunk.language}")
        parts.append(f"File: {chunk.file_path.name}")

        return "\n".join(parts)

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where = {}

        for key, value in filters.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            elif isinstance(value, str) and value.startswith("!"):
                where[key] = {"$ne": value[1:]}
            else:
                where[key] = value

        return where

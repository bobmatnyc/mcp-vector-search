"""Database abstraction and ChromaDB implementation for MCP Vector Search."""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .connection_pool import ChromaConnectionPool
from .exceptions import (
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
    IndexCorruptionError,
    SearchError,
)
from .models import CodeChunk, IndexStats, SearchResult


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
    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
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
        """Initialize ChromaDB client and collection with corruption recovery."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Check for corruption before initializing
            await self._detect_and_recover_corruption()

            # Create client with new API
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Create or get collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Semantic code search collection",
                },
            )

            logger.debug(f"ChromaDB initialized at {self.persist_directory}")

        except Exception as e:
            # Check if this is a corruption error
            error_msg = str(e).lower()
            if any(indicator in error_msg for indicator in [
                "pickle", "unpickling", "eof", "ran out of input",
                "hnsw", "index", "deserialize", "corrupt"
            ]):
                logger.warning(f"Detected index corruption: {e}")
                # Try to recover
                await self._recover_from_corruption()
                # Retry initialization
                await self.initialize()
            else:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise DatabaseInitializationError(
                    f"ChromaDB initialization failed: {e}"
                ) from e

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
            results = self._collection.get(where={"file_path": file_path})

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

    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
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
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
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
                        strict=False,
                    )
                ):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    # For cosine distance, use a more permissive conversion that handles distances > 1.0
                    # Convert to a 0-1 similarity score where lower distances = higher similarity
                    similarity = max(0.0, 1.0 / (1.0 + distance))

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
                total_files=len(
                    {
                        m.get("file_path", "")
                        for m in sample_results.get("metadatas", [])
                    }
                ),
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

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
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

    async def _detect_and_recover_corruption(self) -> None:
        """Detect and recover from index corruption proactively."""
        # Check for common corruption indicators in ChromaDB files
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        
        # If database doesn't exist yet, nothing to check
        if not chroma_db_path.exists():
            return
            
        # Check for HNSW index files that might be corrupted
        collection_path = self.persist_directory / "chroma-collections.parquet"
        index_path = self.persist_directory / "index"
        
        if index_path.exists():
            # Look for pickle files in the index
            pickle_files = list(index_path.glob("**/*.pkl"))
            pickle_files.extend(list(index_path.glob("**/*.pickle")))
            
            for pickle_file in pickle_files:
                try:
                    # Try to read the pickle file to detect corruption
                    import pickle
                    with open(pickle_file, 'rb') as f:
                        pickle.load(f)
                except (EOFError, pickle.UnpicklingError, Exception) as e:
                    logger.warning(f"Corrupted index file detected: {pickle_file} - {e}")
                    await self._recover_from_corruption()
                    return

    async def _recover_from_corruption(self) -> None:
        """Recover from index corruption by rebuilding the index."""
        logger.info("Attempting to recover from index corruption...")
        
        # Create backup directory
        backup_dir = self.persist_directory.parent / f"{self.persist_directory.name}_backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Backup current state (in case we need it)
        import time
        timestamp = int(time.time())
        backup_path = backup_dir / f"backup_{timestamp}"
        
        if self.persist_directory.exists():
            try:
                shutil.copytree(self.persist_directory, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
        
        # Clear the corrupted index
        if self.persist_directory.exists():
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleared corrupted index at {self.persist_directory}")
            except Exception as e:
                logger.error(f"Failed to clear corrupted index: {e}")
                raise IndexCorruptionError(
                    f"Could not clear corrupted index: {e}"
                ) from e
        
        # Recreate the directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info("Index directory recreated. Please re-index your codebase.")

    async def health_check(self) -> bool:
        """Check database health and integrity.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            # First check if client is initialized
            if not self._client or not self._collection:
                logger.warning("Database not initialized")
                return False
            
            # Try a simple operation to test the connection
            try:
                # Attempt to get count - this will fail if index is corrupted
                count = self._collection.count()
                logger.debug(f"Health check passed: {count} chunks in database")
                
                # Try a minimal query to ensure search works
                self._collection.query(
                    query_texts=["test"],
                    n_results=1,
                    include=["metadatas"]
                )
                
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(indicator in error_msg for indicator in [
                    "pickle", "unpickling", "eof", "ran out of input",
                    "hnsw", "index", "deserialize", "corrupt"
                ]):
                    logger.error(f"Index corruption detected during health check: {e}")
                    return False
                else:
                    # Some other error
                    logger.warning(f"Health check failed: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False


class PooledChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation with connection pooling for improved performance."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "code_search",
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: float = 300.0,
        max_connection_age: float = 3600.0,
    ) -> None:
        """Initialize pooled ChromaDB vector database.

        Args:
            persist_directory: Directory to persist database
            embedding_function: Function to generate embeddings
            collection_name: Name of the collection
            max_connections: Maximum number of connections in pool
            min_connections: Minimum number of connections to maintain
            max_idle_time: Maximum time a connection can be idle (seconds)
            max_connection_age: Maximum age of a connection (seconds)
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name

        self._pool = ChromaConnectionPool(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
            max_connections=max_connections,
            min_connections=min_connections,
            max_idle_time=max_idle_time,
            max_connection_age=max_connection_age,
        )

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        await self._pool.initialize()
        logger.debug(f"Pooled ChromaDB initialized at {self.persist_directory}")

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
        logger.debug("Pooled ChromaDB connections closed")

    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Add code chunks to the database using pooled connection."""
        if not chunks:
            return

        # Ensure pool is initialized
        if not self._pool._initialized:
            await self._pool.initialize()

        try:
            async with self._pool.get_connection() as conn:
                # Prepare data for ChromaDB
                documents = []
                metadatas = []
                ids = []

                for chunk in chunks:
                    documents.append(chunk.content)
                    metadatas.append(
                        {
                            "file_path": str(chunk.file_path),
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "language": chunk.language,
                            "chunk_type": chunk.chunk_type,
                            "function_name": chunk.function_name or "",
                            "class_name": chunk.class_name or "",
                        }
                    )
                    ids.append(chunk.id)

                # Add to collection
                conn.collection.add(documents=documents, metadatas=metadatas, ids=ids)

                logger.debug(f"Added {len(chunks)} chunks to database")

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise DocumentAdditionError(f"Failed to add chunks: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar code chunks using pooled connection."""
        # Ensure pool is initialized
        if not self._pool._initialized:
            await self._pool.initialize()

        try:
            async with self._pool.get_connection() as conn:
                # Build where clause
                where_clause = self._build_where_clause(filters) if filters else None

                # Perform search
                results = conn.collection.query(
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
                            strict=False,
                        )
                    ):
                        # Convert distance to similarity (ChromaDB uses cosine distance)
                        # For cosine distance, use a more permissive conversion that handles distances > 1.0
                        # Convert to a 0-1 similarity score where lower distances = higher similarity
                        similarity = max(0.0, 1.0 / (1.0 + distance))

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
        """Delete all chunks for a specific file using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                # Get all chunks for this file
                results = conn.collection.get(
                    where={"file_path": str(file_path)}, include=["metadatas"]
                )

                if not results["ids"]:
                    return 0

                # Delete the chunks
                conn.collection.delete(ids=results["ids"])

                deleted_count = len(results["ids"])
                logger.debug(f"Deleted {deleted_count} chunks for file: {file_path}")
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_path}: {e}")
            raise DatabaseError(f"Failed to delete chunks: {e}") from e

    async def get_stats(self) -> IndexStats:
        """Get database statistics using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                # Get total count
                count = conn.collection.count()

                # Get all metadata to analyze
                results = conn.collection.get(include=["metadatas"])

                # Analyze languages and files
                languages = set()
                files = set()

                for metadata in results["metadatas"]:
                    if "language" in metadata:
                        languages.add(metadata["language"])
                    if "file_path" in metadata:
                        files.add(metadata["file_path"])

                # Count languages and file types
                language_counts = {}
                file_type_counts = {}

                for metadata in results["metadatas"]:
                    # Count languages
                    lang = metadata.get("language", "unknown")
                    language_counts[lang] = language_counts.get(lang, 0) + 1

                    # Count file types
                    file_path = metadata.get("file_path", "")
                    if file_path:
                        ext = Path(file_path).suffix or "no_extension"
                        file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

                # Estimate index size (rough approximation)
                index_size_mb = count * 0.001  # Rough estimate

                return IndexStats(
                    total_chunks=count,
                    total_files=len(files),
                    languages=language_counts,
                    file_types=file_type_counts,
                    index_size_mb=index_size_mb,
                    last_updated="unknown",  # ChromaDB doesn't track this
                    embedding_model="unknown"  # TODO: Track this in metadata
                )

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise DatabaseError(f"Failed to get stats: {e}") from e

    async def remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks for a specific file using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                # Get all chunks for this file
                results = conn.collection.get(where={"file_path": file_path})

                if not results["ids"]:
                    return 0

                # Delete the chunks
                conn.collection.delete(ids=results["ids"])

                return len(results["ids"])

        except Exception as e:
            logger.error(f"Failed to remove chunks for file {file_path}: {e}")
            return 0

    async def reset(self) -> None:
        """Reset the database using pooled connection."""
        try:
            async with self._pool.get_connection() as conn:
                conn.client.reset()
                # Reinitialize the pool after reset
                await self._pool.close()
                await self._pool.initialize()
                logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise DatabaseError(f"Failed to reset database: {e}") from e

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Build ChromaDB where clause from filters."""
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if key == "language" and value:
                conditions.append({"language": {"$eq": value}})
            elif key == "file_path" and value:
                if isinstance(value, list):
                    conditions.append({"file_path": {"$in": [str(p) for p in value]}})
                else:
                    conditions.append({"file_path": {"$eq": str(value)}})
            elif key == "chunk_type" and value:
                conditions.append({"chunk_type": {"$eq": value}})

        if not conditions:
            return None
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return conditions[0]

    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return self._pool.get_stats()

    async def health_check(self) -> bool:
        """Perform a health check on the database and connection pool."""
        try:
            # Check pool health
            pool_healthy = await self._pool.health_check()
            if not pool_healthy:
                return False
                
            # Try a simple query to verify database integrity
            try:
                async with self._pool.get_connection() as conn:
                    # Test basic operations
                    conn.collection.count()
                    conn.collection.query(
                        query_texts=["test"],
                        n_results=1,
                        include=["metadatas"]
                    )
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if any(indicator in error_msg for indicator in [
                    "pickle", "unpickling", "eof", "ran out of input",
                    "hnsw", "index", "deserialize", "corrupt"
                ]):
                    logger.error(f"Index corruption detected: {e}")
                    # Attempt recovery
                    await self._recover_from_corruption()
                    return False
                else:
                    logger.warning(f"Health check failed: {e}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def _recover_from_corruption(self) -> None:
        """Recover from index corruption by rebuilding the index."""
        logger.info("Attempting to recover from index corruption...")
        
        # Close the pool first
        await self._pool.close()
        
        # Create backup directory
        backup_dir = self.persist_directory.parent / f"{self.persist_directory.name}_backup"
        backup_dir.mkdir(exist_ok=True)
        
        # Backup current state
        import time
        timestamp = int(time.time())
        backup_path = backup_dir / f"backup_{timestamp}"
        
        if self.persist_directory.exists():
            try:
                shutil.copytree(self.persist_directory, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
        
        # Clear the corrupted index
        if self.persist_directory.exists():
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleared corrupted index at {self.persist_directory}")
            except Exception as e:
                logger.error(f"Failed to clear corrupted index: {e}")
                raise IndexCorruptionError(
                    f"Could not clear corrupted index: {e}"
                ) from e
        
        # Recreate the directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Reinitialize the pool
        await self._pool.initialize()
        logger.info("Index recovered. Please re-index your codebase.")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

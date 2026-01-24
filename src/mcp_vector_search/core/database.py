"""Database abstraction and ChromaDB implementation for MCP Vector Search."""

import asyncio
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from ..config.defaults import get_model_dimensions
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
        self._recovery_attempted = False  # Guard against infinite recursion

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection with corruption recovery."""
        try:
            import chromadb

            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # LAYER 1: Check for corruption before initializing (SQLite + HNSW checks)
            await self._detect_and_recover_corruption()

            # LAYER 2: Wrap ChromaDB initialization with Rust panic detection
            try:
                # Create client with new API
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )

                # Configure SQLite busy_timeout to prevent indefinite waits on locked database
                try:
                    chroma_db_path = self.persist_directory / "chroma.sqlite3"
                    if chroma_db_path.exists():
                        import sqlite3

                        temp_conn = sqlite3.connect(str(chroma_db_path))
                        temp_conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
                        temp_conn.close()
                        logger.debug("Configured SQLite busy_timeout=30000ms")
                except Exception as e:
                    logger.warning(f"Failed to configure SQLite busy_timeout: {e}")

                # Create or get collection with optimized HNSW parameters
                # M=32, ef_construction=400, ef_search=75 provide better search quality
                # at the cost of slightly slower indexing (acceptable trade-off for code search)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={
                        "description": "Semantic code search collection",
                        "hnsw:space": "cosine",  # Cosine similarity for semantic search
                        "hnsw:construction_ef": 400,  # Higher construction quality (default: 200)
                        "hnsw:M": 32,  # More connections per node (default: 16)
                        "hnsw:search_ef": 75,  # Better search recall (default: 10)
                    },
                )

                # Reset recovery flag on successful initialization
                self._recovery_attempted = False

                # Check for dimension mismatch (migration detection)
                await self._check_dimension_compatibility()

                logger.debug(f"ChromaDB initialized at {self.persist_directory}")

            except BaseException as init_error:
                # Re-raise system exceptions we should never catch
                if isinstance(
                    init_error, KeyboardInterrupt | SystemExit | GeneratorExit
                ):
                    raise

                # LAYER 2: Detect Rust panic patterns during initialization
                error_msg = str(init_error).lower()

                # Rust panic patterns (common ChromaDB Rust panics)
                rust_panic_patterns = [
                    "range start index",
                    "out of range",
                    "panic",
                    "thread panicked",
                    "slice of length",
                    "index out of bounds",
                ]

                if any(pattern in error_msg for pattern in rust_panic_patterns):
                    logger.warning(
                        f"Rust panic detected during ChromaDB initialization: {init_error}"
                    )
                    logger.info(
                        "Attempting automatic recovery from database corruption..."
                    )
                    await self._recover_from_corruption()

                    # Retry initialization ONCE after recovery
                    try:
                        logger.info(
                            "Retrying ChromaDB initialization after recovery..."
                        )
                        self._client = chromadb.PersistentClient(
                            path=str(self.persist_directory),
                            settings=chromadb.Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                            ),
                        )

                        self._collection = self._client.get_or_create_collection(
                            name=self.collection_name,
                            embedding_function=self.embedding_function,
                            metadata={
                                "description": "Semantic code search collection",
                                "hnsw:space": "cosine",
                                "hnsw:construction_ef": 400,
                                "hnsw:M": 32,
                                "hnsw:search_ef": 75,
                            },
                        )

                        logger.info("ChromaDB successfully initialized after recovery")

                    except BaseException as retry_error:
                        # Re-raise system exceptions
                        if isinstance(
                            retry_error, KeyboardInterrupt | SystemExit | GeneratorExit
                        ):
                            raise

                        logger.error(
                            f"Failed to recover from database corruption: {retry_error}"
                        )
                        # Mark recovery as attempted to prevent infinite loops
                        self._recovery_attempted = True
                        raise DatabaseError(
                            f"Failed to recover from database corruption. "
                            f"Please run 'mcp-vector-search reset index' to clear the database. "
                            f"Error: {retry_error}"
                        ) from retry_error
                else:
                    # Not a Rust panic, re-raise original exception
                    raise

        except (DatabaseError, DatabaseInitializationError):
            # Re-raise our own errors without re-processing
            raise
        except Exception as e:
            # Check if this is a corruption error (legacy detection for backward compatibility)
            error_msg = str(e).lower()
            corruption_indicators = [
                "pickle",
                "unpickling",
                "eof",
                "ran out of input",
                "hnsw",
                "index",
                "deserialize",
                "corrupt",
                "file is not a database",  # SQLite corruption
                "database error",  # ChromaDB database errors
            ]

            if any(indicator in error_msg for indicator in corruption_indicators):
                # Prevent infinite recursion - only attempt recovery once
                if self._recovery_attempted:
                    logger.error(
                        f"Recovery already attempted but corruption persists: {e}"
                    )
                    raise DatabaseInitializationError(
                        f"Failed to recover from database corruption. "
                        f"Please run 'mcp-vector-search reset index' to clear and rebuild the database. Error: {e}"
                    ) from e

                logger.warning(f"Detected index corruption: {e}")
                self._recovery_attempted = True

                # Try to recover
                await self._recover_from_corruption()

                # Retry initialization ONE TIME
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

    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database with optional structural metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
                    Example: {"chunk_id_1": {"cognitive_complexity": 5, ...}, ...}
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        if not chunks:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for chunk in chunks:
                # Debug: Check first chunk content
                if len(documents) == 0:
                    import sys

                    has_meta = "Language:" in chunk.content and "File:" in chunk.content
                    print("\n[DATABASE] First chunk content check:", file=sys.stderr)
                    print(f"  Type: {chunk.chunk_type}", file=sys.stderr)
                    print(f"  File: {chunk.file_path.name}", file=sys.stderr)
                    print(
                        f"  Has metadata IN chunk.content: {has_meta}", file=sys.stderr
                    )
                    print(
                        f"  Last 100 chars: {repr(chunk.content[-100:])}",
                        file=sys.stderr,
                    )

                # Store original content directly in documents (no metadata appended)
                # The embedding will be created from the original content
                documents.append(chunk.content)

                # Create metadata (searchable fields as metadata, not appended to content)
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
                    # Hierarchy fields (convert lists to JSON strings for ChromaDB)
                    "chunk_id": chunk.chunk_id or "",
                    "parent_chunk_id": chunk.parent_chunk_id or "",
                    "child_chunk_ids": json.dumps(chunk.child_chunk_ids or []),
                    "chunk_depth": chunk.chunk_depth,
                    # Additional metadata (convert lists/dicts to JSON strings)
                    "decorators": json.dumps(chunk.decorators or []),
                    "parameters": json.dumps(chunk.parameters or []),
                    "return_type": chunk.return_type or "",
                    "type_annotations": json.dumps(chunk.type_annotations or {}),
                    # Monorepo support
                    "subproject_name": chunk.subproject_name or "",
                    "subproject_path": chunk.subproject_path or "",
                }

                # Merge structural metrics if provided
                if metrics and chunk.chunk_id and chunk.chunk_id in metrics:
                    chunk_metrics = metrics[chunk.chunk_id]
                    metadata.update(chunk_metrics)

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
                        # Document contains the original content (no metadata appended)
                        # Parse code smells from JSON if present
                        code_smells = []
                        if "code_smells" in metadata:
                            try:
                                code_smells = json.loads(metadata["code_smells"])
                            except (json.JSONDecodeError, TypeError):
                                code_smells = []

                        # Calculate quality score from metrics (0-100 scale)
                        quality_score = None
                        if (
                            "cognitive_complexity" in metadata
                            and "smell_count" in metadata
                        ):
                            # Simple quality score: penalize complexity and smells
                            complexity = metadata["cognitive_complexity"]
                            smells = metadata["smell_count"]

                            # Start with 100, penalize for complexity and smells
                            score = 100
                            # Complexity penalty: -2 points per complexity unit
                            score -= min(50, complexity * 2)
                            # Smell penalty: -10 points per smell
                            score -= min(30, smells * 10)

                            quality_score = max(0, score)

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
                            # Quality metrics from structural analysis
                            cognitive_complexity=metadata.get("cognitive_complexity"),
                            cyclomatic_complexity=metadata.get("cyclomatic_complexity"),
                            max_nesting_depth=metadata.get("max_nesting_depth"),
                            parameter_count=metadata.get("parameter_count"),
                            lines_of_code=metadata.get("lines_of_code"),
                            complexity_grade=metadata.get("complexity_grade"),
                            code_smells=code_smells,
                            smell_count=metadata.get("smell_count"),
                            quality_score=quality_score,
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

    async def get_stats(self, skip_stats: bool = False) -> IndexStats:
        """Get database statistics with optimized chunked queries.

        Args:
            skip_stats: If True, skip detailed statistics collection to avoid potential crashes
                       on large databases (automatically enabled for databases >500MB)
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # SAFETY CHECK: Detect large databases before calling count()
            # ChromaDB's Rust backend can segfault on large databases (>500MB)
            chroma_db_path = self.persist_directory / "chroma.sqlite3"
            db_size_mb = 0.0
            db_size_bytes = 0

            if chroma_db_path.exists():
                db_size_bytes = chroma_db_path.stat().st_size
                db_size_mb = db_size_bytes / (1024 * 1024)

                # Automatically enable safe mode for large databases
                if db_size_mb > 500 and not skip_stats:
                    logger.warning(
                        f"Large database detected ({db_size_mb:.1f} MB). "
                        "Skipping detailed statistics to prevent potential crashes."
                    )
                    skip_stats = True

            # If skip_stats is enabled, return minimal safe stats
            if skip_stats:
                return IndexStats(
                    total_files=0,
                    total_chunks="Large DB (count skipped for safety)",
                    languages={},
                    file_types={},
                    index_size_mb=db_size_mb,
                    last_updated="Skipped (large database)",
                    embedding_model="unknown",
                    database_size_bytes=db_size_bytes,
                )

            # Get total count (fast operation, but can crash on large DBs)
            count = self._collection.count()

            if count == 0:
                return IndexStats(
                    total_files=0,
                    total_chunks=0,
                    languages={},
                    file_types={},
                    index_size_mb=0.0,
                    last_updated="N/A",
                    embedding_model="unknown",
                )

            # Process in chunks to avoid loading everything at once
            batch_size_limit = 1000

            files = set()
            language_counts: dict[str, int] = {}
            file_type_counts: dict[str, int] = {}

            offset = 0
            while offset < count:
                # Fetch batch
                batch_size = min(batch_size_limit, count - offset)
                logger.debug(
                    f"Processing database stats: batch {offset // batch_size_limit + 1}, "
                    f"{offset}-{offset + batch_size} of {count} chunks"
                )

                results = self._collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )

                # Process batch metadata
                for metadata in results.get("metadatas", []):
                    # Language stats
                    lang = metadata.get("language", "unknown")
                    language_counts[lang] = language_counts.get(lang, 0) + 1

                    # File stats
                    file_path = metadata.get("file_path", "")
                    if file_path:
                        files.add(file_path)
                        ext = Path(file_path).suffix or "no_extension"
                        file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

                offset += batch_size

                # Yield to event loop periodically to prevent blocking
                await asyncio.sleep(0)

            # Estimate index size (rough approximation: ~1KB per chunk)
            index_size_mb = count * 0.001

            return IndexStats(
                total_files=len(files),
                total_chunks=count,
                languages=language_counts,
                file_types=file_type_counts,
                index_size_mb=index_size_mb,
                last_updated="unknown",
                embedding_model="unknown",
            )

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            # Return empty stats instead of raising
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

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all code chunks with metadata
        """
        if not self._collection:
            raise DatabaseNotInitializedError("Database not initialized")

        try:
            # Get all documents from collection
            results = self._collection.get(include=["metadatas", "documents"])

            chunks = []
            if results and results.get("ids"):
                for i, _chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    # Document now contains the original content (no metadata appended)
                    content = results["documents"][i]

                    # Parse JSON strings back to lists/dicts
                    child_chunk_ids = metadata.get("child_chunk_ids", "[]")
                    if isinstance(child_chunk_ids, str):
                        child_chunk_ids = json.loads(child_chunk_ids)

                    decorators = metadata.get("decorators", "[]")
                    if isinstance(decorators, str):
                        decorators = json.loads(decorators)

                    parameters = metadata.get("parameters", "[]")
                    if isinstance(parameters, str):
                        parameters = json.loads(parameters)

                    type_annotations = metadata.get("type_annotations", "{}")
                    if isinstance(type_annotations, str):
                        type_annotations = json.loads(type_annotations)

                    chunk = CodeChunk(
                        content=content,
                        file_path=Path(metadata["file_path"]),
                        start_line=metadata["start_line"],
                        end_line=metadata["end_line"],
                        language=metadata["language"],
                        chunk_type=metadata.get("chunk_type", "code"),
                        function_name=metadata.get("function_name"),
                        class_name=metadata.get("class_name"),
                        docstring=metadata.get("docstring"),
                        imports=metadata.get("imports", []),
                        complexity_score=metadata.get("complexity_score", 0.0),
                        chunk_id=metadata.get("chunk_id"),
                        parent_chunk_id=metadata.get("parent_chunk_id"),
                        child_chunk_ids=child_chunk_ids,
                        chunk_depth=metadata.get("chunk_depth", 0),
                        decorators=decorators,
                        parameters=parameters,
                        return_type=metadata.get("return_type"),
                        type_annotations=type_annotations,
                        subproject_name=metadata.get("subproject_name"),
                        subproject_path=metadata.get("subproject_path"),
                    )
                    chunks.append(chunk)

            logger.debug(f"Retrieved {len(chunks)} chunks from database")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

    def _create_searchable_text(self, chunk: CodeChunk) -> str:
        """Create optimized searchable text from code chunk."""
        import sys

        print("WARNING: _create_searchable_text IS BEING CALLED!", file=sys.stderr)
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
        """Build ChromaDB where clause from filters.

        Supports filtering by:
        - language, file_path, chunk_type (standard fields)
        - complexity_grade (A, B, C, D, F)
        - smell_count (0, >0)
        - cognitive_complexity (range queries using $and)

        Args:
            filters: Dictionary of filter criteria

        Returns:
            ChromaDB where clause
        """
        # If filters already contain ChromaDB operators ($and, $or), pass through
        if "$and" in filters or "$or" in filters:
            return filters

        where = {}

        for key, value in filters.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            elif isinstance(value, str) and value.startswith("!"):
                where[key] = {"$ne": value[1:]}
            elif isinstance(value, dict):
                # Support operator queries like {"$gte": 10}
                where[key] = value
            else:
                where[key] = value

        return where

    async def _check_dimension_compatibility(self) -> None:
        """Check for embedding dimension mismatch and warn if re-indexing needed.

        Detects when index was created with different embedding model (different dimensions).
        Logs clear warning to guide users through migration.
        """
        if not self._collection:
            return

        try:
            # Get collection count to check if index exists
            count = self._collection.count()
            if count == 0:
                # Empty index, no compatibility check needed
                return

            # Get embedding function model name
            model_name = getattr(self.embedding_function, "model_name", None)
            if not model_name:
                # Can't determine model, skip check
                return

            # Get expected dimensions for current model
            try:
                expected_dims = get_model_dimensions(model_name)
            except ValueError:
                # Unknown model, can't validate dimensions
                logger.debug(
                    f"Cannot validate dimensions for unknown model: {model_name}"
                )
                return

            # Peek at one embedding to get actual dimensions
            # This is more reliable than checking metadata
            try:
                result = self._collection.peek(limit=1)
                if result and "embeddings" in result and result["embeddings"]:
                    actual_dims = len(result["embeddings"][0])

                    if actual_dims != expected_dims:
                        logger.warning(
                            f"\n"
                            f"╔═══════════════════════════════════════════════════════════════════╗\n"
                            f"║ EMBEDDING DIMENSION MISMATCH DETECTED                             ║\n"
                            f"╠═══════════════════════════════════════════════════════════════════╣\n"
                            f"║ Current model: {model_name:<50}║\n"
                            f"║ Expected dimensions: {expected_dims:<47}║\n"
                            f"║ Index dimensions: {actual_dims:<50}║\n"
                            f"║                                                                   ║\n"
                            f"║ The index was created with a different embedding model.          ║\n"
                            f"║ Re-indexing is required for correct search results.              ║\n"
                            f"║                                                                   ║\n"
                            f"║ To re-index:                                                      ║\n"
                            f"║   mcp-vector-search index --force                                 ║\n"
                            f"║                                                                   ║\n"
                            f"║ Or use legacy model for compatibility:                            ║\n"
                            f"║   export MCP_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2║\n"
                            f"╚═══════════════════════════════════════════════════════════════════╝\n"
                        )
            except Exception as peek_error:
                logger.debug(
                    f"Could not peek embeddings for dimension check: {peek_error}"
                )

        except Exception as e:
            logger.debug(f"Dimension compatibility check failed: {e}")

    async def _detect_and_recover_corruption(self) -> None:
        """Detect and recover from index corruption proactively.

        This method checks for:
        1. SQLite database corruption (LAYER 1: Pre-initialization check)
        2. HNSW pickle file corruption
        3. Metadata/data inconsistencies
        4. File size anomalies
        """
        # LAYER 1: Check SQLite database integrity FIRST (before ChromaDB initialization)
        chroma_db_path = self.persist_directory / "chroma.sqlite3"

        # If database doesn't exist yet, nothing to check
        if not chroma_db_path.exists():
            return

        # SQLite integrity check - catches corruption BEFORE Rust panic
        try:
            import sqlite3

            logger.debug("Running SQLite integrity check...")
            conn = sqlite3.connect(str(chroma_db_path))
            cursor = conn.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]
            conn.close()

            if result != "ok":
                logger.warning(f"SQLite database corruption detected: {result}")
                logger.info("Initiating automatic recovery from database corruption...")
                await self._recover_from_corruption()
                return

            logger.debug("SQLite integrity check passed")

        except sqlite3.Error as e:
            logger.warning(f"SQLite database error during integrity check: {e}")
            logger.info("Initiating automatic recovery from database corruption...")
            await self._recover_from_corruption()
            return

        # Check for HNSW index files that might be corrupted
        index_path = self.persist_directory / "index"

        if index_path.exists():
            # Look for pickle files in the index (HNSW metadata)
            pickle_files = list(index_path.glob("**/*.pkl"))
            pickle_files.extend(list(index_path.glob("**/*.pickle")))
            pickle_files.extend(list(index_path.glob("**/*.bin")))  # Binary HNSW files

            logger.debug(
                f"Checking {len(pickle_files)} HNSW index files for corruption..."
            )

            for pickle_file in pickle_files:
                try:
                    # Check file size - suspiciously small files might be corrupted
                    file_size = pickle_file.stat().st_size
                    if file_size == 0:
                        logger.warning(
                            f"Empty HNSW index file detected: {pickle_file} (0 bytes)"
                        )
                        await self._recover_from_corruption()
                        return

                    # Only validate pickle files (not binary .bin files)
                    if pickle_file.suffix in (".pkl", ".pickle"):
                        # Try to read the pickle file to detect corruption
                        import pickle  # nosec B403 # Trusted internal index files only

                        with open(pickle_file, "rb") as f:
                            data = pickle.load(f)  # nosec B301 # Trusted internal index files only

                            # Additional validation: check if data structure is valid
                            if data is None:
                                logger.warning(
                                    f"HNSW index file contains None data: {pickle_file}"
                                )
                                await self._recover_from_corruption()
                                return

                            # Check for metadata consistency (if it's a dict)
                            if isinstance(data, dict):
                                # Look for known metadata keys that should exist
                                if "space" in data and "dim" in data:
                                    # Validate dimensions are reasonable
                                    if data.get("dim", 0) <= 0:
                                        logger.warning(
                                            f"Invalid dimensions in HNSW index: {pickle_file} (dim={data.get('dim')})"
                                        )
                                        await self._recover_from_corruption()
                                        return

                except (EOFError, pickle.UnpicklingError) as e:
                    logger.warning(f"Pickle corruption detected in {pickle_file}: {e}")
                    await self._recover_from_corruption()
                    return
                except Exception as e:
                    # Check if this is a Rust panic pattern
                    error_msg = str(e).lower()
                    if "range start index" in error_msg and "out of range" in error_msg:
                        logger.warning(
                            f"Rust panic pattern detected in {pickle_file}: {e}"
                        )
                        await self._recover_from_corruption()
                        return
                    else:
                        logger.warning(
                            f"Error reading HNSW index file {pickle_file}: {e}"
                        )
                        # Continue checking other files before deciding to recover
                        continue

            logger.debug("HNSW index files validation passed")

    async def _recover_from_corruption(self) -> None:
        """Recover from index corruption by rebuilding the index.

        This method:
        1. Creates a timestamped backup of the corrupted index
        2. Clears the corrupted index directory
        3. Recreates the directory structure
        4. Logs detailed recovery steps and instructions
        """
        logger.warning("=" * 80)
        logger.warning("INDEX CORRUPTION DETECTED - Initiating recovery...")
        logger.warning("=" * 80)

        # Create backup directory
        backup_dir = (
            self.persist_directory.parent / f"{self.persist_directory.name}_backup"
        )
        backup_dir.mkdir(exist_ok=True)

        # Backup current state (in case we need it for debugging)
        import time

        timestamp = int(time.time())
        backup_path = backup_dir / f"backup_{timestamp}"

        if self.persist_directory.exists():
            try:
                shutil.copytree(self.persist_directory, backup_path)
                logger.info(f"✓ Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"⚠ Could not create backup: {e}")

        # Clear the corrupted index
        if self.persist_directory.exists():
            try:
                # Log what we're about to delete
                total_size = sum(
                    f.stat().st_size
                    for f in self.persist_directory.rglob("*")
                    if f.is_file()
                )
                logger.info(
                    f"Clearing corrupted index ({total_size / 1024 / 1024:.2f} MB)..."
                )

                shutil.rmtree(self.persist_directory)
                logger.info(f"✓ Cleared corrupted index at {self.persist_directory}")
            except Exception as e:
                logger.error(f"✗ Failed to clear corrupted index: {e}")
                raise IndexCorruptionError(
                    f"Could not clear corrupted index: {e}. "
                    f"Please manually delete {self.persist_directory} and try again."
                ) from e

        # Recreate the directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Index directory recreated")

        logger.warning("=" * 80)
        logger.warning("RECOVERY COMPLETE - Next steps:")
        logger.warning("  1. Run 'mcp-vector-search index' to rebuild the index")
        logger.warning(f"  2. Backup saved to: {backup_path}")
        logger.warning("=" * 80)

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
                    query_texts=["test"], n_results=1, include=["metadatas"]
                )

                return True

            except Exception as e:
                error_msg = str(e).lower()
                if any(
                    indicator in error_msg
                    for indicator in [
                        "pickle",
                        "unpickling",
                        "eof",
                        "ran out of input",
                        "hnsw",
                        "index",
                        "deserialize",
                        "corrupt",
                    ]
                ):
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

    async def add_chunks(
        self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
    ) -> None:
        """Add code chunks to the database using pooled connection with optional metrics.

        Args:
            chunks: List of code chunks to add
            metrics: Optional dict mapping chunk IDs to ChunkMetrics.to_metadata() dicts
        """
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
                    # Store original content in documents (no metadata appended)
                    documents.append(chunk.content)

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
                        # Hierarchy fields (convert lists to JSON strings for ChromaDB)
                        "chunk_id": chunk.chunk_id or "",
                        "parent_chunk_id": chunk.parent_chunk_id or "",
                        "child_chunk_ids": json.dumps(chunk.child_chunk_ids or []),
                        "chunk_depth": chunk.chunk_depth,
                        # Additional metadata (convert lists/dicts to JSON strings)
                        "decorators": json.dumps(chunk.decorators or []),
                        "parameters": json.dumps(chunk.parameters or []),
                        "return_type": chunk.return_type or "",
                        "type_annotations": json.dumps(chunk.type_annotations or {}),
                        # Monorepo support
                        "subproject_name": chunk.subproject_name or "",
                        "subproject_path": chunk.subproject_path or "",
                    }

                    # Merge structural metrics if provided
                    if metrics and chunk.chunk_id and chunk.chunk_id in metrics:
                        chunk_metrics = metrics[chunk.chunk_id]
                        metadata.update(chunk_metrics)

                    metadatas.append(metadata)
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
                            # Document contains the original content (no metadata appended)
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
        """Get database statistics with connection pooling and chunked queries."""
        try:
            async with self._pool.get_connection() as conn:
                # Get total count (fast operation)
                count = conn.collection.count()

                if count == 0:
                    return IndexStats(
                        total_files=0,
                        total_chunks=0,
                        languages={},
                        file_types={},
                        index_size_mb=0.0,
                        last_updated="N/A",
                        embedding_model="unknown",
                    )

                # Process in chunks to avoid loading everything at once
                batch_size_limit = 1000

                files = set()
                language_counts: dict[str, int] = {}
                file_type_counts: dict[str, int] = {}

                offset = 0
                while offset < count:
                    # Fetch batch
                    batch_size = min(batch_size_limit, count - offset)
                    logger.debug(
                        f"Processing database stats: batch {offset // batch_size_limit + 1}, "
                        f"{offset}-{offset + batch_size} of {count} chunks"
                    )

                    results = conn.collection.get(
                        include=["metadatas"],
                        limit=batch_size,
                        offset=offset,
                    )

                    # Process batch metadata
                    for metadata in results.get("metadatas", []):
                        # Language stats
                        lang = metadata.get("language", "unknown")
                        language_counts[lang] = language_counts.get(lang, 0) + 1

                        # File stats
                        file_path = metadata.get("file_path", "")
                        if file_path:
                            files.add(file_path)
                            ext = Path(file_path).suffix or "no_extension"
                            file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

                    offset += batch_size

                    # Yield to event loop periodically to prevent blocking
                    await asyncio.sleep(0)

                # Estimate index size (rough approximation: ~1KB per chunk)
                index_size_mb = count * 0.001

                return IndexStats(
                    total_files=len(files),
                    total_chunks=count,
                    languages=language_counts,
                    file_types=file_type_counts,
                    index_size_mb=index_size_mb,
                    last_updated="unknown",
                    embedding_model="unknown",
                )

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            # Return empty stats instead of raising
            return IndexStats(
                total_files=0,
                total_chunks=0,
                languages={},
                file_types={},
                index_size_mb=0.0,
                last_updated="error",
                embedding_model="unknown",
            )

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

    async def get_all_chunks(self) -> list[CodeChunk]:
        """Get all chunks from the database using pooled connection.

        Returns:
            List of all code chunks with metadata
        """
        try:
            async with self._pool.get_connection() as conn:
                # Get all documents from collection
                results = conn.collection.get(include=["metadatas", "documents"])

                chunks = []
                if results and results.get("ids"):
                    for i, _chunk_id in enumerate(results["ids"]):
                        metadata = results["metadatas"][i]
                        content = results["documents"][i]

                        # Parse JSON strings back to lists/dicts
                        child_chunk_ids = metadata.get("child_chunk_ids", "[]")
                        if isinstance(child_chunk_ids, str):
                            child_chunk_ids = json.loads(child_chunk_ids)

                        decorators = metadata.get("decorators", "[]")
                        if isinstance(decorators, str):
                            decorators = json.loads(decorators)

                        parameters = metadata.get("parameters", "[]")
                        if isinstance(parameters, str):
                            parameters = json.loads(parameters)

                        type_annotations = metadata.get("type_annotations", "{}")
                        if isinstance(type_annotations, str):
                            type_annotations = json.loads(type_annotations)

                        chunk = CodeChunk(
                            content=content,
                            file_path=Path(metadata["file_path"]),
                            start_line=metadata["start_line"],
                            end_line=metadata["end_line"],
                            language=metadata["language"],
                            chunk_type=metadata.get("chunk_type", "code"),
                            function_name=metadata.get("function_name"),
                            class_name=metadata.get("class_name"),
                            docstring=metadata.get("docstring"),
                            imports=metadata.get("imports", []),
                            complexity_score=metadata.get("complexity_score", 0.0),
                            chunk_id=metadata.get("chunk_id"),
                            parent_chunk_id=metadata.get("parent_chunk_id"),
                            child_chunk_ids=child_chunk_ids,
                            chunk_depth=metadata.get("chunk_depth", 0),
                            decorators=decorators,
                            parameters=parameters,
                            return_type=metadata.get("return_type"),
                            type_annotations=type_annotations,
                            subproject_name=metadata.get("subproject_name"),
                            subproject_path=metadata.get("subproject_path"),
                        )
                        chunks.append(chunk)

                logger.debug(f"Retrieved {len(chunks)} chunks from database")
                return chunks

        except Exception as e:
            logger.error(f"Failed to get all chunks: {e}")
            raise DatabaseError(f"Failed to get all chunks: {e}") from e

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
                        query_texts=["test"], n_results=1, include=["metadatas"]
                    )
                return True
            except Exception as e:
                error_msg = str(e).lower()
                if any(
                    indicator in error_msg
                    for indicator in [
                        "pickle",
                        "unpickling",
                        "eof",
                        "ran out of input",
                        "hnsw",
                        "index",
                        "deserialize",
                        "corrupt",
                    ]
                ):
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
        backup_dir = (
            self.persist_directory.parent / f"{self.persist_directory.name}_backup"
        )
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

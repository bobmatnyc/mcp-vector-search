"""Typed exception hierarchy for mcp-vector-search.

Provides specific exception types so consumers can catch exactly what they
need instead of using a blanket ``except Exception``.

Hierarchy overview::

    MCPVectorSearchError (base)
    |-- SearchError
    |   +-- QueryExpansionError
    |-- IndexingError
    |   +-- ParsingError
    |-- DatabaseError
    |   |-- DatabaseInitializationError
    |   |-- DatabaseNotInitializedError
    |   |-- ConnectionPoolError
    |   |-- DocumentAdditionError
    |   |-- IndexCorruptionError
    |   +-- RustPanicError
    |-- EmbeddingError
    |-- ConfigurationError
    |-- InitializationError
    +-- ProjectError
        |-- ProjectNotFoundError
        +-- ProjectInitializationError

Backward compatibility
----------------------
- ``SearchError`` previously inherited from ``DatabaseError``.  It now
  inherits directly from ``MCPVectorSearchError``.  Code that catches
  ``MCPVectorSearchError`` is unaffected; code that catches
  ``DatabaseError`` expecting to also catch ``SearchError`` must be
  updated.
- ``ParsingError`` previously inherited from ``MCPVectorSearchError``.
  It now inherits from ``IndexingError``.  ``except MCPVectorSearchError``
  still catches it; ``except IndexingError`` now also catches it.
"""

from typing import Any


class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search.

    All public exceptions in this package inherit from this class, so
    ``except MCPVectorSearchError`` catches every domain error.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


# ---------------------------------------------------------------------------
# Search errors
# ---------------------------------------------------------------------------


class SearchError(MCPVectorSearchError):
    """Search operation failed.

    Raised when a semantic or keyword search cannot be completed.
    """

    pass


class QueryExpansionError(SearchError):
    """Query expansion failed during search preprocessing.

    Raised when the query expansion / synonym generation step fails.
    The original query can usually still be searched without expansion.
    """

    pass


# ---------------------------------------------------------------------------
# Indexing errors
# ---------------------------------------------------------------------------


class IndexingError(MCPVectorSearchError):
    """Indexing operation failed.

    Base class for errors during file parsing, chunking, or index
    construction.  Named ``IndexingError`` (not ``IndexError``) to
    avoid shadowing the Python built-in ``IndexError``.
    """

    pass


class ParsingError(IndexingError):
    """Code parsing errors.

    Raised when a file cannot be parsed into code chunks (syntax
    errors, unsupported language, encoding issues, etc.).
    """

    pass


# ---------------------------------------------------------------------------
# Database / storage errors
# ---------------------------------------------------------------------------


class DatabaseError(MCPVectorSearchError):
    """Database-related errors."""

    pass


class DatabaseInitializationError(DatabaseError):
    """Database initialization failed."""

    pass


class DatabaseNotInitializedError(DatabaseError):
    """Operation attempted on uninitialized database."""

    pass


class ConnectionPoolError(DatabaseError):
    """Connection pool operation failed."""

    pass


class DocumentAdditionError(DatabaseError):
    """Failed to add documents to database."""

    pass


class IndexCorruptionError(DatabaseError):
    """Index corruption detected."""

    pass


class RustPanicError(DatabaseError):
    """ChromaDB Rust bindings panic detected.

    This error occurs when ChromaDB's Rust bindings encounter
    HNSW index metadata inconsistencies, typically manifesting as:
    'range start index X out of range for slice of length Y'
    """

    pass


# ---------------------------------------------------------------------------
# Other domain errors
# ---------------------------------------------------------------------------


class EmbeddingError(MCPVectorSearchError):
    """Embedding generation errors."""

    pass


class ConfigurationError(MCPVectorSearchError):
    """Configuration validation errors."""

    pass


class InitializationError(MCPVectorSearchError):
    """General initialization failure.

    Raised when the search engine or indexer cannot be set up
    (missing configuration, incompatible state, etc.).
    """

    pass


class ProjectError(MCPVectorSearchError):
    """Project management errors."""

    pass


class ProjectNotFoundError(ProjectError):
    """Project directory or configuration not found."""

    pass


class ProjectInitializationError(ProjectError):
    """Failed to initialize project."""

    pass

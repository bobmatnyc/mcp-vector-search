"""Typed exception hierarchy for mcp-vector-search.

Hierarchy
---------
MCPVectorSearchError (base)
├── DatabaseError          – LanceDB / storage layer errors
│   ├── DatabaseInitializationError
│   ├── DatabaseNotInitializedError
│   ├── ConnectionPoolError
│   ├── DocumentAdditionError
│   ├── IndexCorruptionError
│   └── RustPanicError
├── SearchError            – search-time failures (query, reranking, etc.)
├── IndexingError          – indexing-time failures (named to avoid shadowing built-in IndexError)
│   └── ParsingError       – code parsing subset of indexing errors
├── EmbeddingError         – embedding generation errors
├── ConfigError            – configuration / validation errors
│   └── ConfigurationError – (legacy alias)
├── InitializationError    – engine / model startup errors
└── ProjectError           – project management errors
    ├── ProjectNotFoundError
    └── ProjectInitializationError

Backward compatibility: ``SearchError`` previously inherited from ``DatabaseError``.
It now inherits directly from ``MCPVectorSearchError`` so callers can distinguish
search-layer failures from storage-layer failures.  Code that catches
``MCPVectorSearchError`` (or ``Exception``) still works unchanged.
"""

from typing import Any

# ── convenience alias so consumers can write ``from mcp_vector_search import MVSError``
MVSError = None  # defined after class below


class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


# Convenience alias
MVSError = MCPVectorSearchError  # type: ignore[assignment]


# ── Database layer ──────────────────────────────────────────────────────


class DatabaseError(MCPVectorSearchError):
    """Database-related errors (LanceDB / storage layer)."""

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
    """Rust bindings panic detected (HNSW index metadata inconsistencies)."""

    pass


# ── Search layer ────────────────────────────────────────────────────────


class SearchError(MCPVectorSearchError):
    """Search operation failed.

    Raised by ``SemanticSearchEngine.search()`` and ``search_similar()``.

    .. versionchanged:: 3.0.33
       Now inherits from ``MCPVectorSearchError`` (was ``DatabaseError``).
    """

    pass


class QueryExpansionError(SearchError):
    """Query expansion / preprocessing failed."""

    pass


# ── Indexing layer ──────────────────────────────────────────────────────


class IndexingError(MCPVectorSearchError):
    """Indexing operation failed.

    Named ``IndexingError`` (not ``IndexError``) to avoid shadowing
    the Python built-in ``IndexError``.

    Raised by ``SemanticIndexer.index_project()`` and ``index_file()``.
    """

    pass


# Alias so callers can write ``mcp_vector_search.exceptions.IndexError`` without
# shadowing the built-in in their own module scope.
IndexError = IndexingError  # noqa: A001


class ParsingError(IndexingError):
    """Code parsing errors (subset of indexing errors)."""

    pass


# ── Embedding layer ─────────────────────────────────────────────────────


class EmbeddingError(MCPVectorSearchError):
    """Embedding generation errors."""

    pass


# ── Configuration layer ─────────────────────────────────────────────────


class ConfigError(MCPVectorSearchError):
    """Configuration / validation errors."""

    pass


# Legacy alias for backward compatibility
ConfigurationError = ConfigError


# ── Initialization layer ────────────────────────────────────────────────


class InitializationError(MCPVectorSearchError):
    """Engine / model startup errors."""

    pass


# ── Project layer ───────────────────────────────────────────────────────


class ProjectError(MCPVectorSearchError):
    """Project management errors."""

    pass


class ProjectNotFoundError(ProjectError):
    """Project directory or configuration not found."""

    pass


class ProjectInitializationError(ProjectError):
    """Failed to initialize project."""

    pass

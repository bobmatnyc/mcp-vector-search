"""Custom exception hierarchy for MCP Vector Search."""

from typing import Any


class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


# ---------------------------------------------------------------------------
# Public typed exception hierarchy (Issue #110)
# ---------------------------------------------------------------------------


class MVSError(MCPVectorSearchError):
    """Base exception for all mcp-vector-search public API errors.

    All exceptions raised by public API methods inherit from this class,
    allowing callers to catch any library error with a single ``except MVSError``.
    """

    pass


class SearchError(MVSError):
    """Search failures (embedding errors, DB errors, malformed queries).

    Raised by:
    - ``SemanticSearchEngine.search()``
    - ``SemanticSearchEngine.search_similar()``
    - ``SemanticSearchEngine.search_by_context()``
    """

    pass


class IndexingError(MVSError):
    """Indexing failures (file parse errors, embedding failures, table errors).

    Raised by:
    - ``SemanticIndexer.index_project()``

    Note: Named ``IndexingError`` (not ``IndexError``) to avoid shadowing the
    Python built-in ``IndexError``.  Import as
    ``from mcp_vector_search.core.exceptions import IndexingError`` or use
    the ``IndexError`` alias exported from the package root.
    """

    pass


# Alias so callers can write ``mcp_vector_search.exceptions.IndexError`` without
# shadowing the built-in in their own module scope.
IndexError = IndexingError  # noqa: A001


class ConfigError(MVSError):
    """Configuration issues (missing deps, bad paths, model mismatch).

    Raised when the project configuration is invalid or unusable.
    """

    pass


class InitializationError(MVSError):
    """Startup failures (model loading, DB connection).

    Raised when the library cannot initialise its runtime dependencies.
    """

    pass


# ---------------------------------------------------------------------------
# Lower-level / internal exceptions (pre-existing hierarchy, preserved as-is)
# ---------------------------------------------------------------------------


class DatabaseError(MVSError):
    """Database/backend failures (LanceDB, Lance, PyArrow).

    Also the base class for all database-level internal errors.
    """

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


class ParsingError(MVSError):
    """Code parsing errors."""

    pass


class EmbeddingError(MVSError):
    """Embedding generation errors."""

    pass


class ConfigurationError(MVSError):
    """Configuration validation errors."""

    pass


class ProjectError(MVSError):
    """Project management errors."""

    pass


class ProjectNotFoundError(ProjectError):
    """Project directory or configuration not found."""

    pass


class ProjectInitializationError(ProjectError):
    """Failed to initialize project."""

    pass

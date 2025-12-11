"""Custom exception hierarchy for MCP Vector Search."""

from typing import Any


class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


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


class SearchError(DatabaseError):
    """Search operation failed."""

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


class ParsingError(MCPVectorSearchError):
    """Code parsing errors."""

    pass


class EmbeddingError(MCPVectorSearchError):
    """Embedding generation errors."""

    pass


class ConfigurationError(MCPVectorSearchError):
    """Configuration validation errors."""

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

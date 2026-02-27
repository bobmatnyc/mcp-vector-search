"""Core functionality for MCP Vector Search."""

from .exceptions import (
    ConfigurationError,
    ConnectionPoolError,
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    DocumentAdditionError,
    EmbeddingError,
    IndexCorruptionError,
    IndexingError,
    InitializationError,
    MCPVectorSearchError,
    ParsingError,
    ProjectError,
    ProjectInitializationError,
    ProjectNotFoundError,
    QueryExpansionError,
    RustPanicError,
    SearchError,
)
from .git import (
    GitError,
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
    GitReferenceError,
)

__all__ = [
    # Exceptions
    "ConfigurationError",
    "ConnectionPoolError",
    "DatabaseError",
    "DatabaseInitializationError",
    "DatabaseNotInitializedError",
    "DocumentAdditionError",
    "EmbeddingError",
    "IndexCorruptionError",
    "IndexingError",
    "InitializationError",
    "MCPVectorSearchError",
    "ParsingError",
    "ProjectError",
    "ProjectInitializationError",
    "ProjectNotFoundError",
    "QueryExpansionError",
    "RustPanicError",
    "SearchError",
    # Git
    "GitError",
    "GitManager",
    "GitNotAvailableError",
    "GitNotRepoError",
    "GitReferenceError",
]

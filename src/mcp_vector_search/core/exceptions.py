"""Custom exception hierarchy for MCP Vector Search."""

from typing import Any, Dict, Optional


class MCPVectorSearchError(Exception):
    """Base exception for MCP Vector Search."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
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


class DocumentAdditionError(DatabaseError):
    """Failed to add documents to database."""
    pass


class SearchError(DatabaseError):
    """Search operation failed."""
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

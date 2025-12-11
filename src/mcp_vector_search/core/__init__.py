"""Core functionality for MCP Vector Search."""

from mcp_vector_search.core.git import (
    GitError,
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
    GitReferenceError,
)

__all__ = [
    "GitError",
    "GitManager",
    "GitNotAvailableError",
    "GitNotRepoError",
    "GitReferenceError",
]

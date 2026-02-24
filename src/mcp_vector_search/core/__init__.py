"""Core functionality for MCP Vector Search."""

from .git import (
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

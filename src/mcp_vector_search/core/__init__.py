"""Core functionality for MCP Vector Search."""

from .git import (
    GitError,
    GitManager,
    GitNotAvailableError,
    GitNotRepoError,
    GitReferenceError,
)
from .models import IndexResult, ProjectStatus

__all__ = [
    "GitError",
    "GitManager",
    "GitNotAvailableError",
    "GitNotRepoError",
    "GitReferenceError",
    "IndexResult",
    "ProjectStatus",
]

"""Utility modules for MCP Vector Search."""

from .gitignore import (
    GitignoreParser,
    GitignorePattern,
    create_gitignore_parser,
    is_path_gitignored,
)
from .gitignore_updater import ensure_gitignore_entry
from .timing import (
    PerformanceProfiler,
    SearchProfiler,
    TimingResult,
    get_global_profiler,
    print_global_report,
    time_async_block,
    time_block,
    time_function,
)
from .version import get_user_agent, get_version_info, get_version_string

__all__ = [
    # Gitignore utilities
    "GitignoreParser",
    "GitignorePattern",
    "create_gitignore_parser",
    "is_path_gitignored",
    "ensure_gitignore_entry",
    # Timing utilities
    "PerformanceProfiler",
    "TimingResult",
    "time_function",
    "time_block",
    "time_async_block",
    "get_global_profiler",
    "print_global_report",
    "SearchProfiler",
    # Version utilities
    "get_version_info",
    "get_version_string",
    "get_user_agent",
]

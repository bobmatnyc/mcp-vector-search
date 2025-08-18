"""Version utilities for MCP Vector Search.

This module provides utilities for accessing and formatting version information.
"""

from typing import Any

from .. import __author__, __build__, __email__, __version__


def get_version_info() -> dict[str, Any]:
    """Get complete version information.

    Returns:
        Dictionary containing version, build, and package metadata
    """
    return {
        "version": __version__,
        "build": __build__,
        "author": __author__,
        "email": __email__,
        "package": "mcp-vector-search",
        "version_string": f"{__version__} (build {__build__})",
    }


def get_version_string(include_build: bool = True) -> str:
    """Get formatted version string.

    Args:
        include_build: Whether to include build number

    Returns:
        Formatted version string
    """
    if include_build:
        return f"{__version__} (build {__build__})"
    return __version__


def get_user_agent() -> str:
    """Get user agent string for HTTP requests.

    Returns:
        User agent string including version
    """
    return f"mcp-vector-search/{__version__}"

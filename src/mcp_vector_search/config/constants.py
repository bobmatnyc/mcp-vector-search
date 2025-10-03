"""Project-wide constants for MCP Vector Search.

This module contains all magic numbers and configuration constants
used throughout the application to improve maintainability and clarity.
"""

# Timeout Constants (in seconds)
SUBPROCESS_INSTALL_TIMEOUT = 120  # Timeout for package installation commands
SUBPROCESS_SHORT_TIMEOUT = 10  # Short timeout for quick commands (version checks, etc.)
SUBPROCESS_MCP_TIMEOUT = 30  # Timeout for MCP server operations
SUBPROCESS_TEST_TIMEOUT = 5  # Timeout for server test operations
CONNECTION_POOL_TIMEOUT = 30.0  # Connection pool acquisition timeout

# Chunking Constants
DEFAULT_CHUNK_SIZE = 50  # Default number of lines per code chunk
TEXT_CHUNK_SIZE = 30  # Number of lines per text/markdown chunk
SEARCH_RESULT_LIMIT = 20  # Default number of search results to return

# Threshold Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # Default similarity threshold for search (0.0-1.0)
HIGH_SIMILARITY_THRESHOLD = 0.75  # Higher threshold for more precise matches

# Cache Constants
DEFAULT_CACHE_SIZE = 256  # Default LRU cache size for file reads

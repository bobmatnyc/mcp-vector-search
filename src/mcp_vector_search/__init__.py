"""MCP Vector Search - CLI-first semantic code search with MCP integration."""

import warnings

# Suppress Pydantic warnings from lancedb embeddings (ColPaliEmbeddings, SigLipEmbeddings)
# These warnings appear when lancedb.embeddings classes use model_name field which conflicts
# with Pydantic's protected "model_" namespace. This is a lancedb issue, not ours.
warnings.filterwarnings("ignore", message=".*has conflict with protected namespace.*")

__version__ = "3.0.33"
__build__ = "283"
__author__ = "Robert Matsuoka"
__email__ = "bob@matsuoka.com"

from .core.exceptions import (  # noqa: E402
    ConfigError,
    ConfigurationError,
    DatabaseError,
    DatabaseInitializationError,
    DatabaseNotInitializedError,
    EmbeddingError,
    IndexCorruptionError,
    IndexingError,
    InitializationError,
    MCPVectorSearchError,
    MVSError,
    ParsingError,
    ProjectError,
    ProjectInitializationError,
    ProjectNotFoundError,
    QueryExpansionError,
    SearchError,
)
from .core.models import ContentChunk  # noqa: E402

__all__ = [
    # Public typed hierarchy (Issue #110)
    "MVSError",
    "MCPVectorSearchError",
    "SearchError",
    "QueryExpansionError",
    "IndexingError",
    "ConfigError",
    "ConfigurationError",
    "InitializationError",
    "DatabaseError",
    "DatabaseInitializationError",
    "DatabaseNotInitializedError",
    "EmbeddingError",
    "IndexCorruptionError",
    "ParsingError",
    "ProjectError",
    "ProjectInitializationError",
    "ProjectNotFoundError",
    # Data types
    "ContentChunk",
    # Version info
    "__version__",
    "__build__",
]

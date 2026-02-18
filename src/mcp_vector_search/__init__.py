"""MCP Vector Search - CLI-first semantic code search with MCP integration."""

import warnings

# Suppress Pydantic warnings from lancedb embeddings (ColPaliEmbeddings, SigLipEmbeddings)
# These warnings appear when lancedb.embeddings classes use model_name field which conflicts
# with Pydantic's protected "model_" namespace. This is a lancedb issue, not ours.
warnings.filterwarnings("ignore", message=".*has conflict with protected namespace.*")

__version__ = "2.2.33"
__build__ = "253"
__author__ = "Robert Matsuoka"
__email__ = "bob@matsuoka.com"

from .core.exceptions import MCPVectorSearchError  # noqa: E402

__all__ = ["MCPVectorSearchError", "__version__", "__build__"]

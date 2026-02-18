"""Embedding dimension compatibility checker (legacy - LanceDB migration).

This module is retained for backward compatibility but is no longer actively used
since the migration to LanceDB. LanceDB handles dimension validation internally.

DEPRECATION NOTICE: This module will be removed in a future version.
"""

from loguru import logger


class DimensionChecker:
    """DEPRECATED: Dimension checking is now handled by LanceDB internally.

    This class is retained for backward compatibility but no longer performs
    active dimension checking since the migration from ChromaDB to LanceDB.
    """

    @staticmethod
    async def _safe_collection_count_by_path(
        persist_directory: str, collection_name: str, timeout: float = 5.0
    ) -> int | None:
        """DEPRECATED: No longer needed with LanceDB.

        Args:
            persist_directory: Database persist directory
            collection_name: Collection name
            timeout: Timeout in seconds

        Returns:
            None (method is deprecated)
        """
        logger.debug(
            "DimensionChecker._safe_collection_count_by_path is deprecated (LanceDB migration)"
        )
        return None

    @staticmethod
    def _log_mismatch_warning(
        model_name: str, expected_dims: int, actual_dims: int
    ) -> None:
        """Log a formatted warning message for dimension mismatch.

        Args:
            model_name: Name of the current embedding model
            expected_dims: Expected embedding dimensions
            actual_dims: Actual dimensions in the index
        """
        logger.warning(
            f"\n"
            f"╔═══════════════════════════════════════════════════════════════════╗\n"
            f"║ EMBEDDING DIMENSION MISMATCH DETECTED                             ║\n"
            f"╠═══════════════════════════════════════════════════════════════════╣\n"
            f"║ Current model: {model_name:<50}║\n"
            f"║ Expected dimensions: {expected_dims:<47}║\n"
            f"║ Index dimensions: {actual_dims:<50}║\n"
            f"║                                                                   ║\n"
            f"║ The index was created with a different embedding model.          ║\n"
            f"║ Re-indexing is required for correct search results.              ║\n"
            f"║                                                                   ║\n"
            f"║ To re-index:                                                      ║\n"
            f"║   mcp-vector-search index --force                                 ║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
        )

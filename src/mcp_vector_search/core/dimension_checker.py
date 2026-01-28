"""Embedding dimension compatibility checker for ChromaDB."""

from typing import Any

from loguru import logger

from ..config.defaults import get_model_dimensions


class DimensionChecker:
    """Checks embedding dimension compatibility for index migrations.

    Detects when an index was created with a different embedding model
    (different dimensions) and provides clear guidance for re-indexing.
    """

    @staticmethod
    async def check_compatibility(collection: Any, embedding_function: Any) -> None:
        """Check for embedding dimension mismatch and warn if re-indexing needed.

        Args:
            collection: ChromaDB collection instance
            embedding_function: Current embedding function
        """
        if not collection:
            return

        try:
            # Get collection count to check if index exists
            count = collection.count()
            if count == 0:
                # Empty index, no compatibility check needed
                return

            # Get embedding function model name
            model_name = getattr(embedding_function, "model_name", None)
            if not model_name:
                # Can't determine model, skip check
                return

            # Get expected dimensions for current model
            try:
                expected_dims = get_model_dimensions(model_name)
            except ValueError:
                # Unknown model, can't validate dimensions
                logger.debug(
                    f"Cannot validate dimensions for unknown model: {model_name}"
                )
                return

            # Peek at one embedding to get actual dimensions
            # This is more reliable than checking metadata
            try:
                result = collection.peek(limit=1)
                if result and "embeddings" in result and result["embeddings"]:
                    actual_dims = len(result["embeddings"][0])

                    if actual_dims != expected_dims:
                        DimensionChecker._log_mismatch_warning(
                            model_name, expected_dims, actual_dims
                        )
            except Exception as peek_error:
                logger.debug(
                    f"Could not peek embeddings for dimension check: {peek_error}"
                )

        except Exception as e:
            logger.debug(f"Dimension compatibility check failed: {e}")

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
            f"║                                                                   ║\n"
            f"║ Or use legacy model for compatibility:                            ║\n"
            f"║   export MCP_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2║\n"
            f"╚═══════════════════════════════════════════════════════════════════╝\n"
        )

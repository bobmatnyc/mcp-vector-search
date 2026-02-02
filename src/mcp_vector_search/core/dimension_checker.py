"""Embedding dimension compatibility checker for ChromaDB."""

import multiprocessing
import signal
from typing import Any

from loguru import logger

from ..config.defaults import get_model_dimensions


def _count_in_subprocess(
    persist_directory: str, collection_name: str, result_queue: multiprocessing.Queue
) -> None:
    """Standalone subprocess function to safely call collection.count().

    This MUST be a module-level function (not a method) so it can be pickled
    by multiprocessing. Opens its own ChromaDB client in the subprocess to
    avoid pickling the collection object.

    This runs in a separate process to isolate potential bus errors.
    If the count operation crashes (bus error), the parent process
    can detect it and handle gracefully.

    Args:
        persist_directory: Path to ChromaDB persist directory
        collection_name: Name of the collection to count
        result_queue: Queue to return the count result
    """
    try:
        # Disable signal handling in subprocess to allow clean termination
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

        # Import ChromaDB in subprocess (fresh instance)
        import chromadb

        # Create a new client pointing to the same database
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get the collection by name
        collection = client.get_collection(collection_name)

        # Attempt the dangerous count operation
        count = collection.count()
        result_queue.put(("success", count))

    except Exception as e:
        # Catch Python exceptions
        result_queue.put(("error", str(e)))


class DimensionChecker:
    """Checks embedding dimension compatibility for index migrations.

    Detects when an index was created with a different embedding model
    (different dimensions) and provides clear guidance for re-indexing.
    """

    @staticmethod
    async def _safe_collection_count(
        collection: Any, timeout: float = 5.0
    ) -> int | None:
        """Safely get collection count with subprocess isolation.

        DEPRECATED: This method signature is kept for backward compatibility
        but now delegates to _safe_collection_count_by_path.

        The collection object cannot be pickled for subprocess communication,
        so this extracts the persist_directory and collection name from the
        collection object and delegates to the path-based method.

        Args:
            collection: ChromaDB collection instance
            timeout: Timeout in seconds for count operation

        Returns:
            Collection count, or None if operation failed/timed out
        """
        try:
            # Extract persist directory and collection name from collection
            # ChromaDB collection objects have _client attribute with _settings
            if hasattr(collection, "_client") and hasattr(
                collection._client, "_settings"
            ):
                persist_dir = str(collection._client._settings.persist_directory)
                collection_name = collection.name

                # Delegate to path-based method
                return await DimensionChecker._safe_collection_count_by_path(
                    persist_dir, collection_name, timeout
                )
            else:
                # For non-ChromaDB collections (tests, mocks), we can't use
                # subprocess isolation since we can't reconstruct the collection.
                # Log at debug level and return None to indicate we can't safely count.
                logger.debug(
                    "Collection object doesn't have _client._settings attribute "
                    "(likely a mock/test object). Cannot use subprocess isolation. "
                    "Returning None to indicate unsafe to count."
                )
                return None

        except Exception as e:
            logger.debug(f"Error in safe collection count: {e}")
            return None

    @staticmethod
    async def _safe_collection_count_by_path(
        persist_directory: str, collection_name: str, timeout: float = 5.0
    ) -> int | None:
        """Safely get collection count with subprocess isolation using database path.

        Uses multiprocessing to isolate the count operation. If ChromaDB's
        Rust backend encounters corrupted HNSW files and triggers a bus error,
        the subprocess dies but the main process survives.

        Args:
            persist_directory: Path to ChromaDB persist directory
            collection_name: Name of the collection
            timeout: Timeout in seconds for count operation

        Returns:
            Collection count, or None if operation failed/timed out
        """
        try:
            # Create a queue for result communication
            ctx = multiprocessing.get_context("spawn")
            result_queue = ctx.Queue()

            # Create subprocess for isolated count operation
            # Pass database path and collection name (both picklable)
            process = ctx.Process(
                target=_count_in_subprocess,
                args=(persist_directory, collection_name, result_queue),
            )

            # Start process
            process.start()

            # Wait for result with timeout
            process.join(timeout=timeout)

            # Check if process completed successfully
            if process.is_alive():
                # Timeout - kill process
                logger.warning(
                    f"Collection count timed out after {timeout}s - possible corruption"
                )
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()  # Force kill if terminate doesn't work
                return None

            # Check exit code
            if process.exitcode != 0:
                # Process crashed (likely bus error)
                logger.warning(
                    f"Collection count subprocess crashed with exit code {process.exitcode} - likely index corruption"
                )
                return None

            # Try to get result from queue
            if not result_queue.empty():
                status, value = result_queue.get_nowait()
                if status == "success":
                    return value
                else:
                    logger.debug(f"Collection count failed: {value}")
                    return None

            # No result available
            logger.debug("Collection count completed but no result returned")
            return None

        except Exception as e:
            logger.debug(f"Error in safe collection count: {e}")
            return None

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
            # SAFETY: Use subprocess-isolated count to prevent bus errors
            # If the index is corrupted, collection.count() can trigger a
            # SIGBUS in ChromaDB's Rust backend. Isolating this in a subprocess
            # allows the main process to survive and trigger recovery.
            count = await DimensionChecker._safe_collection_count(
                collection, timeout=5.0
            )

            if count is None:
                # Count failed - could be pickling issue or index corruption
                # Use debug level to avoid noise during normal chat operation
                logger.debug(
                    "Failed to get collection count - may need to run "
                    "'mcp-vector-search reset index' if issues persist."
                )
                return

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

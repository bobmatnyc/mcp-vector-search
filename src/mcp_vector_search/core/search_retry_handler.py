"""Retry logic and error handling for search operations."""

import asyncio
from typing import Any

from loguru import logger

from .database import VectorDatabase
from .exceptions import RustPanicError, SearchError
from .models import SearchResult


class SearchRetryHandler:
    """Handles search retry logic and error detection."""

    @staticmethod
    def is_rust_panic_error(error: Exception) -> bool:
        """Detect ChromaDB Rust panic errors.

        Args:
            error: Exception to check

        Returns:
            True if this is a Rust panic error
        """
        error_msg = str(error).lower()

        # Check for the specific Rust panic pattern
        # "range start index X out of range for slice of length Y"
        if "range start index" in error_msg and "out of range" in error_msg:
            return True

        # Check for other Rust panic indicators
        rust_panic_patterns = [
            "rust panic",
            "pyo3_runtime.panicexception",
            "thread 'tokio-runtime-worker' panicked",
            "rust/sqlite/src/db.rs",  # Specific to the known ChromaDB issue
        ]

        return any(pattern in error_msg for pattern in rust_panic_patterns)

    @staticmethod
    def is_corruption_error(error: Exception) -> bool:
        """Detect index corruption errors.

        Args:
            error: Exception to check

        Returns:
            True if this is a corruption error
        """
        error_msg = str(error).lower()

        corruption_indicators = [
            "pickle",
            "unpickling",
            "eof",
            "ran out of input",
            "hnsw",
            "deserialize",
            "corrupt",
        ]

        return any(indicator in error_msg for indicator in corruption_indicators)

    async def search_with_retry(
        self,
        database: VectorDatabase,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        threshold: float,
        max_retries: int = 3,
    ) -> list[SearchResult]:
        """Execute search with retry logic and exponential backoff.

        Args:
            database: Vector database instance
            query: Processed search query
            limit: Maximum number of results
            filters: Optional filters
            threshold: Similarity threshold
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            List of search results

        Raises:
            RustPanicError: If Rust panic persists after retries
            SearchError: If search fails for other reasons
        """
        last_error = None
        backoff_delays = [0, 0.1, 0.5]  # Immediate, 100ms, 500ms

        for attempt in range(max_retries):
            try:
                # Add delay for retries (exponential backoff)
                if attempt > 0 and backoff_delays[attempt] > 0:
                    await asyncio.sleep(backoff_delays[attempt])
                    logger.debug(
                        f"Retrying search after {backoff_delays[attempt]}s delay (attempt {attempt + 1}/{max_retries})"
                    )

                # Perform the actual search
                results = await database.search(
                    query=query,
                    limit=limit,
                    filters=filters,
                    similarity_threshold=threshold,
                )

                # Success! If we had retries, log that we recovered
                if attempt > 0:
                    logger.info(
                        f"Search succeeded after {attempt + 1} attempts (recovered from transient error)"
                    )

                return results

            except BaseException as e:
                # Re-raise system exceptions we should never catch
                if isinstance(e, KeyboardInterrupt | SystemExit | GeneratorExit):
                    raise

                last_error = e

                # Check if this is a Rust panic
                if self.is_rust_panic_error(e):
                    logger.warning(
                        f"ChromaDB Rust panic detected (attempt {attempt + 1}/{max_retries}): {e}"
                    )

                    # If this is the last retry, escalate to corruption recovery
                    if attempt == max_retries - 1:
                        logger.error(
                            "Rust panic persisted after all retries - index may be corrupted"
                        )
                        raise RustPanicError(
                            "ChromaDB Rust panic detected. The HNSW index may be corrupted. "
                            "Please run 'mcp-vector-search reset' followed by 'mcp-vector-search index' to rebuild."
                        ) from e

                    # Otherwise, continue to next retry
                    continue

                # Check for general corruption
                elif self.is_corruption_error(e):
                    logger.error(f"Index corruption detected: {e}")
                    raise SearchError(
                        "Index corruption detected. Please run 'mcp-vector-search reset' "
                        "followed by 'mcp-vector-search index' to rebuild."
                    ) from e

                # Some other error - don't retry, just fail
                else:
                    logger.error(f"Search failed: {e}")
                    raise SearchError(f"Search failed: {e}") from e

        # Should never reach here, but just in case
        raise SearchError(
            f"Search failed after {max_retries} retries: {last_error}"
        ) from last_error

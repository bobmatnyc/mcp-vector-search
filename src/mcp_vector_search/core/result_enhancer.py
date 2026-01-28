"""Result enhancement and file caching for semantic search."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger

from ..config.constants import DEFAULT_CACHE_SIZE
from .models import SearchResult


class ResultEnhancer:
    """Handles result enhancement with context and file caching."""

    def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        """Initialize result enhancer.

        Args:
            cache_size: Maximum cache size for file reads
        """
        # File content cache for performance (proper LRU with OrderedDict)
        self._file_cache: OrderedDict[Path, list[str]] = OrderedDict()
        self._cache_maxsize = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    async def read_file_lines_cached(self, file_path: Path) -> list[str]:
        """Read file lines with proper LRU caching for performance.

        Args:
            file_path: Path to the file

        Returns:
            List of file lines

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Check cache - move to end if found (most recently used)
        if file_path in self._file_cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._file_cache.move_to_end(file_path)
            return self._file_cache[file_path]

        self._cache_misses += 1

        # Read file asynchronously
        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()
                lines = content.splitlines(keepends=True)

            # Proper LRU: if cache is full, remove least recently used (first item)
            if len(self._file_cache) >= self._cache_maxsize:
                # Remove least recently used entry (first item in OrderedDict)
                self._file_cache.popitem(last=False)

            # Add to cache (will be at end, most recently used)
            self._file_cache[file_path] = lines
            return lines

        except FileNotFoundError:
            # Cache the miss to avoid repeated failed attempts
            if len(self._file_cache) >= self._cache_maxsize:
                self._file_cache.popitem(last=False)
            self._file_cache[file_path] = []
            raise

    async def enhance_result(
        self, result: SearchResult, include_context: bool
    ) -> SearchResult:
        """Enhance search result with additional information.

        Args:
            result: Original search result
            include_context: Whether to include context lines

        Returns:
            Enhanced search result
        """
        if not include_context:
            return result

        try:
            # Read the source file using cached method
            lines = await self.read_file_lines_cached(result.file_path)

            if not lines:  # File not found or empty
                return result

            # Get context lines before and after
            context_size = 3
            start_idx = max(0, result.start_line - 1 - context_size)
            end_idx = min(len(lines), result.end_line + context_size)

            context_before = [
                line.rstrip() for line in lines[start_idx : result.start_line - 1]
            ]
            context_after = [line.rstrip() for line in lines[result.end_line : end_idx]]

            # Update result with context
            result.context_before = context_before
            result.context_after = context_after

        except FileNotFoundError:
            # File was deleted since indexing - silently skip context
            # This is normal when index is stale; use --force to reindex
            logger.debug(f"File no longer exists (stale index): {result.file_path}")
            result.file_missing = True  # Mark for potential filtering
        except Exception as e:
            logger.warning(f"Failed to get context for {result.file_path}: {e}")

        return result

    async def enhance_with_file_context(
        self, results: list[SearchResult], context_files: list[Path]
    ) -> list[SearchResult]:
        """Enhance results by considering context from specific files.

        Args:
            results: Original search results
            context_files: Files to use for context

        Returns:
            Enhanced search results
        """
        # Read context from files using async I/O
        context_content = []
        for file_path in context_files:
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
                    context_content.append(content)
            except Exception as e:
                logger.warning(f"Failed to read context file {file_path}: {e}")

        if not context_content:
            return results

        # Boost results that are related to context files
        context_text = " ".join(context_content).lower()

        for result in results:
            # Check if result is from one of the context files
            if result.file_path in context_files:
                result.similarity_score = min(1.0, result.similarity_score + 0.1)

            # Check if result content relates to context
            if result.function_name:
                func_name_lower = result.function_name.lower()
                if func_name_lower in context_text:
                    result.similarity_score = min(1.0, result.similarity_score + 0.05)

            if result.class_name:
                class_name_lower = result.class_name.lower()
                if class_name_lower in context_text:
                    result.similarity_score = min(1.0, result.similarity_score + 0.05)

        # Re-sort by updated scores
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def clear_cache(self) -> None:
        """Clear the file read cache."""
        self._file_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("File read cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hits, misses, size, and hit rate
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._file_cache),
            "maxsize": self._cache_maxsize,
            "hit_rate": f"{hit_rate:.2%}",
        }

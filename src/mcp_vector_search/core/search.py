"""Semantic search engine for MCP Vector Search."""

import re
import time
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger

from .auto_indexer import AutoIndexer, SearchTriggeredIndexer
from .database import VectorDatabase
from .exceptions import RustPanicError, SearchError
from .models import SearchResult
from .query_analyzer import QueryAnalyzer
from .query_processor import QueryProcessor
from .result_enhancer import ResultEnhancer
from .result_ranker import ResultRanker
from .search_retry_handler import SearchRetryHandler


class SemanticSearchEngine:
    """Semantic search engine for code search.

    This class coordinates search operations by delegating to specialized components:
    - QueryProcessor: Query preprocessing and threshold calculation
    - SearchRetryHandler: Retry logic and error handling
    - ResultEnhancer: Result enhancement with context and caching
    - ResultRanker: Result reranking and scoring
    - QueryAnalyzer: Query analysis and suggestions
    """

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.3,
        auto_indexer: AutoIndexer | None = None,
        enable_auto_reindex: bool = True,
    ) -> None:
        """Initialize semantic search engine.

        Args:
            database: Vector database instance
            project_root: Project root directory
            similarity_threshold: Default similarity threshold
            auto_indexer: Optional auto-indexer for semi-automatic reindexing
            enable_auto_reindex: Whether to enable automatic reindexing
        """
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold
        self.auto_indexer = auto_indexer
        self.enable_auto_reindex = enable_auto_reindex

        # Initialize search-triggered indexer if auto-indexer is provided
        self.search_triggered_indexer = None
        if auto_indexer and enable_auto_reindex:
            self.search_triggered_indexer = SearchTriggeredIndexer(auto_indexer)

        # Health check throttling (only check every 60 seconds)
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 60.0

        # Initialize helper components
        self._query_processor = QueryProcessor(base_threshold=similarity_threshold)
        self._retry_handler = SearchRetryHandler()
        self._result_enhancer = ResultEnhancer()
        self._result_ranker = ResultRanker()
        self._query_analyzer = QueryAnalyzer(self._query_processor)

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        include_context: bool = True,
    ) -> list[SearchResult]:
        """Perform semantic search for code.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (language, file_path, etc.)
            similarity_threshold: Minimum similarity score
            include_context: Whether to include context lines

        Returns:
            List of search results
        """
        if not query.strip():
            return []

        # Throttled health check before search (only every 60 seconds)
        await self._perform_health_check()

        # Auto-reindex check before search
        await self._perform_auto_reindex_check()

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._query_processor.get_adaptive_threshold(query)
        )

        try:
            # Preprocess query
            processed_query = self._query_processor.preprocess_query(query)

            # Perform vector search with retry logic
            results = await self._retry_handler.search_with_retry(
                database=self.database,
                query=processed_query,
                limit=limit,
                filters=filters,
                threshold=threshold,
            )

            # Post-process results
            enhanced_results = []
            for result in results:
                enhanced_result = await self._result_enhancer.enhance_result(
                    result, include_context
                )
                enhanced_results.append(enhanced_result)

            # Apply additional ranking if needed
            ranked_results = self._result_ranker.rerank_results(enhanced_results, query)

            logger.debug(
                f"Search for '{query}' with threshold {threshold:.3f} returned {len(ranked_results)} results"
            )
            return ranked_results

        except (RustPanicError, SearchError):
            # These errors are already properly formatted with user guidance
            raise
        except Exception as e:
            # Unexpected error - wrap it in SearchError
            logger.error(f"Unexpected search error for query '{query}': {e}")
            raise SearchError(f"Search failed: {e}") from e

    async def search_similar(
        self,
        file_path: Path,
        function_name: str | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Find code similar to a specific function or file.

        Args:
            file_path: Path to the reference file
            function_name: Specific function name (optional)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar code results
        """
        try:
            # Read the reference file using async I/O
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()

            # If function name is specified, try to extract just that function
            if function_name:
                function_content = self._extract_function_content(
                    content, function_name
                )
                if function_content:
                    content = function_content

            # Use the content as the search query
            return await self.search(
                query=content,
                limit=limit,
                similarity_threshold=similarity_threshold,
                include_context=True,
            )

        except Exception as e:
            logger.error(f"Similar search failed for {file_path}: {e}")
            raise SearchError(f"Similar search failed: {e}") from e

    async def search_by_context(
        self,
        context_description: str,
        focus_areas: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for code based on contextual description.

        Args:
            context_description: Description of what you're looking for
            focus_areas: Areas to focus on (e.g., ["security", "authentication"])
            limit: Maximum number of results

        Returns:
            List of contextually relevant results
        """
        # Build enhanced query with focus areas
        query_parts = [context_description]

        if focus_areas:
            query_parts.extend(focus_areas)

        enhanced_query = " ".join(query_parts)

        return await self.search(
            query=enhanced_query,
            limit=limit,
            include_context=True,
        )

    async def search_with_context(
        self,
        query: str,
        context_files: list[Path] | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Enhanced search with contextual analysis and suggestions.

        Args:
            query: Search query
            context_files: Optional list of files to provide context
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score

        Returns:
            Dictionary with results, analysis, and suggestions
        """
        # Analyze the query
        query_analysis = self._query_analyzer.analyze_query(query)

        # Perform the search
        results = await self.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            include_context=True,
        )

        # Get related query suggestions
        suggestions = self._query_analyzer.suggest_related_queries(query, results)

        # Enhance results with additional context if context files provided
        if context_files:
            results = await self._result_enhancer.enhance_with_file_context(
                results, context_files
            )

        # Calculate result quality metrics
        quality_metrics = self._query_analyzer.calculate_result_quality(results, query)

        return {
            "query": query,
            "analysis": query_analysis,
            "results": results,
            "suggestions": suggestions,
            "metrics": quality_metrics,
            "total_results": len(results),
        }

    def analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze search query and provide suggestions for improvement.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with analysis results and suggestions
        """
        return self._query_analyzer.analyze_query(query)

    def suggest_related_queries(
        self, query: str, results: list[SearchResult]
    ) -> list[str]:
        """Suggest related queries based on search results.

        Args:
            query: Original search query
            results: Search results

        Returns:
            List of suggested related queries
        """
        return self._query_analyzer.suggest_related_queries(query, results)

    async def get_search_stats(self) -> dict[str, Any]:
        """Get search engine statistics.

        Returns:
            Dictionary with search statistics
        """
        try:
            db_stats = await self.database.get_stats()

            return {
                "total_chunks": db_stats.total_chunks,
                "languages": db_stats.languages,
                "similarity_threshold": self.similarity_threshold,
                "project_root": str(self.project_root),
            }

        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}

    def clear_cache(self) -> None:
        """Clear the file read cache."""
        self._result_enhancer.clear_cache()

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hits, misses, size, and hit rate
        """
        return self._result_enhancer.get_cache_info()

    # Private helper methods

    async def _perform_health_check(self) -> None:
        """Perform throttled health check on database."""
        current_time = time.time()
        if current_time - self._last_health_check >= self._health_check_interval:
            try:
                if hasattr(self.database, "health_check"):
                    is_healthy = await self.database.health_check()
                    if not is_healthy:
                        logger.warning(
                            "Database health check failed - attempting recovery"
                        )
                        # Health check already attempts recovery, so we can proceed
                    self._last_health_check = current_time
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                self._last_health_check = current_time

    async def _perform_auto_reindex_check(self) -> None:
        """Perform auto-reindex check before search."""
        if self.search_triggered_indexer:
            try:
                await self.search_triggered_indexer.pre_search_hook()
            except Exception as e:
                logger.warning(f"Auto-reindex check failed: {e}")

    @staticmethod
    def _extract_function_content(content: str, function_name: str) -> str | None:
        """Extract content of a specific function from code.

        Args:
            content: Full file content
            function_name: Name of function to extract

        Returns:
            Function content if found, None otherwise
        """
        # Simple regex-based extraction (could be improved with AST)
        pattern = rf"^\s*def\s+{re.escape(function_name)}\s*\("
        lines = content.splitlines()

        for i, line in enumerate(lines):
            if re.match(pattern, line):
                # Found function start, now find the end
                start_line = i
                indent_level = len(line) - len(line.lstrip())

                # Find end of function
                end_line = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():  # Skip empty lines
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent <= indent_level:
                            end_line = j
                            break

                return "\n".join(lines[start_line:end_line])

        return None

    # Expose internal methods for backward compatibility (used in tests)
    def _get_adaptive_threshold(self, query: str) -> float:
        """Get adaptive similarity threshold (backward compatibility)."""
        return self._query_processor.get_adaptive_threshold(query)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query (backward compatibility)."""
        return self._query_processor.preprocess_query(query)

    async def _enhance_result(
        self, result: SearchResult, include_context: bool
    ) -> SearchResult:
        """Enhance result (backward compatibility)."""
        return await self._result_enhancer.enhance_result(result, include_context)

    def _rerank_results(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """Rerank results (backward compatibility)."""
        return self._result_ranker.rerank_results(results, query)

    async def _read_file_lines_cached(self, file_path: Path) -> list[str]:
        """Read file lines cached (backward compatibility)."""
        return await self._result_enhancer.read_file_lines_cached(file_path)

    def _calculate_result_quality(
        self, results: list[SearchResult], query: str
    ) -> dict[str, Any]:
        """Calculate result quality (backward compatibility)."""
        return self._query_analyzer.calculate_result_quality(results, query)

    async def _enhance_with_file_context(
        self, results: list[SearchResult], context_files: list[Path]
    ) -> list[SearchResult]:
        """Enhance with file context (backward compatibility)."""
        return await self._result_enhancer.enhance_with_file_context(
            results, context_files
        )

    async def _search_with_retry(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        threshold: float,
        max_retries: int = 3,
    ) -> list[SearchResult]:
        """Search with retry (backward compatibility)."""
        return await self._retry_handler.search_with_retry(
            database=self.database,
            query=query,
            limit=limit,
            filters=filters,
            threshold=threshold,
            max_retries=max_retries,
        )

    @staticmethod
    def _is_rust_panic_error(error: Exception) -> bool:
        """Detect Rust panic errors (backward compatibility)."""
        return SearchRetryHandler.is_rust_panic_error(error)

    @staticmethod
    def _is_corruption_error(error: Exception) -> bool:
        """Detect corruption errors (backward compatibility)."""
        return SearchRetryHandler.is_corruption_error(error)

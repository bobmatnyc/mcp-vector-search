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
from .knowledge_graph import KnowledgeGraph
from .models import SearchResult
from .query_analyzer import QueryAnalyzer
from .query_processor import QueryProcessor
from .result_enhancer import ResultEnhancer
from .result_ranker import ResultRanker
from .search_retry_handler import SearchRetryHandler
from .vectors_backend import VectorsBackend


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
        enable_kg: bool = True,
    ) -> None:
        """Initialize semantic search engine.

        Args:
            database: Vector database instance
            project_root: Project root directory
            similarity_threshold: Default similarity threshold
            auto_indexer: Optional auto-indexer for semi-automatic reindexing
            enable_auto_reindex: Whether to enable automatic reindexing
            enable_kg: Whether to enable knowledge graph enhancement
        """
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold
        self.auto_indexer = auto_indexer
        self.enable_auto_reindex = enable_auto_reindex
        self.enable_kg = enable_kg

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

        # Two-phase architecture: lazy detection of VectorsBackend
        # We check at search time rather than init time because vectors.lance
        # may not exist yet when SemanticSearchEngine is created
        self._vectors_backend: VectorsBackend | None = None
        self._vectors_backend_checked = False

        # Knowledge graph (lazy initialization)
        self._kg: KnowledgeGraph | None = None
        self._kg_checked = False

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

            # Two-phase architecture: check for VectorsBackend on first search
            if not self._vectors_backend_checked:
                self._check_vectors_backend()
                self._vectors_backend_checked = True

            # Check for knowledge graph on first search
            if not self._kg_checked:
                await self._check_knowledge_graph()
                self._kg_checked = True

            # Use VectorsBackend if available, otherwise fall back to ChromaDB
            if self._vectors_backend:
                results = await self._search_vectors_backend(
                    processed_query, limit, filters, threshold
                )
            else:
                # Legacy ChromaDB search
                results = await self._retry_handler.search_with_retry(
                    database=self.database,
                    query=processed_query,
                    limit=limit,
                    filters=filters,
                    threshold=threshold,
                )

            # Enhance with knowledge graph context if available
            # Gracefully skip if KG is unavailable (may be building)
            if self._kg and self.enable_kg:
                try:
                    results = await self._enhance_with_kg(results, query)
                except Exception as e:
                    logger.debug(f"KG enhancement skipped: {e}")
                    # Continue without KG - still return results

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
            async with aiofiles.open(
                file_path, encoding="utf-8", errors="replace"
            ) as f:
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

    def _check_vectors_backend(self) -> None:
        """Check if VectorsBackend is available for two-phase architecture.

        This enables the search engine to automatically use the new LanceDB-based
        vectors_backend when available, while falling back to ChromaDB for legacy support.

        Called lazily on first search to ensure vectors.lance exists.
        """
        try:
            # Detect index path from database persist_directory
            if hasattr(self.database, "persist_directory"):
                index_path = self.database.persist_directory

                # Handle both old (config.index_path) and new (config.index_path/lance) paths
                # Check if index_path already ends with "lance"
                if index_path.name == "lance":
                    # New path format: index_path already is the lance directory
                    lance_path = index_path
                else:
                    # Old path format: need to add /lance
                    lance_path = index_path / "lance"

                # Check if vectors.lance table exists (LanceDB format is a directory)
                vectors_path = lance_path / "vectors.lance"
                if vectors_path.exists() and vectors_path.is_dir():
                    # Instantiate VectorsBackend with lance path
                    vectors_backend = VectorsBackend(lance_path)
                    self._vectors_backend = vectors_backend
                    logger.debug(
                        "Two-phase architecture detected: using VectorsBackend for search"
                    )
                else:
                    logger.debug(
                        f"VectorsBackend not found at {vectors_path}, using legacy ChromaDB search"
                    )
            else:
                logger.debug("Database has no persist_directory, using legacy search")
        except Exception as e:
            logger.debug(f"VectorsBackend detection failed: {e}, using legacy search")
            self._vectors_backend = None

    async def _check_knowledge_graph(self) -> None:
        """Check if knowledge graph is available and initialize if needed.

        This lazily initializes the KG on first search if it exists.
        Gracefully handles unavailable or building KG.
        """
        if not self.enable_kg:
            return

        try:
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg_db_path = kg_path / "code_kg"

            if kg_db_path.exists():
                self._kg = KnowledgeGraph(kg_path)
                await self._kg.initialize()
                logger.debug("Knowledge graph loaded for search enhancement")
            else:
                logger.debug(
                    "Knowledge graph not found yet (may be building in background), "
                    "search will work without KG enhancement"
                )
        except Exception as e:
            logger.debug(
                f"Knowledge graph initialization failed: {e}, "
                "continuing without KG enhancement"
            )
            self._kg = None

    async def _enhance_with_kg(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """Enhance results with knowledge graph context.

        Boosts results that have related entities matching query terms.

        Args:
            results: Initial search results
            query: Search query

        Returns:
            Enhanced and re-sorted results
        """
        if not self._kg:
            return results

        try:
            # Extract query terms for matching
            query_terms = set(query.lower().split())

            for result in results:
                # Find related entities (1 hop for performance)
                chunk_id = getattr(result, "chunk_id", None)
                if not chunk_id:
                    continue

                try:
                    related = await self._kg.find_related(chunk_id, max_hops=1)

                    # Boost if related entities match query terms
                    for rel in related:
                        rel_name = rel["name"].lower()
                        if any(term in rel_name for term in query_terms):
                            # Small boost for KG relationship match
                            result.similarity_score = min(
                                1.0, result.similarity_score + 0.02
                            )
                            logger.debug(
                                f"KG boost: {result.file_path} related to {rel['name']}"
                            )

                except Exception as e:
                    logger.debug(f"Failed to enhance result with KG: {e}")
                    continue

            # Re-sort by updated scores
            return sorted(results, key=lambda r: r.similarity_score, reverse=True)

        except Exception as e:
            logger.warning(f"KG enhancement failed: {e}")
            return results

    async def _search_vectors_backend(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        threshold: float,
    ) -> list[SearchResult]:
        """Search using VectorsBackend (two-phase architecture).

        Args:
            query: Preprocessed search query
            limit: Maximum number of results
            filters: Optional metadata filters
            threshold: Similarity threshold

        Returns:
            List of search results

        Raises:
            SearchError: If search fails
        """
        try:
            # Initialize vectors_backend if needed
            if self._vectors_backend._db is None:
                await self._vectors_backend.initialize()

            # Generate query embedding
            # Extract embedding function from database (ChromaDB wrapper)
            if hasattr(self.database, "_embedding_function"):
                embedding_func = self.database._embedding_function
            elif hasattr(self.database, "embedding_function"):
                embedding_func = self.database.embedding_function
            else:
                raise SearchError(
                    "Cannot access embedding function from database for vector search"
                )

            # Check for model mismatch
            current_model = getattr(embedding_func, "model_name", "unknown")

            # Try to read stored model from first vector in table
            if self._vectors_backend._table is not None:
                try:
                    # Get first row to check stored model_version
                    df = self._vectors_backend._table.to_pandas().head(1)
                    if not df.empty and "model_version" in df.columns:
                        stored_model = df.iloc[0]["model_version"]

                        # Check for model name mismatch
                        if (
                            stored_model != current_model
                            and stored_model not in current_model
                        ):
                            logger.warning(
                                f"Model mismatch: Index was built with '{stored_model}', "
                                f"searching with '{current_model}'. Results may be poor. "
                                f"Reindex with --force to rebuild with current model."
                            )

                        # Check for dimension mismatch
                        stored_vector = df.iloc[0]["vector"]
                        stored_dim = len(stored_vector)
                        query_dim = len(embedding_func([query])[0])

                        if stored_dim != query_dim:
                            raise SearchError(
                                f"Dimension mismatch: Index has {stored_dim}D vectors, "
                                f"current model produces {query_dim}D. "
                                f"Reindex with --force to rebuild with current model."
                            )
                except Exception as e:
                    # Non-fatal: log and continue
                    logger.debug(f"Could not check model version: {e}")

            # Generate embedding
            query_vector = embedding_func([query])[0]

            # Search vectors backend
            raw_results = await self._vectors_backend.search(
                query_vector, limit=limit, filters=filters
            )

            # Convert to SearchResult format
            search_results = []
            for idx, result in enumerate(raw_results):
                # Calculate similarity score from distance
                # LanceDB returns _distance (cosine distance, 0-2 range)
                # Convert to similarity: 0 distance -> 1.0 similarity, 2 distance -> 0.0
                distance = result.get("_distance", 1.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                # Apply threshold filter
                if similarity < threshold:
                    continue

                search_result = SearchResult(
                    file_path=Path(result["file_path"]),
                    content=result["content"],
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    language=result.get("language", "unknown"),
                    similarity_score=similarity,
                    rank=idx + 1,  # 1-based ranking
                    chunk_type=result.get("chunk_type", "unknown"),
                    function_name=result.get("name"),  # Map 'name' to 'function_name'
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"VectorsBackend search failed: {e}")
            raise SearchError(f"Vector search failed: {e}") from e

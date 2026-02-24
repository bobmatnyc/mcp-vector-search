"""Semantic search engine for MCP Vector Search."""

import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger

from .auto_indexer import AutoIndexer, SearchTriggeredIndexer
from .bm25_backend import BM25Backend
from .database import VectorDatabase
from .exceptions import RustPanicError, SearchError
from .knowledge_graph import KnowledgeGraph
from .mmr import mmr_rerank
from .models import SearchResult
from .query_analyzer import QueryAnalyzer
from .query_expander import QueryExpander
from .query_processor import QueryProcessor
from .reranker import CrossEncoderReranker
from .result_enhancer import ResultEnhancer
from .result_ranker import ResultRanker
from .search_retry_handler import SearchRetryHandler
from .vectors_backend import VectorsBackend

# Reciprocal Rank Fusion (RRF) smoothing constant
# Default k=60 is standard in literature for balancing vector and keyword search ranks
RRF_K = 60


class SearchMode(str, Enum):
    """Search mode for semantic search engine."""

    VECTOR = "vector"  # Pure vector similarity search
    BM25 = "bm25"  # Pure BM25 keyword search
    HYBRID = "hybrid"  # Hybrid: RRF fusion of vector + BM25


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
        enable_query_expansion: bool = True,
    ) -> None:
        """Initialize semantic search engine.

        Args:
            database: Vector database instance
            project_root: Project root directory
            similarity_threshold: Default similarity threshold
            auto_indexer: Optional auto-indexer for semi-automatic reindexing
            enable_auto_reindex: Whether to enable automatic reindexing
            enable_kg: Whether to enable knowledge graph enhancement
            enable_query_expansion: Whether to enable query expansion with synonyms
        """
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold
        self.auto_indexer = auto_indexer
        self.enable_auto_reindex = enable_auto_reindex
        self.enable_kg = enable_kg
        self.enable_query_expansion = enable_query_expansion

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

        # Initialize query expander with custom synonyms if available
        custom_synonyms_path = project_root / ".mcp-vector-search" / "synonyms.json"
        self._query_expander = QueryExpander(
            custom_synonyms_path=custom_synonyms_path
            if custom_synonyms_path.exists()
            else None
        )

        # Two-phase architecture: lazy detection of VectorsBackend
        # We check at search time rather than init time because vectors.lance
        # may not exist yet when SemanticSearchEngine is created
        self._vectors_backend: VectorsBackend | None = None
        self._vectors_backend_checked = False

        # Code vectors backend (lazy initialization for code enrichment)
        self._code_vectors_backend: VectorsBackend | None = None
        self._code_vectors_backend_checked = False
        self._code_embedding_func = None  # Cache CodeT5+ model

        # Knowledge graph (lazy initialization)
        self._kg: KnowledgeGraph | None = None
        self._kg_checked = False

        # BM25 backend (lazy initialization)
        self._bm25_backend: BM25Backend | None = None
        self._bm25_backend_checked = False

        # Cross-encoder reranker (lazy initialization)
        self._reranker: CrossEncoderReranker | None = None

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        include_context: bool = True,
        search_mode: SearchMode = SearchMode.HYBRID,
        hybrid_alpha: float = 0.7,
        expand: bool = True,
        use_rerank: bool = True,
        rerank_top_n: int = 50,
        use_mmr: bool = True,
        diversity: float = 0.5,
    ) -> list[SearchResult]:
        """Perform semantic search for code.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (language, file_path, etc.)
            similarity_threshold: Minimum similarity score
            include_context: Whether to include context lines
            search_mode: Search mode (vector, bm25, or hybrid)
            hybrid_alpha: Weight for vector search in hybrid mode (0.0-1.0, default 0.7)
                         1.0 = pure vector, 0.0 = pure BM25
            expand: Whether to expand query with synonyms (default: True)
            use_rerank: Whether to apply cross-encoder reranking (default: True)
            rerank_top_n: Number of candidates to retrieve before reranking (default: 50)
                         Reranking over-retrieves then reranks to top limit*3 for MMR
            use_mmr: Whether to apply MMR diversity filtering (default: True)
            diversity: Diversity parameter for MMR (0.0-1.0, default 0.5)
                      0.0 = pure relevance, 1.0 = maximum diversity

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

            # Check for BM25 backend on first search
            if not self._bm25_backend_checked:
                self._check_bm25_backend()
                self._bm25_backend_checked = True

            # Query expansion: generate variants if enabled
            if expand and self.enable_query_expansion:
                query_variants = self._query_expander.expand(processed_query)
                logger.debug(
                    f"Query expansion: {len(query_variants)} variants for '{processed_query}'"
                )
            else:
                query_variants = [processed_query]

            # Determine retrieval limit based on reranking
            # If reranking enabled, over-retrieve to get more candidates
            retrieval_limit = rerank_top_n if use_rerank else limit

            # Search with all query variants and merge results
            all_results: dict[str, SearchResult] = {}  # chunk_id -> result (best score)

            for variant_query in query_variants:
                # Route search based on mode
                if search_mode == SearchMode.BM25:
                    # Pure BM25 search
                    variant_results = await self._search_bm25(
                        variant_query, retrieval_limit, filters, threshold
                    )
                elif search_mode == SearchMode.HYBRID:
                    # Hybrid search with RRF fusion
                    variant_results = await self._search_hybrid(
                        variant_query, retrieval_limit, filters, threshold, hybrid_alpha
                    )
                else:
                    # Pure vector search (default)
                    # Use VectorsBackend if available, otherwise fall back to ChromaDB
                    if self._vectors_backend:
                        variant_results = await self._search_vectors_backend(
                            variant_query, retrieval_limit, filters, threshold
                        )
                    else:
                        # Legacy ChromaDB search
                        variant_results = await self._retry_handler.search_with_retry(
                            database=self.database,
                            query=variant_query,
                            limit=retrieval_limit,
                            filters=filters,
                            threshold=threshold,
                        )

                # Merge results: keep highest score per chunk_id
                for result in variant_results:
                    # Get chunk_id (unique identifier for each code chunk)
                    chunk_id = getattr(result, "chunk_id", None)
                    if not chunk_id:
                        # Fallback: construct chunk_id from file_path + line range
                        chunk_id = (
                            f"{result.file_path}:{result.start_line}-{result.end_line}"
                        )

                    # Keep result with highest similarity score
                    if (
                        chunk_id not in all_results
                        or result.similarity_score
                        > all_results[chunk_id].similarity_score
                    ):
                        all_results[chunk_id] = result

            # Convert merged results to list and sort by score
            results = sorted(
                all_results.values(), key=lambda r: r.similarity_score, reverse=True
            )

            if expand and self.enable_query_expansion and len(query_variants) > 1:
                logger.debug(
                    f"Query expansion merged {len(all_results)} unique results from {len(query_variants)} variants"
                )

            # Apply cross-encoder reranking if enabled and we have results
            # Reranking improves precision by scoring (query, document) pairs jointly
            if use_rerank and results and len(results) > 1:
                try:
                    results = await self._apply_cross_encoder_reranking(
                        results=results,
                        query=processed_query,
                        requested_limit=limit,
                    )
                except Exception as e:
                    logger.warning(
                        f"Cross-encoder reranking failed, continuing with original results: {e}"
                    )

            # Limit to requested number of results (after reranking, before MMR)
            # Keep limit*3 for MMR to have candidates to diversify from
            if use_mmr and len(results) > limit:
                results = results[: limit * 3]
            else:
                results = results[:limit]

            # Apply MMR diversity filtering if enabled and we have enough results
            # MMR improves diversity by penalizing results similar to already-selected ones
            if use_mmr and results and len(results) > 1 and self._vectors_backend:
                try:
                    results = await self._apply_mmr_reranking(
                        results=results,
                        query=processed_query,
                        diversity=diversity,
                        requested_limit=limit,
                    )
                except Exception as e:
                    logger.warning(
                        f"MMR reranking failed, continuing with original results: {e}"
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
                        # Use embed_query() so query prefix is applied for asymmetric models
                        if hasattr(embedding_func, "embed_query"):
                            query_dim = len(embedding_func.embed_query(query))
                        else:
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

            # Generate embedding — use embed_query() so query_prefix is applied
            # for asymmetric models (e.g. nomic-ai/CodeRankEmbed).
            # embed_query() falls back to __call__ with no prefix for symmetric models.
            if hasattr(embedding_func, "embed_query"):
                query_vector = embedding_func.embed_query(query)
            else:
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

            # Code enrichment: opt-in only (set MCP_CODE_ENRICHMENT=true to enable)
            # CodeT5+ enrichment is experimental - see feature/code-index branch
            if os.environ.get("MCP_CODE_ENRICHMENT", "").lower() == "true":
                search_results = await self._enrich_with_code_vectors(
                    query, search_results, limit, filters
                )

            return search_results

        except Exception as e:
            logger.error(f"VectorsBackend search failed: {e}")
            raise SearchError(f"Vector search failed: {e}") from e

    async def _enrich_with_code_vectors(
        self,
        query: str,
        main_results: list[SearchResult],
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Enrich search results with code_vectors table if available.

        This method performs a second search using the CodeT5+ model on the
        code_vectors table (if it exists) and boosts results that appear in BOTH
        the main index and code index.

        Args:
            query: Original search query
            main_results: Results from main vectors.lance search
            limit: Maximum number of results
            filters: Optional metadata filters

        Returns:
            Enriched and re-sorted search results
        """
        # Lazy check for code_vectors backend on first search
        if not self._code_vectors_backend_checked:
            await self._check_code_vectors_backend()
            self._code_vectors_backend_checked = True

        # If no code_vectors backend, return main results unchanged
        if not self._code_vectors_backend:
            return main_results

        try:
            # Initialize code_vectors backend if needed
            if self._code_vectors_backend._db is None:
                await self._code_vectors_backend.initialize()

            # Lazy load CodeT5+ embedding function (cache for performance)
            if self._code_embedding_func is None:
                from ..config.defaults import DEFAULT_EMBEDDING_MODELS
                from .embeddings import CodeBERTEmbeddingFunction

                code_model = DEFAULT_EMBEDDING_MODELS["code_specialized"]
                self._code_embedding_func = CodeBERTEmbeddingFunction(code_model)
                logger.debug(f"Loaded CodeT5+ model for code enrichment: {code_model}")

            # Generate code embedding for query — use embed_query() so any
            # query prefix configured for the code model is applied correctly.
            if hasattr(self._code_embedding_func, "embed_query"):
                code_query_vector = self._code_embedding_func.embed_query(query)
            else:
                code_query_vector = self._code_embedding_func([query])[0]

            # Search code_vectors with higher threshold (0.75 vs 0.5)
            # Code-specific models need higher confidence for relevance
            code_threshold = 0.75
            code_raw_results = await self._code_vectors_backend.search(
                code_query_vector,
                limit=limit * 2,
                filters=filters,  # Get more candidates
            )

            # Convert to chunk_id -> similarity map for fast lookup
            code_similarity_map = {}
            for result in code_raw_results:
                # Calculate similarity from distance
                distance = result.get("_distance", 1.0)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                # Apply code-specific threshold
                if similarity >= code_threshold:
                    chunk_id = result.get("chunk_id")
                    if chunk_id:
                        code_similarity_map[chunk_id] = similarity

            if not code_similarity_map:
                # No code results above threshold
                return main_results

            logger.info(
                f"Code index detected, enriching search results "
                f"(found {len(code_similarity_map)} code matches above {code_threshold} threshold)"
            )

            # Boost main results that also appear in code index
            boosted_count = 0
            for result in main_results:
                # Get chunk_id from result (construct from file_path + line range)
                chunk_id = getattr(result, "chunk_id", None)
                if not chunk_id:
                    # Fallback: construct chunk_id from file path and lines
                    # This matches the chunk_id format used during indexing
                    chunk_id = (
                        f"{result.file_path}:{result.start_line}-{result.end_line}"
                    )

                # If result appears in code index, boost its score
                if chunk_id in code_similarity_map:
                    # Boost by 0.15 for appearing in both indices
                    original_score = result.similarity_score
                    result.similarity_score = min(1.0, result.similarity_score + 0.15)
                    boosted_count += 1
                    logger.debug(
                        f"Boosted {result.file_path}:{result.start_line} "
                        f"from {original_score:.3f} to {result.similarity_score:.3f} "
                        f"(code similarity: {code_similarity_map[chunk_id]:.3f})"
                    )

            if boosted_count > 0:
                logger.info(
                    f"Boosted {boosted_count} results that appear in both main and code indices"
                )
                # Re-sort by updated scores
                main_results.sort(key=lambda r: r.similarity_score, reverse=True)

            return main_results

        except Exception as e:
            # Non-fatal: code enrichment failure shouldn't break search
            logger.warning(
                f"Code enrichment failed (continuing with main results): {e}"
            )
            return main_results

    async def _check_code_vectors_backend(self) -> None:
        """Check if code_vectors.lance table exists for code enrichment.

        This enables automatic code-specific search enrichment when available.
        """
        try:
            # Detect lance path from database persist_directory
            if hasattr(self.database, "persist_directory"):
                index_path = self.database.persist_directory

                # Handle both old and new path formats
                if index_path.name == "lance":
                    lance_path = index_path
                else:
                    lance_path = index_path / "lance"

                # Check if code_vectors.lance table exists
                code_vectors_path = lance_path / "code_vectors.lance"
                if code_vectors_path.exists() and code_vectors_path.is_dir():
                    # Instantiate VectorsBackend for code_vectors with 256D (CodeT5+)
                    self._code_vectors_backend = VectorsBackend(
                        lance_path, vector_dim=256, table_name="code_vectors"
                    )
                    logger.debug(
                        "Code vectors table detected, search enrichment enabled"
                    )
                else:
                    logger.debug(
                        "No code_vectors table found, code enrichment disabled"
                    )
            else:
                logger.debug(
                    "Database has no persist_directory, code enrichment disabled"
                )
        except Exception as e:
            logger.debug(f"Code vectors backend detection failed: {e}")
            self._code_vectors_backend = None

    async def _apply_cross_encoder_reranking(
        self,
        results: list[SearchResult],
        query: str,
        requested_limit: int,
    ) -> list[SearchResult]:
        """Apply cross-encoder reranking for higher precision results.

        Cross-encoders score (query, document) pairs jointly, producing more
        accurate relevance scores than bi-encoder similarity. This improves
        precision at the cost of some latency.

        Args:
            results: Initial search results to rerank
            query: Original search query
            requested_limit: Requested number of final results

        Returns:
            Reranked results with updated scores

        Note:
            - Lazy-initializes cross-encoder on first use
            - Reranks all candidates, returns top limit*3 for MMR
            - Updates similarity_score with cross-encoder scores
            - Gracefully falls back on failure (non-fatal)
        """
        # Lazy-initialize reranker
        if self._reranker is None:
            if not CrossEncoderReranker().is_available:
                logger.debug(
                    "Cross-encoder reranking unavailable (sentence-transformers not installed)"
                )
                return results

            self._reranker = CrossEncoderReranker()
            logger.debug(
                f"Initialized cross-encoder reranker: {self._reranker.model_name}"
            )

        # Extract document contents for reranking
        documents = [result.content for result in results]

        # Rerank all candidates
        # Returns list of (original_index, score) tuples sorted by score
        reranked_indices = self._reranker.rerank(
            query=query,
            documents=documents,
            top_k=min(len(results), requested_limit * 3),  # Keep 3x for MMR
        )

        # Build reranked results with normalized scores
        # Cross-encoder outputs raw logits — apply sigmoid to map to [0, 1] probability
        import math

        reranked_results = []
        for rank, (original_idx, score) in enumerate(reranked_indices, start=1):
            result = results[original_idx]
            # Sigmoid normalization: logit → probability of relevance [0.0, 1.0]
            result.similarity_score = 1.0 / (1.0 + math.exp(-score))
            result.rank = rank
            reranked_results.append(result)

        logger.info(
            f"Cross-encoder reranked {len(results)} candidates → "
            f"top {len(reranked_results)} results "
            f"(score range: {reranked_results[0].similarity_score:.3f} - "
            f"{reranked_results[-1].similarity_score:.3f})"
        )

        return reranked_results

    async def _apply_mmr_reranking(
        self,
        results: list[SearchResult],
        query: str,
        diversity: float,
        requested_limit: int,
    ) -> list[SearchResult]:
        """Apply MMR (Maximal Marginal Relevance) reranking for diverse results.

        MMR reduces redundancy by penalizing results similar to already-selected ones.
        This is particularly useful when the top results are very similar to each other.

        Args:
            results: Initial search results to rerank
            query: Original search query
            diversity: Diversity parameter (0.0=pure diversity, 1.0=pure relevance)
            requested_limit: Requested number of final results

        Returns:
            Reranked results with improved diversity

        Note:
            - Over-retrieves by 3x to give MMR more candidates to diversify from
            - Maps diversity (user-facing) to lambda_param (algorithm): lambda = 1 - diversity
            - Requires embeddings from vectors_backend for similarity calculations
        """
        import numpy as np

        # Over-retrieve by 3x to give MMR candidates to choose from
        # This allows MMR to select diverse results from a larger pool
        retrieval_limit = min(len(results), requested_limit * 3)
        candidate_results = results[:retrieval_limit]

        # Get query embedding (generate from query string)
        if hasattr(self.database, "_embedding_function"):
            embedding_func = self.database._embedding_function
        elif hasattr(self.database, "embedding_function"):
            embedding_func = self.database.embedding_function
        else:
            logger.warning("Cannot access embedding function, skipping MMR")
            return results[:requested_limit]

        # Use embed_query() so query_prefix is applied for asymmetric models
        if hasattr(embedding_func, "embed_query"):
            query_vector = embedding_func.embed_query(query)
        else:
            query_vector = embedding_func([query])[0]
        query_embedding = np.array(query_vector, dtype=np.float32)

        # Get embeddings for all candidates
        # Extract chunk_ids from results (may be stored as attribute or constructed)
        chunk_ids = []
        for result in candidate_results:
            chunk_id = getattr(result, "chunk_id", None)
            if not chunk_id:
                # Fallback: construct chunk_id from file path and lines
                chunk_id = f"{result.file_path}:{result.start_line}-{result.end_line}"
            chunk_ids.append(chunk_id)

        # Retrieve embeddings from vectors backend
        chunk_vectors = await self._vectors_backend.get_chunk_vectors_batch(chunk_ids)

        # Filter results to only those with embeddings available
        valid_indices = []
        valid_embeddings = []
        valid_scores = []
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id in chunk_vectors:
                valid_indices.append(i)
                valid_embeddings.append(chunk_vectors[chunk_id])
                valid_scores.append(candidate_results[i].similarity_score)

        if len(valid_indices) < 2:
            # Not enough embeddings for MMR, return original results
            logger.debug(
                f"MMR skipped: only {len(valid_indices)} results have embeddings"
            )
            return results[:requested_limit]

        # Convert to numpy arrays
        candidate_embeddings = np.array(valid_embeddings, dtype=np.float32)

        # Convert diversity parameter to lambda_param
        # diversity=0.0 -> lambda=1.0 (pure relevance, no diversity)
        # diversity=1.0 -> lambda=0.0 (pure diversity, no relevance)
        # diversity=0.5 -> lambda=0.5 (balanced)
        lambda_param = 1.0 - diversity

        # Run MMR reranking
        selected_indices = mmr_rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_scores=valid_scores,
            lambda_param=lambda_param,
            top_k=requested_limit,
        )

        # Map selected indices back to original results
        # selected_indices are indices into valid_indices
        reranked_results = []
        for mmr_idx in selected_indices:
            original_idx = valid_indices[mmr_idx]
            result = candidate_results[original_idx]
            # Update rank to reflect new position
            result.rank = len(reranked_results) + 1
            reranked_results.append(result)

        logger.info(
            f"MMR reranking: selected {len(reranked_results)} diverse results from {len(candidate_results)} candidates "
            f"(diversity={diversity:.2f}, lambda={lambda_param:.2f})"
        )

        return reranked_results

    def _check_bm25_backend(self) -> None:
        """Check if BM25 index exists and load it.

        This enables hybrid search when BM25 index is available.
        Falls back gracefully to vector-only if BM25 is missing.

        If BM25 index doesn't exist but LanceDB does, attempts lazy build.
        """
        try:
            # Detect BM25 index path from database persist_directory
            if hasattr(self.database, "persist_directory"):
                index_path = self.database.persist_directory
                bm25_path = index_path / "bm25_index.pkl"

                if bm25_path.exists():
                    # Load BM25 index
                    self._bm25_backend = BM25Backend()
                    self._bm25_backend.load(bm25_path)
                    logger.debug(
                        f"BM25 index loaded from {bm25_path} "
                        f"({self._bm25_backend.get_stats()['chunk_count']} chunks)"
                    )
                else:
                    # BM25 index missing: try lazy build if LanceDB exists
                    logger.debug(
                        f"BM25 index not found at {bm25_path}, attempting lazy build..."
                    )
                    if self._try_lazy_build_bm25(bm25_path):
                        logger.info(
                            "BM25 index built successfully (lazy initialization)"
                        )
                    else:
                        logger.debug("BM25 lazy build failed, hybrid search disabled")
            else:
                logger.debug("Database has no persist_directory, BM25 disabled")
        except Exception as e:
            logger.debug(f"BM25 backend detection failed: {e}")
            self._bm25_backend = None

    def _try_lazy_build_bm25(self, bm25_path: Path) -> bool:
        """Attempt to build BM25 index from existing vectors.lance data.

        Args:
            bm25_path: Path to save BM25 index

        Returns:
            True if build succeeded, False otherwise
        """
        try:
            # Directly open LanceDB vectors.lance table
            # This avoids complex async initialization of VectorsBackend
            if not hasattr(self.database, "persist_directory"):
                logger.debug("Database has no persist_directory")
                return False

            index_path = self.database.persist_directory

            # Handle both old (index_path) and new (index_path/lance) paths
            if index_path.name == "lance":
                lance_path = index_path
            else:
                lance_path = index_path / "lance"

            vectors_table_path = lance_path / "vectors.lance"
            if not vectors_table_path.exists():
                logger.debug(f"vectors.lance not found at {vectors_table_path}")
                return False

            # Open LanceDB table directly
            import lancedb

            db = lancedb.connect(str(lance_path))
            table = db.open_table("vectors")

            # Query all records
            df = table.to_pandas()

            if df.empty:
                logger.debug("Vectors table is empty, cannot build BM25 index")
                return False

            # Convert DataFrame rows to dicts for BM25Backend
            # vectors.lance schema: chunk_id, vector, file_path, content, language,
            #                       start_line, end_line, chunk_type, name, hierarchy_path
            chunks_for_bm25 = []
            for _, row in df.iterrows():
                chunk_dict = {
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "name": row.get("name", ""),  # vectors.lance uses "name" field
                    "file_path": row["file_path"],
                    "chunk_type": row.get("chunk_type", "code"),
                }
                chunks_for_bm25.append(chunk_dict)

            # Build and save BM25 index
            self._bm25_backend = BM25Backend()
            self._bm25_backend.build_index(chunks_for_bm25)
            self._bm25_backend.save(bm25_path)

            logger.info(
                f"Built BM25 index with {len(chunks_for_bm25)} chunks (lazy initialization)"
            )
            return True

        except Exception as e:
            logger.debug(f"Lazy BM25 build failed: {e}")
            self._bm25_backend = None
            return False

    async def _search_bm25(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        threshold: float,
    ) -> list[SearchResult]:
        """Pure BM25 keyword search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters
            threshold: Minimum similarity threshold (unused for BM25, kept for API compatibility)

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If BM25 backend not available or search fails
        """
        if not self._bm25_backend or not self._bm25_backend.is_built():
            raise SearchError(
                "BM25 index not available. Run 'mcp-vector-search index' to build it."
            )

        try:
            # Get BM25 results (returns list of (chunk_id, score) tuples)
            bm25_results = self._bm25_backend.search(query, limit=limit * 2)

            if not bm25_results:
                return []

            # Convert BM25 results to SearchResult objects
            # Need to fetch chunk metadata from vectors backend
            if not self._vectors_backend:
                raise SearchError("Vector backend not available for metadata lookup")

            # Initialize vectors backend if needed
            if self._vectors_backend._db is None:
                await self._vectors_backend.initialize()

            # Fetch chunks by chunk_id
            search_results = []
            for chunk_id, bm25_score in bm25_results[:limit]:
                # Query vectors table for chunk metadata
                try:
                    # Use chunk_id to fetch metadata
                    df = self._vectors_backend._table.to_pandas().query(
                        f"chunk_id == '{chunk_id}'"
                    )
                    if df.empty:
                        continue

                    row = df.iloc[0]

                    # Create SearchResult with BM25 score as similarity
                    # Normalize BM25 score to 0-1 range (heuristic: divide by 10, cap at 1.0)
                    normalized_score = min(1.0, bm25_score / 10.0)

                    search_result = SearchResult(
                        file_path=Path(row["file_path"]),
                        content=row["content"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        language=row.get("language", "unknown"),
                        similarity_score=normalized_score,
                        rank=len(search_results) + 1,
                        chunk_type=row.get("chunk_type", "unknown"),
                        function_name=row.get("name"),
                    )
                    search_results.append(search_result)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch metadata for chunk {chunk_id}: {e}"
                    )
                    continue

            logger.debug(
                f"BM25 search for '{query}' returned {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise SearchError(f"BM25 search failed: {e}") from e

    async def _search_hybrid(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        threshold: float,
        hybrid_alpha: float,
    ) -> list[SearchResult]:
        """Hybrid search using Reciprocal Rank Fusion (RRF) of vector + BM25 results.

        RRF formula: score(d) = sum(1 / (k + rank_i)) for each retrieval method i
        where k=60 is a smoothing constant (default in literature).

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters
            threshold: Minimum similarity threshold for vector search
            hybrid_alpha: Weight for vector search (0.0-1.0)
                         1.0 = pure vector, 0.0 = pure BM25, 0.7 = balanced with vector preference

        Returns:
            List of SearchResult objects sorted by RRF score
        """
        # Fallback to vector-only if BM25 not available
        if not self._bm25_backend or not self._bm25_backend.is_built():
            logger.warning(
                "BM25 index not available, falling back to vector-only search. "
                "Run 'mcp-vector-search index' to build BM25 index for hybrid search."
            )
            if self._vectors_backend:
                return await self._search_vectors_backend(
                    query, limit, filters, threshold
                )
            else:
                return await self._retry_handler.search_with_retry(
                    database=self.database,
                    query=query,
                    limit=limit,
                    filters=filters,
                    threshold=threshold,
                )

        try:
            # Get vector results
            if self._vectors_backend:
                vector_results = await self._search_vectors_backend(
                    query, limit * 2, filters, threshold
                )
            else:
                vector_results = await self._retry_handler.search_with_retry(
                    database=self.database,
                    query=query,
                    limit=limit * 2,
                    filters=filters,
                    threshold=threshold,
                )

            # Get BM25 results
            bm25_results = self._bm25_backend.search(query, limit=limit * 2)

            # Build maps: chunk_id -> rank (1-based)
            vector_ranks = {
                self._get_chunk_id(result): i + 1
                for i, result in enumerate(vector_results)
            }

            bm25_ranks = {
                chunk_id: i + 1 for i, (chunk_id, _) in enumerate(bm25_results)
            }

            # Compute RRF scores with weighted fusion
            rrf_scores: dict[str, float] = {}

            # Combine ranks from both methods
            all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

            for chunk_id in all_chunk_ids:
                vector_rank = vector_ranks.get(chunk_id, float("inf"))
                bm25_rank = bm25_ranks.get(chunk_id, float("inf"))

                # RRF score with weighted fusion
                # Apply alpha weighting: higher alpha = more vector influence
                vector_score = (
                    hybrid_alpha / (RRF_K + vector_rank)
                    if vector_rank != float("inf")
                    else 0.0
                )
                bm25_score = (
                    (1.0 - hybrid_alpha) / (RRF_K + bm25_rank)
                    if bm25_rank != float("inf")
                    else 0.0
                )

                rrf_scores[chunk_id] = vector_score + bm25_score

            # Normalize RRF scores to 0.0-1.0 range for consistent display
            if rrf_scores:
                max_rrf = max(rrf_scores.values())
                if max_rrf > 0:
                    rrf_scores = {
                        cid: score / max_rrf for cid, score in rrf_scores.items()
                    }

            # Sort by RRF score descending
            sorted_chunk_ids = sorted(
                rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
            )

            # Build SearchResult list from sorted chunk IDs
            # Use vector_results as source since they have full metadata
            chunk_id_to_result = {self._get_chunk_id(r): r for r in vector_results}

            hybrid_results = []
            for i, chunk_id in enumerate(sorted_chunk_ids[:limit]):
                if chunk_id in chunk_id_to_result:
                    result = chunk_id_to_result[chunk_id]
                    # Update similarity score to RRF score
                    result.similarity_score = rrf_scores[chunk_id]
                    result.rank = i + 1
                    hybrid_results.append(result)
                else:
                    # Chunk found in BM25 but not in vector results
                    # Fetch metadata from vectors backend
                    try:
                        if self._vectors_backend and self._vectors_backend._table:
                            df = self._vectors_backend._table.to_pandas().query(
                                f"chunk_id == '{chunk_id}'"
                            )
                            if not df.empty:
                                row = df.iloc[0]
                                search_result = SearchResult(
                                    file_path=Path(row["file_path"]),
                                    content=row["content"],
                                    start_line=row["start_line"],
                                    end_line=row["end_line"],
                                    language=row.get("language", "unknown"),
                                    similarity_score=rrf_scores[chunk_id],
                                    rank=i + 1,
                                    chunk_type=row.get("chunk_type", "unknown"),
                                    function_name=row.get("name"),
                                )
                                hybrid_results.append(search_result)
                    except Exception as e:
                        logger.debug(
                            f"Failed to fetch metadata for BM25-only chunk {chunk_id}: {e}"
                        )

            logger.info(
                f"Hybrid search (α={hybrid_alpha:.2f}) for '{query}' combined "
                f"{len(vector_results)} vector + {len(bm25_results)} BM25 results → "
                f"{len(hybrid_results)} final results"
            )

            return hybrid_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}") from e

    @staticmethod
    def _get_chunk_id(result: SearchResult) -> str:
        """Get chunk_id from SearchResult.

        Args:
            result: SearchResult object

        Returns:
            Chunk ID (file_path:start_line-end_line format)
        """
        # Check if result has chunk_id attribute (from vectors backend)
        if hasattr(result, "chunk_id") and result.chunk_id:
            return result.chunk_id

        # Fallback: construct chunk_id from file path and line range
        return f"{result.file_path}:{result.start_line}-{result.end_line}"

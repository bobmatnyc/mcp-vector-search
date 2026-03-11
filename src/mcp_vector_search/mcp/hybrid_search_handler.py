"""Hybrid search MCP handler — fuses semantic, BM25, and KG results via RRF.

This module implements the ``search_hybrid`` MCP tool, which runs three
complementary search strategies in parallel and synthesises them into a
single ranked list using Reciprocal Rank Fusion (RRF).

Strategies
----------
1. **Semantic** — dense-vector similarity via the existing
   ``SemanticSearchEngine.search(SearchMode.VECTOR)``.
2. **Text** — BM25 keyword search via
   ``SemanticSearchEngine.search(SearchMode.BM25)``.
3. **KG** — knowledge-graph entity traversal: find entities whose names
   match query terms, then collect their ``file_path`` values and boost
   those files in the fusion step.

Fusion
------
Results from all strategies are deduplicated by ``(file_path, start_line)``
and scored with the standard RRF formula::

    rrf_score = sum(1.0 / (k + rank_i))   for k = 60

The fused list is sorted descending by RRF score and truncated to the
requested ``limit``.

Partial failure handling
------------------------
If any strategy fails (e.g. KG not built, BM25 index missing), a warning
is logged and the fusion continues with the remaining strategies.  Partial
results are always better than a total failure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..core.knowledge_graph import KnowledgeGraph
from ..core.models import SearchResult
from ..core.search import SearchMode, SemanticSearchEngine

# Standard RRF smoothing constant (k=60 is the literature default).
_RRF_K: int = 60

# Strategy names exposed in the output schema.
_STRATEGY_SEMANTIC = "semantic"
_STRATEGY_TEXT = "text"
_STRATEGY_KG = "kg"
_ALL_STRATEGIES = [_STRATEGY_SEMANTIC, _STRATEGY_TEXT, _STRATEGY_KG]


@dataclass
class _CandidateResult:
    """Internal representation of a deduplicated result during RRF fusion."""

    file_path: str
    content: str
    line_start: int | None
    line_end: int | None
    chunk_type: str | None
    strategies_found: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    text_score: float | None = None
    kg_score: float | None = None
    rrf_score: float = 0.0

    @property
    def dedup_key(self) -> tuple[str, int | None]:
        """Stable deduplication key: (file_path, start_line)."""
        return (self.file_path, self.line_start)


class HybridSearchHandlers:
    """MCP handler for the ``search_hybrid`` tool."""

    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        project_root: Path,
    ) -> None:
        """Initialise handler.

        Args:
            search_engine: Shared ``SemanticSearchEngine`` instance.
            project_root: Absolute path to the project root; used to
                locate the KG database directory.
        """
        self.search_engine = search_engine
        self.project_root = project_root

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_search_hybrid(self, args: dict[str, Any]) -> CallToolResult:
        """Handle the ``search_hybrid`` MCP tool call.

        Args:
            args: Tool arguments from the MCP caller.

        Returns:
            ``CallToolResult`` whose content is a JSON-encoded response.
        """
        query: str = args.get("query", "").strip()
        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="query parameter is required")],
                isError=True,
            )

        limit: int = int(args.get("limit", 10))
        requested_strategies: list[str] = list(args.get("strategies", _ALL_STRATEGIES))
        # Guard against unknown strategy names.
        active_strategies = [s for s in requested_strategies if s in _ALL_STRATEGIES]
        if not active_strategies:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"No valid strategies requested. Valid values: {_ALL_STRATEGIES}",
                    )
                ],
                isError=True,
            )

        # Candidate pool size: each strategy retrieves up to limit*3 results
        # so the RRF fusion has a rich pool to work with.
        candidate_limit = limit * 3

        # ------------------------------------------------------------------
        # Run all strategies concurrently; failures are caught per-strategy.
        # ------------------------------------------------------------------
        semantic_task = (
            self._run_semantic(query, candidate_limit)
            if _STRATEGY_SEMANTIC in active_strategies
            else self._noop()
        )
        text_task = (
            self._run_text(query, candidate_limit)
            if _STRATEGY_TEXT in active_strategies
            else self._noop()
        )
        kg_task = (
            self._run_kg(query, candidate_limit)
            if _STRATEGY_KG in active_strategies
            else self._noop()
        )

        semantic_results, text_results, kg_results = await asyncio.gather(
            semantic_task, text_task, kg_task
        )

        strategy_counts: dict[str, int] = {
            _STRATEGY_SEMANTIC: len(semantic_results),
            _STRATEGY_TEXT: len(text_results),
            _STRATEGY_KG: len(kg_results),
        }

        # ------------------------------------------------------------------
        # Fuse results with RRF.
        # ------------------------------------------------------------------
        fused = self._fuse_rrf(
            semantic_results=semantic_results,
            text_results=text_results,
            kg_results=kg_results,
        )

        # Fix 3: Drop results below minimum confidence threshold.
        # 0.01 corresponds to ~rank 100 in a single strategy — too low to surface.
        min_rrf_score = 0.01
        fused = [c for c in fused if c.rrf_score >= min_rrf_score]

        total_before_dedup = len(semantic_results) + len(text_results) + len(kg_results)
        total_after_dedup = len(fused)

        # Fix 1: Build warnings for KG requested but contributing 0 results.
        warnings: list[str] = []
        if _STRATEGY_KG in active_strategies and strategy_counts[_STRATEGY_KG] == 0:
            kg_db_path = (
                self.project_root / ".mcp-vector-search" / "knowledge_graph" / "code_kg"
            )
            if not kg_db_path.exists():
                warnings.append(
                    "KG index not built — run kg_build first to enable KG strategy"
                )
            else:
                warnings.append("KG strategy returned no results for this query")

        # Apply final limit.
        top_results = fused[:limit]

        # ------------------------------------------------------------------
        # Serialise output.
        # ------------------------------------------------------------------
        output_results = [
            {
                "file_path": r.file_path,
                "content": r.content,
                "score": round(r.rrf_score, 6),
                "strategies": r.strategies_found,
                "semantic_score": r.semantic_score,
                "text_score": r.text_score,
                "kg_score": r.kg_score,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "chunk_type": r.chunk_type,
            }
            for r in top_results
        ]

        response_payload = {
            "results": output_results,
            "query": query,
            "strategy_counts": {
                **strategy_counts,
                "total_before_dedup": total_before_dedup,
                "total_after_dedup": total_after_dedup,
            },
            "warnings": warnings,
        }

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(response_payload, indent=2),
                )
            ],
            isError=False,
        )

    # ------------------------------------------------------------------
    # Per-strategy runners
    # ------------------------------------------------------------------

    async def _run_semantic(self, query: str, limit: int) -> list[SearchResult]:
        """Run dense-vector semantic search.

        Returns an empty list (not an exception) on failure so the fusion
        can continue with partial results.
        """
        try:
            results = await self.search_engine.search(
                query=query,
                limit=limit,
                search_mode=SearchMode.VECTOR,
                use_rerank=False,  # Skip heavy reranking; RRF handles ranking.
                use_mmr=False,  # Skip MMR; fusion provides diversity.
                expand=False,  # No query expansion; keep strategy pure.
            )
            logger.debug(
                "search_hybrid semantic: %d results for %r", len(results), query
            )
            return results
        except Exception as exc:
            logger.warning(
                "search_hybrid: semantic strategy failed (partial results): %s", exc
            )
            return []

    async def _run_text(self, query: str, limit: int) -> list[SearchResult]:
        """Run BM25 full-text keyword search.

        Returns an empty list (not an exception) on failure so the fusion
        can continue with partial results.
        """
        try:
            results = await self.search_engine.search(
                query=query,
                limit=limit,
                search_mode=SearchMode.BM25,
                use_rerank=False,
                use_mmr=False,
                expand=False,
            )
            logger.debug(
                "search_hybrid text/BM25: %d results for %r", len(results), query
            )
            return results
        except Exception as exc:
            logger.warning(
                "search_hybrid: text strategy failed (partial results): %s", exc
            )
            return []

    async def _run_kg(self, query: str, limit: int) -> list[SearchResult]:
        """Run knowledge-graph entity search.

        Finds KG entities whose names match query terms and returns
        synthetic ``SearchResult`` objects pointing to those files.

        The KG does not return chunk-level scores; all KG hits receive a
        uniform ``kg_score`` of ``1.0`` so the RRF formula treats every
        KG match as a rank-1 hit from that strategy.  This is intentional:
        the KG is used as a precision signal ("this file is structurally
        related"), not a relevance scorer.

        Returns an empty list (not an exception) on failure so the fusion
        can continue with partial results.
        """
        try:
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg_db_path = kg_path / "code_kg"

            if not kg_db_path.exists():
                logger.debug("search_hybrid: KG not built yet, skipping KG strategy")
                return []

            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            try:
                stats = await kg.get_stats()
                if stats.get("total_entities", 0) == 0:
                    logger.debug("search_hybrid: KG is empty, skipping KG strategy")
                    return []

                # Extract meaningful terms from the query: skip stop-words
                # and very short tokens that would match too broadly.
                stop_words = {
                    "a",
                    "an",
                    "the",
                    "is",
                    "in",
                    "at",
                    "of",
                    "on",
                    "to",
                    "and",
                    "or",
                    "for",
                    "with",
                    "by",
                    "from",
                    "as",
                    "be",
                    "this",
                    "that",
                    "it",
                    "its",
                    "are",
                    "was",
                    "were",
                }
                query_terms = [
                    t
                    for t in query.lower().split()
                    if t not in stop_words and len(t) >= 3
                ]

                if not query_terms:
                    return []

                # Collect file_path hits from entity lookups.
                seen_files: set[str] = set()
                kg_results: list[SearchResult] = []

                for term in query_terms[:5]:  # Cap at 5 terms to avoid KG overload.
                    entity_id = await kg.find_entity_by_name(term)
                    if not entity_id:
                        continue

                    related = await kg.find_related(entity_id, max_hops=1)
                    for rel in related:
                        file_path = rel.get("file_path")
                        if not file_path or file_path in seen_files:
                            continue
                        seen_files.add(file_path)

                        # Build a minimal SearchResult for this KG hit.
                        # Content is left empty — it will be set from the
                        # semantic/text result if the same chunk is found
                        # there; otherwise the consumer sees the file path.
                        kg_result = SearchResult(
                            file_path=Path(file_path),
                            content="",
                            start_line=0,
                            end_line=0,
                            language="",
                            similarity_score=1.0,
                            rank=len(kg_results) + 1,
                            chunk_type="kg_entity",
                        )
                        kg_results.append(kg_result)

                        if len(kg_results) >= limit:
                            break

                    if len(kg_results) >= limit:
                        break

                logger.debug(
                    "search_hybrid KG: %d file hits for %r", len(kg_results), query
                )
                return kg_results

            finally:
                await kg.close()

        except Exception as exc:
            logger.warning(
                "search_hybrid: KG strategy failed (partial results): %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _fuse_rrf(
        self,
        semantic_results: list[SearchResult],
        text_results: list[SearchResult],
        kg_results: list[SearchResult],
    ) -> list[_CandidateResult]:
        """Merge results from all strategies using Reciprocal Rank Fusion.

        Deduplication key is ``(file_path, start_line)``.  The result with
        the richest ``content`` (longest string) wins when the same chunk
        appears in multiple strategies — this ensures KG stubs (empty
        content) do not shadow semantic/text hits for the same file.

        Args:
            semantic_results: Ranked list from the semantic strategy.
            text_results: Ranked list from the BM25 text strategy.
            kg_results: Ranked list from the KG strategy.

        Returns:
            Deduplicated list sorted by RRF score descending.
        """
        # Map dedup_key -> _CandidateResult
        candidates: dict[tuple[str, int | None], _CandidateResult] = {}

        def _process(
            results: list[SearchResult],
            strategy_name: str,
            score_attr: str,
        ) -> None:
            for rank_0based, result in enumerate(results):
                rank = rank_0based + 1  # 1-based rank for RRF
                rrf_contribution = 1.0 / (_RRF_K + rank)

                file_path_str = str(result.file_path)
                start_line: int | None = (
                    result.start_line if result.start_line != 0 else None
                )
                key: tuple[str, int | None] = (file_path_str, start_line)

                if key not in candidates:
                    candidates[key] = _CandidateResult(
                        file_path=file_path_str,
                        content=result.content,
                        line_start=start_line,
                        line_end=result.end_line if result.end_line != 0 else None,
                        chunk_type=result.chunk_type or None,
                    )

                candidate = candidates[key]

                # Keep the richest content (longest string wins over KG stubs).
                if len(result.content) > len(candidate.content):
                    candidate.content = result.content
                    candidate.line_start = start_line
                    candidate.line_end = (
                        result.end_line if result.end_line != 0 else None
                    )
                    if result.chunk_type and result.chunk_type != "kg_entity":
                        candidate.chunk_type = result.chunk_type

                # Accumulate RRF score.
                candidate.rrf_score += rrf_contribution

                # Record which strategy found this result.
                if strategy_name not in candidate.strategies_found:
                    candidate.strategies_found.append(strategy_name)

                # Store the raw score for that strategy.
                setattr(candidate, score_attr, result.similarity_score)

        _process(semantic_results, _STRATEGY_SEMANTIC, "semantic_score")
        _process(text_results, _STRATEGY_TEXT, "text_score")
        _process(kg_results, _STRATEGY_KG, "kg_score")

        # Boost source code results over documentation.
        # Counteracts doc-dominance where research docs outrank the source they describe.
        for candidate in candidates.values():
            if "/src/" in candidate.file_path or candidate.file_path.startswith("src/"):
                candidate.rrf_score *= 1.15

        # Sort by RRF score descending.
        return sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _noop() -> list[SearchResult]:
        """Async no-op placeholder for disabled strategies."""
        return []

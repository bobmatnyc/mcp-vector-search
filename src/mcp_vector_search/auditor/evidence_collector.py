"""Evidence collector for the privacy-policy auditor.

Task 6: Executes QueryPlans against the target repo's vector index and knowledge
graph, returning deduplicated Evidence objects.

Calls the underlying Python APIs directly — NOT via MCP — using the same
internal modules the MCP server uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from .claim_router import QueryPlan
from .models import Evidence

# Module-level flag: emit the KG-unavailable warning at most once per process.
_kg_unavailable_warned = False

# ---------------------------------------------------------------------------
# Search engine / KG bootstrap helpers
# ---------------------------------------------------------------------------


async def _get_search_engine(target_repo: Path):
    """Bootstrap a SemanticSearchEngine for the target repo.

    Returns None if the index does not exist yet (logs a warning).
    """
    try:
        from ..core.embeddings import create_embedding_function
        from ..core.factory import create_database
        from ..core.project import ProjectManager
        from ..core.search import SemanticSearchEngine

        project_manager = ProjectManager(target_repo)
        config = project_manager.load_config()

        embedding_function, _ = create_embedding_function(
            model_name=config.embedding_model
        )
        database = create_database(
            persist_directory=config.index_path / "lance",
            embedding_function=embedding_function,
        )
        await database.__aenter__()

        chunk_count = database.get_chunk_count()
        if chunk_count == 0:
            logger.warning(
                "Target repo %s has no vector index — skipping vector search tools. "
                "Run 'mvs index' in the target repo first.",
                target_repo,
            )
            return None

        engine = SemanticSearchEngine(
            database=database,
            project_root=target_repo,
            enable_auto_reindex=False,
        )
        return engine

    except Exception as exc:
        logger.warning(
            "Failed to initialise search engine for %s: %s — skipping vector tools",
            target_repo,
            exc,
        )
        return None


async def _get_knowledge_graph(target_repo: Path):
    """Bootstrap a KnowledgeGraph for the target repo.

    Returns None if the KG does not exist yet (logs a warning the first time
    only — subsequent calls for the same run are silent).
    """
    global _kg_unavailable_warned

    try:
        from ..core.knowledge_graph import KnowledgeGraph

        kg_path = target_repo / ".mcp-vector-search" / "knowledge_graph"
        if not kg_path.exists():
            if not _kg_unavailable_warned:
                logger.warning(
                    "Knowledge graph not available for %s. "
                    "KG-based evidence collection will be skipped. "
                    "Build one with: mvs index --project-root %s",
                    target_repo,
                    target_repo,
                )
                _kg_unavailable_warned = True
            return None

        kg = KnowledgeGraph(kg_path)
        await kg.initialize()
        return kg

    except Exception as exc:
        if not _kg_unavailable_warned:
            logger.warning(
                "Failed to initialise knowledge graph for %s: %s — skipping KG tools",
                target_repo,
                exc,
            )
            _kg_unavailable_warned = True
        return None


# ---------------------------------------------------------------------------
# Tool-specific result adapters
# ---------------------------------------------------------------------------


def _results_to_evidence(
    results: list[Any],
    tool: str,
    query: str,
) -> list[Evidence]:
    """Convert SemanticSearchEngine search results to Evidence objects.

    Args:
        results: List of SearchResult objects.
        tool: Tool name that produced the results.
        query: The query that was run.

    Returns:
        List of Evidence objects.
    """
    evidence_list: list[Evidence] = []
    for result in results:
        file_path = getattr(result, "file_path", "") or ""
        start_line = getattr(result, "start_line", 0) or 0
        end_line = getattr(result, "end_line", 0) or 0
        snippet = getattr(result, "content", "") or ""
        score = float(getattr(result, "score", 0.0) or 0.0)

        if not file_path:
            continue

        evidence_list.append(
            Evidence(
                tool=tool,  # type: ignore[arg-type]
                query=query,
                file_path=str(file_path),
                start_line=int(start_line),
                end_line=int(end_line),
                snippet=snippet[:2000],  # cap snippet length
                score=score,
                kg_path=None,
            )
        )
    return evidence_list


async def _run_search_code(
    engine,
    plan: QueryPlan,
) -> list[Evidence]:
    """Execute a search_code query via the SemanticSearchEngine."""
    from ..core.search import SearchMode

    try:
        limit = plan.params.get("limit", 10)
        results = await engine.search(
            query=plan.query,
            limit=limit,
            search_mode=SearchMode.VECTOR,
            use_rerank=False,
        )
        return _results_to_evidence(results, "search_code", plan.query)
    except Exception as exc:
        logger.warning("search_code failed for query '%s': %s", plan.query, exc)
        return []


async def _run_search_hybrid(
    engine,
    plan: QueryPlan,
) -> list[Evidence]:
    """Execute a search_hybrid query via the SemanticSearchEngine."""
    from ..core.search import SearchMode

    try:
        limit = plan.params.get("limit", 10)
        results = await engine.search(
            query=plan.query,
            limit=limit,
            search_mode=SearchMode.HYBRID,
            use_rerank=False,
        )
        return _results_to_evidence(results, "search_hybrid", plan.query)
    except Exception as exc:
        logger.warning("search_hybrid failed for query '%s': %s", plan.query, exc)
        return []


async def _run_kg_query(
    kg,
    plan: QueryPlan,
) -> list[Evidence]:
    """Execute a kg_query by searching KG entity names for matching terms."""
    try:
        # Search for entities whose names contain the query keywords
        query_lower = plan.query.lower()
        words = [w for w in query_lower.split() if len(w) > 3][:5]

        evidence_list: list[Evidence] = []
        seen_paths: set[str] = set()

        for word in words:
            try:
                cypher = (
                    "MATCH (e:CodeEntity) "
                    "WHERE lower(e.name) CONTAINS $word "
                    "RETURN e.name, e.file_path, e.entity_type "
                    "LIMIT 5"
                )
                result = kg._execute_query(cypher, {"word": word})
                while result.has_next():
                    row = result.get_next()
                    entity_name = row[0] if row[0] else ""
                    file_path = row[1] if row[1] else ""
                    entity_type = row[2] if row[2] else ""

                    if not file_path or file_path in seen_paths:
                        continue
                    seen_paths.add(file_path)

                    # Try to find callers for context — build kg_path
                    kg_path = _find_kg_path(kg, entity_name)

                    evidence_list.append(
                        Evidence(
                            tool="kg_query",
                            query=plan.query,
                            file_path=str(file_path),
                            start_line=0,
                            end_line=0,
                            snippet=f"[{entity_type}] {entity_name}",
                            score=0.7,
                            kg_path=kg_path,
                        )
                    )
            except Exception as exc:
                logger.debug("KG word query '%s' failed: %s", word, exc)

        return evidence_list

    except Exception as exc:
        logger.warning("kg_query failed for query '%s': %s", plan.query, exc)
        return []


def _find_kg_path(kg, entity_name: str) -> list[str] | None:
    """Find a simple call path to/from the entity in the KG."""
    try:
        cypher = (
            "MATCH (a:CodeEntity)-[:CALLS]->(b:CodeEntity) "
            "WHERE b.name = $name "
            "RETURN a.name, b.name "
            "LIMIT 3"
        )
        result = kg._execute_query(cypher, {"name": entity_name})
        path_parts: list[str] = []
        while result.has_next():
            row = result.get_next()
            caller = row[0] or ""
            callee = row[1] or ""
            if caller and callee:
                path_parts.append(f"{caller} -> {callee}")
        return path_parts if path_parts else None
    except Exception:
        return None


async def _run_find_smells(
    engine,
    plan: QueryPlan,
) -> list[Evidence]:
    """Execute a find_smells-style query via hybrid search for smell patterns."""
    # find_smells is implemented as a hybrid search in M1
    # Full smell detector integration is deferred to M2
    try:
        from ..core.search import SearchMode

        limit = plan.params.get("limit", 10)
        results = await engine.search(
            query=plan.query,
            limit=limit,
            search_mode=SearchMode.HYBRID,
            use_rerank=False,
        )
        evidence = _results_to_evidence(results, "find_smells", plan.query)
        return evidence
    except Exception as exc:
        logger.warning("find_smells failed for query '%s': %s", plan.query, exc)
        return []


# ---------------------------------------------------------------------------
# Dedup helper
# ---------------------------------------------------------------------------


def _dedup_evidence(evidence_list: list[Evidence]) -> list[Evidence]:
    """Deduplicate evidence by (file_path, start_line, end_line).

    When duplicates exist, keep the one with the highest score.
    """
    seen: dict[tuple[str, int, int], Evidence] = {}
    for ev in evidence_list:
        key = (ev.file_path, ev.start_line, ev.end_line)
        existing = seen.get(key)
        if existing is None or ev.score > existing.score:
            seen[key] = ev
    return list(seen.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def collect(target_repo: Path, plans: list[QueryPlan]) -> list[Evidence]:
    """Collect evidence for a list of QueryPlans.

    Dispatches each plan to the appropriate tool, deduplicates results,
    and returns a flat list of Evidence objects.

    Args:
        target_repo: Root directory of the repository being audited.
        plans: List of QueryPlans to execute.

    Returns:
        Deduplicated list of Evidence objects.
    """
    if not plans:
        return []

    # Lazy-init search engine and KG (only once per collect call)
    engine = None
    kg = None

    engine_attempted = False
    kg_attempted = False

    all_evidence: list[Evidence] = []

    for plan in plans:
        tool = plan.tool

        if tool in ("search_code", "search_hybrid", "find_smells"):
            if not engine_attempted:
                engine = await _get_search_engine(target_repo)
                engine_attempted = True
            if engine is None:
                logger.debug("Skipping %s — no search engine available", tool)
                continue

            if tool == "search_code":
                results = await _run_search_code(engine, plan)
            elif tool == "search_hybrid":
                results = await _run_search_hybrid(engine, plan)
            else:
                results = await _run_find_smells(engine, plan)
            all_evidence.extend(results)

        elif tool in ("kg_query", "kg_callers_at_commit", "trace_execution_flow"):
            if not kg_attempted:
                kg = await _get_knowledge_graph(target_repo)
                kg_attempted = True
            if kg is None:
                logger.debug("Skipping %s — no knowledge graph available", tool)
                continue

            results = await _run_kg_query(kg, plan)
            all_evidence.extend(results)

        else:
            logger.warning("Unknown tool '%s' in QueryPlan — skipping", tool)

    deduped = _dedup_evidence(all_evidence)
    logger.debug(
        "collect(): %d plans → %d raw evidence → %d after dedup",
        len(plans),
        len(all_evidence),
        len(deduped),
    )
    return deduped

"""MCP handlers for knowledge graph functionality."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..core.embeddings import create_embedding_function
from ..core.factory import create_database
from ..core.kg_builder import KGBuilder
from ..core.knowledge_graph import KnowledgeGraph
from ..core.project import ProjectManager


class KGHandlers:
    """MCP handlers for knowledge graph operations."""

    def __init__(self, project_root: Path):
        """Initialize KG handlers.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root

    async def handle_kg_build(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_build tool call.

        Args:
            args: Tool arguments containing:
                - force (bool): Force rebuild even if graph exists
                - skip_documents (bool): Skip DOCUMENTS extraction (faster)
                - limit (int | None): Limit chunks to process (for testing)

        Returns:
            CallToolResult with build statistics
        """
        try:
            force = args.get("force", False)
            skip_documents = args.get("skip_documents", False)
            limit = args.get("limit")

            # Load project configuration
            project_manager = ProjectManager(self.project_root)
            config = project_manager.load_config()

            # Setup embedding function and database
            embedding_function, _ = create_embedding_function(
                model_name=config.embedding_model
            )
            database = create_database(
                persist_directory=config.index_path / "lance",
                embedding_function=embedding_function,
            )

            # Initialize database
            await database.__aenter__()

            try:
                # Check if index exists
                chunk_count = database.get_chunk_count()
                if chunk_count == 0:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text="No index found. Run 'index_project' first.",
                            )
                        ],
                        isError=True,
                    )

                # Initialize knowledge graph
                kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
                kg = KnowledgeGraph(kg_path)
                await kg.initialize()

                # Check if graph already exists
                stats = await kg.get_stats()
                if stats["total_entities"] > 0 and not force:
                    await kg.close()
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Knowledge graph already exists "
                                f"({stats['total_entities']} entities). "
                                "Use force=true to rebuild.",
                            )
                        ],
                        isError=False,
                    )

                # Build graph
                builder = KGBuilder(kg, self.project_root)
                build_stats = await builder.build_from_database(
                    database,
                    show_progress=False,
                    limit=limit,
                    skip_documents=skip_documents,
                )

                # Close connections
                await kg.close()

                # Format response
                result = {
                    "status": "success",
                    "statistics": {
                        "entities": build_stats["entities"],
                        "doc_sections": build_stats.get("doc_sections", 0),
                        "documents": build_stats.get("doc_nodes", 0),
                        "relationships": {
                            "calls": build_stats["calls"],
                            "imports": build_stats["imports"],
                            "inherits": build_stats["inherits"],
                            "contains": build_stats["contains"],
                        },
                        "total_relationships": sum(build_stats.values())
                        - build_stats["entities"],
                    },
                }

                return CallToolResult(
                    content=[
                        TextContent(type="text", text=json.dumps(result, indent=2))
                    ],
                    isError=False,
                )

            finally:
                # Cleanup database
                await database.__aexit__(None, None, None)

        except Exception as e:
            logger.error(f"KG build failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Knowledge graph build failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_stats(self, _args: dict[str, Any]) -> CallToolResult:
        """Handle kg_stats tool call.

        Args:
            _args: Tool arguments (none required)

        Returns:
            CallToolResult with KG statistics
        """
        try:
            # Initialize knowledge graph
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Get statistics
            stats = await kg.get_stats()

            # Close connection
            await kg.close()

            # Format response
            result = {
                "status": "success",
                "statistics": {
                    "total_entities": stats["total_entities"],
                    "database_path": stats["database_path"],
                    "relationships": stats.get("relationships", {}),
                },
            }

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG stats failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get knowledge graph statistics: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_ontology(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_ontology tool call.

        Args:
            args: Tool arguments containing:
                - category (str | None): Optional document category filter

        Returns:
            CallToolResult with document ontology tree
        """
        try:
            category = args.get("category")

            # Initialize knowledge graph
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Get ontology data
            ontology = await kg.get_document_ontology(category=category)

            # Close connection
            await kg.close()

            # Format response
            result = {
                "status": "success",
                "ontology": ontology,
            }

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG ontology failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get document ontology: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_ia(self, _args: dict[str, Any]) -> CallToolResult:
        """Handle kg_ia tool call — return IA tree.

        Returns:
            CallToolResult with Information Architecture tree
        """
        try:
            # Initialize knowledge graph
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Get IA tree data
            result = await kg.get_ia_tree()

            # Close connection
            await kg.close()

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str),
                    )
                ],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG IA tree failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get Information Architecture tree: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_query(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_query tool call.

        Args:
            args: Tool arguments containing:
                - entity (str): Entity name to query, OR a tag query in the
                  form "tag:<name>" / "tags:<name>" for tag-based doc lookup.
                  Comma-separated tags perform an AND query:
                  "tag:python,async" → docs that have BOTH tags.
                - query_type (str | None): Set to "tag" to force tag-query mode
                  regardless of the entity string format.
                - relationship (str | None): Relationship type filter
                  (ignored for tag queries).
                - limit (int): Max results (default: 20)

        Returns:
            CallToolResult with query results
        """
        try:
            entity = args.get("entity")
            relationship = args.get("relationship")
            limit = args.get("limit", 20)
            query_type = args.get("query_type", "")

            if not entity:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Missing required parameter: entity",
                        )
                    ],
                    isError=True,
                )

            # --- Detect tag-query mode ---
            # Triggered by query_type="tag" OR entity prefixed with "tag:" / "tags:"
            tag_names: list[str] = []
            is_tag_query = query_type == "tag"
            if not is_tag_query:
                lower_entity = entity.lower()
                if lower_entity.startswith("tag:") or lower_entity.startswith("tags:"):
                    is_tag_query = True
                    # Strip prefix and split comma-separated tags
                    prefix_end = entity.index(":") + 1
                    raw_tags = entity[prefix_end:]
                    tag_names = [t.strip() for t in raw_tags.split(",") if t.strip()]
            if is_tag_query and not tag_names:
                # query_type="tag" path — use entity value directly as tag name(s)
                tag_names = [t.strip() for t in entity.split(",") if t.strip()]

            # Initialize knowledge graph
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Check if graph exists
            stats = await kg.get_stats()
            if stats["total_entities"] == 0:
                await kg.close()
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Knowledge graph is empty. Run 'kg_build' first.",
                        )
                    ],
                    isError=True,
                )

            # --- Tag query path ---
            if is_tag_query:
                tag_results = await kg.find_by_tag_docs(tag_names, limit=limit)
                await kg.close()

                if not tag_results:
                    result = {
                        "status": "success",
                        "query": {
                            "type": "tag",
                            "tags": tag_names,
                        },
                        "results": [],
                        "message": f"No documents found with tag(s): {tag_names}",
                    }
                else:
                    result = {
                        "status": "success",
                        "query": {
                            "type": "tag",
                            "tags": tag_names,
                        },
                        "results": tag_results,
                        "count": len(tag_results),
                    }

                return CallToolResult(
                    content=[
                        TextContent(type="text", text=json.dumps(result, indent=2))
                    ],
                    isError=False,
                )

            # --- Normal entity query path ---
            results = []

            if relationship in ["calls", "called_by"]:
                # Get call graph
                calls = await kg.get_call_graph(entity)
                if relationship == "calls":
                    results = [c for c in calls if c["direction"] == "calls"]
                else:  # called_by
                    results = [c for c in calls if c["direction"] == "called_by"]

            elif relationship in ["inherits", "inherited_by"]:
                # Get inheritance tree
                hierarchy = await kg.get_inheritance_tree(entity)
                if relationship == "inherits":
                    results = [h for h in hierarchy if h["relation"] == "parent"]
                else:  # inherited_by
                    results = [h for h in hierarchy if h["relation"] == "child"]

            elif relationship in ["imports", "imported_by", "contains", "contained_by"]:
                # For these relationships, use find_related with broader search
                related = await kg.find_related(entity, max_hops=1)
                results = related  # TODO: Filter by relationship type when available

            else:
                # No specific relationship, return all related entities
                results = await kg.find_related(entity, max_hops=2)

            # Close connection
            await kg.close()

            # Apply limit
            results = results[:limit]

            # Format response
            if not results:
                result = {
                    "status": "success",
                    "query": {
                        "entity": entity,
                        "relationship": relationship,
                    },
                    "results": [],
                    "message": f"No related entities found for '{entity}'",
                }
            else:
                result = {
                    "status": "success",
                    "query": {
                        "entity": entity,
                        "relationship": relationship,
                    },
                    "results": results,
                    "count": len(results),
                }

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG query failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Knowledge graph query failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_trace_execution_flow(self, args: dict[str, Any]) -> CallToolResult:
        """Handle trace_execution_flow tool call."""
        entry_point = args.get("entry_point", "")
        if not entry_point:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="entry_point parameter is required")
                ],
                isError=True,
            )

        depth = int(args.get("depth", 3))
        direction = args.get("direction", "outgoing")
        if direction not in ("outgoing", "incoming", "both"):
            direction = "outgoing"

        try:
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            result = await kg.trace_execution_flow(
                entry_point=entry_point,
                depth=depth,
                direction=direction,
            )

            await kg.close()
        except Exception as e:
            logger.error(f"trace_execution_flow failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Trace failed: {e}")],
                isError=True,
            )

        if result["entry"] is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"No entity found matching '{entry_point}'. Try a more specific name.",
                    )
                ],
                isError=True,
            )

        # Format as readable text + JSON summary
        entry = result["entry"]
        nodes = result["nodes"]
        edges = result["edges"]

        lines = [
            f"## Execution Flow: {entry['name']}",
            f"Entry: {entry['name']} ({entry.get('entity_type', 'function')}) "
            f"[{entry.get('file_path', '').split('/src/')[-1] if entry.get('file_path') else '?'}]",
            f"Direction: {direction} | Depth: {result['depth_reached']}/{depth} | "
            f"Nodes found: {result['total_nodes']}"
            + (" (truncated)" if result["truncated"] else ""),
            "",
        ]

        if not nodes:
            lines.append(
                f"No {'callees' if direction == 'outgoing' else 'callers'} found."
            )
        else:
            lines.append(f"### Reachable nodes ({len(nodes)}):")
            for node in sorted(nodes, key=lambda n: n["depth"]):
                indent = "  " * node["depth"]
                short_file = (node.get("file_path") or "?").split("/src/")[-1]
                lines.append(
                    f"{indent}[depth {node['depth']}] {node['name']} "
                    f"({node.get('entity_type', '?')}) [{short_file}]"
                )

            lines.append("")
            lines.append(f"### Call edges ({len(edges)}):")
            for edge in edges[:30]:
                lines.append(
                    f"  {edge['from_name']} → {edge['to_name']} (depth {edge['depth']})"
                )
            if len(edges) > 30:
                lines.append(f"  ... and {len(edges) - 30} more edges")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
        )

    async def handle_kg_history(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_history tool call.

        Returns the commit metadata stored in the KG for a named entity.
        V1 semantic: reflects the most recent commit at last kg_build time.

        Args:
            args: Tool arguments:
                - entity_name (str): Entity name to look up.

        Returns:
            CallToolResult with entity history records.
        """
        entity_name = args.get("entity_name", "")
        if not entity_name:
            return CallToolResult(
                content=[TextContent(type="text", text="entity_name is required")],
                isError=True,
            )

        try:
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            history = await kg.get_entity_history(entity_name)
            await kg.close()

            result = {
                "status": "success",
                "entity_name": entity_name,
                "history": history,
                "note": (
                    "V1: reflects the most recent commit per file at kg_build time, "
                    "not the full git log."
                ),
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"kg_history failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"kg_history failed: {e}")],
                isError=True,
            )

    async def handle_kg_callers_at_commit(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_callers_at_commit tool call.

        Returns callers whose stored commit_sha is an ancestor of the given commit.
        V1 semantic: reflects the most recent commit at last kg_build time.

        Args:
            args: Tool arguments:
                - entity_name (str): Name of the callee entity.
                - commit_sha (str): Reference git commit SHA.

        Returns:
            CallToolResult with caller records.
        """
        entity_name = args.get("entity_name", "")
        commit_sha = args.get("commit_sha", "")

        if not entity_name or not commit_sha:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="entity_name and commit_sha are required",
                    )
                ],
                isError=True,
            )

        try:
            kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            callers = await kg.get_callers_at_commit(
                entity_name, commit_sha, self.project_root
            )
            await kg.close()

            result = {
                "status": "success",
                "entity_name": entity_name,
                "commit_sha": commit_sha,
                "callers": callers,
                "note": (
                    "V1: reflects the most recent commit per file at kg_build time, "
                    "not the full git log."
                ),
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"kg_callers_at_commit failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"kg_callers_at_commit failed: {e}")
                ],
                isError=True,
            )

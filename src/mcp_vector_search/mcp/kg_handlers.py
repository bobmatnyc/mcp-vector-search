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
                persist_directory=config.index_path,
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

    async def handle_kg_stats(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_stats tool call.

        Args:
            args: Tool arguments (none required)

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

    async def handle_kg_query(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_query tool call.

        Args:
            args: Tool arguments containing:
                - entity (str): Entity name to query
                - relationship (str | None): Relationship type filter
                - limit (int): Max results (default: 20)

        Returns:
            CallToolResult with query results
        """
        try:
            entity = args.get("entity")
            relationship = args.get("relationship")
            limit = args.get("limit", 20)

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

            # Query based on relationship type
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

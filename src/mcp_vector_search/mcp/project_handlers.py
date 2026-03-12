"""Project management handlers for MCP vector search server."""

from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..core.exceptions import ProjectNotFoundError
from ..core.indexer import SemanticIndexer
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine
from ..utils.gitignore_updater import ensure_gitignore_entry


class ProjectHandlers:
    """Handlers for project management-related MCP tool operations."""

    def __init__(
        self,
        project_manager: ProjectManager,
        search_engine: SemanticSearchEngine | None,
        project_root: Path,
        indexer: SemanticIndexer | None = None,
    ):
        """Initialize project handlers.

        Args:
            project_manager: Project manager instance
            search_engine: Semantic search engine instance (or None if not initialized)
            project_root: Project root directory
            indexer: Semantic indexer instance (for KG status)
        """
        self.project_manager = project_manager
        self.search_engine = search_engine
        self.project_root = project_root
        self.indexer = indexer

    async def handle_get_project_status(self, args: dict[str, Any]) -> CallToolResult:
        """Handle get_project_status tool call.

        Args:
            args: Tool call arguments (unused)

        Returns:
            CallToolResult with project status or error
        """
        try:
            config = self.project_manager.load_config()

            # Get database stats
            if self.search_engine:
                stats = await self.search_engine.database.get_stats()

                # Query VectorsBackend separately to get Phase 2 (embedded vectors) count.
                # chunks_stored = Phase 1 complete; vectors_embedded = Phase 2 complete.
                vectors_embedded = 0
                lance_path = config.index_path / "lance"
                vectors_path = lance_path / "vectors.lance"
                if vectors_path.exists():
                    try:
                        from ..core.vectors_backend import VectorsBackend

                        vb = VectorsBackend(lance_path)
                        await vb.initialize()
                        vb_stats = await vb.get_stats()
                        vectors_embedded = vb_stats.get("total", 0)
                        await vb.close()
                    except Exception as e:
                        logger.debug(f"Could not read vectors_backend stats: {e}")

                chunks_stored = int(stats.total_chunks)

                # Derive two-phase indexing status
                if chunks_stored == 0:
                    indexing_status = "not_indexed"
                elif vectors_embedded == 0:
                    indexing_status = "embedding_incomplete"
                elif vectors_embedded < chunks_stored:
                    indexing_status = "partially_embedded"
                else:
                    indexing_status = "ready"

                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "chunks_stored": chunks_stored,
                    "vectors_embedded": vectors_embedded,
                    "total_files": stats.total_files,
                    "indexing_status": indexing_status,
                    "index_size": (
                        f"{stats.index_size_mb:.2f} MB"
                        if hasattr(stats, "index_size_mb")
                        else "Unknown"
                    ),
                }

                # Add resume hint when Phase 1 complete but Phase 2 not started
                if chunks_stored > 0 and vectors_embedded == 0:
                    status_info["hint"] = (
                        "Chunks are ready but vectors are not embedded. "
                        "Run embed_chunks to complete indexing without re-chunking."
                    )

                # Add KG status
                # Check if indexer is available (for live status during background build)
                if self.indexer:
                    status_info["kg_status"] = self.indexer.get_kg_status()
                    status_info["search_available"] = vectors_embedded > 0
                else:
                    # Check if KG database file exists (for servers without file watching)
                    kg_db_path = (
                        config.project_root
                        / ".mcp-vector-search"
                        / "knowledge_graph"
                        / "kg.db"
                    )
                    if kg_db_path.exists():
                        status_info["kg_status"] = "complete"
                    else:
                        status_info["kg_status"] = "not_started"
                    status_info["search_available"] = vectors_embedded > 0
            else:
                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "status": "Not indexed",
                }

            response_text = self._format_project_status(status_info)
            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except ProjectNotFoundError:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first.",
                    )
                ],
                isError=True,
            )

    async def handle_index_project(
        self, args: dict[str, Any], cleanup_callback, initialize_callback
    ) -> CallToolResult:
        """Handle index_project tool call.

        Args:
            args: Tool call arguments containing force, file_extensions
            cleanup_callback: Async function to cleanup resources before reindexing
            initialize_callback: Async function to reinitialize after reindexing

        Returns:
            CallToolResult with indexing result or error
        """
        force = args.get("force", False)
        file_extensions = args.get("file_extensions")

        try:
            # Ensure .mcp-vector-search/ is gitignored for MCP-first workflows
            # (when 'init' was never run, the gitignore entry may be missing)
            ensure_gitignore_entry(self.project_root)

            # Check if index already exists (unless force is set)
            if not force and self.search_engine:
                stats = await self.search_engine.database.get_stats()
                if int(stats.total_chunks) > 0:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Index already exists with {stats.total_chunks} chunks from {stats.total_files} files. "
                                f"Use force=true to reindex, or run search queries directly. "
                                f"Call get_project_status for more details.",
                            )
                        ]
                    )

            # Import indexing functionality
            from ..cli.commands.index import run_indexing

            # Run indexing
            await run_indexing(
                project_root=self.project_root,
                force_reindex=force,
                extensions=file_extensions,
                show_progress=False,  # Disable progress for MCP
            )

            # Reinitialize search engine after indexing
            await cleanup_callback()
            await initialize_callback()

            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text="Project indexing completed successfully!"
                    )
                ]
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Indexing failed: {str(e)}")],
                isError=True,
            )

    async def handle_embed_chunks(
        self, args: dict[str, Any], cleanup_callback, initialize_callback
    ) -> CallToolResult:
        """Handle embed_chunks tool call.

        Args:
            args: Tool call arguments containing fresh, batch_size
            cleanup_callback: Async function to cleanup resources
            initialize_callback: Async function to reinitialize after embedding

        Returns:
            CallToolResult with embedding result or error
        """
        fresh = args.get("fresh", False)
        batch_size = args.get("batch_size", 512)

        try:
            from ..core.embeddings import create_embedding_function
            from ..core.factory import create_database
            from ..core.indexer import SemanticIndexer
            from ..core.project import ProjectManager

            project_manager = ProjectManager(self.project_root)
            config = project_manager.load_config()

            embedding_function, _ = create_embedding_function(
                model_name=config.embedding_model,
            )

            database = create_database(
                persist_directory=config.index_path / "lance",
                embedding_function=embedding_function,
            )

            indexer = SemanticIndexer(
                database=database,
                project_root=self.project_root,
                config=config,
                batch_size=batch_size,
                skip_blame=True,
            )

            async with database:
                result = await indexer.embed_chunks(fresh=fresh, batch_size=batch_size)

            # Reinitialize search engine after embedding
            await cleanup_callback()
            await initialize_callback()

            embedded = result.get("chunks_embedded", 0)
            batches = result.get("batches_processed", 0)
            duration = result.get("duration_seconds", 0)

            if embedded > 0:
                msg = f"Embedding complete: {embedded:,} chunks embedded in {batches} batches ({duration:.1f}s)"
            else:
                msg = "No pending chunks to embed. Run index_project first to chunk files."

            return CallToolResult(content=[TextContent(type="text", text=msg)])

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Embedding failed: {str(e)}")],
                isError=True,
            )

    def _format_project_status(self, status_info: dict) -> str:
        """Format project status information.

        Args:
            status_info: Dictionary containing project status information

        Returns:
            Formatted text response
        """
        response_text = "# Project Status\n\n"
        response_text += f"**Project Root:** {status_info['project_root']}\n"
        response_text += f"**Index Path:** {status_info['index_path']}\n"
        response_text += (
            f"**File Extensions:** {', '.join(status_info['file_extensions'])}\n"
        )
        response_text += f"**Embedding Model:** {status_info['embedding_model']}\n"
        response_text += f"**Languages:** {', '.join(status_info['languages'])}\n"

        if "chunks_stored" in status_info:
            chunks_stored = status_info["chunks_stored"]
            vectors_embedded = status_info["vectors_embedded"]
            indexing_status = status_info.get("indexing_status", "unknown")

            response_text += f"**Chunks Stored (Phase 1):** {chunks_stored}\n"
            response_text += f"**Vectors Embedded (Phase 2):** {vectors_embedded}\n"
            response_text += f"**Total Files:** {status_info['total_files']}\n"
            response_text += f"**Index Size:** {status_info['index_size']}\n"

            # Indexing status with human-readable label
            status_labels = {
                "not_indexed": "Not indexed",
                "embedding_incomplete": "Chunked but not embedded (Phase 2 incomplete)",
                "partially_embedded": f"Partially embedded ({vectors_embedded}/{chunks_stored} vectors ready)",
                "ready": "Ready",
            }
            status_display = status_labels.get(indexing_status, indexing_status)
            response_text += f"**Indexing Status:** {status_display}\n"

            # Surface hint when Phase 2 is missing
            if "hint" in status_info:
                response_text += f"\n> **Hint:** {status_info['hint']}\n"

            # Add KG status and search availability
            if "kg_status" in status_info:
                kg_status = status_info["kg_status"]
                if kg_status == "not_started":
                    kg_display = "Not built"
                elif kg_status == "building":
                    kg_display = "Building in background..."
                elif kg_status == "complete":
                    kg_display = "Available"
                elif kg_status.startswith("error:"):
                    kg_display = f"Failed: {kg_status[7:]}"
                else:
                    kg_display = kg_status

                response_text += f"**Knowledge Graph:** {kg_display}\n"
                response_text += f"**Search Available:** {'Yes' if status_info['search_available'] else 'No'}\n"
        elif "total_chunks" in status_info:
            # Fallback for callers that still pass the old total_chunks key
            response_text += f"**Total Chunks:** {status_info['total_chunks']}\n"
            response_text += f"**Total Files:** {status_info['total_files']}\n"
            response_text += f"**Index Size:** {status_info['index_size']}\n"

            if "kg_status" in status_info:
                kg_status = status_info["kg_status"]
                if kg_status == "not_started":
                    kg_display = "Not built"
                elif kg_status == "building":
                    kg_display = "Building in background..."
                elif kg_status == "complete":
                    kg_display = "Available"
                elif kg_status.startswith("error:"):
                    kg_display = f"Failed: {kg_status[7:]}"
                else:
                    kg_display = kg_status

                response_text += f"**Knowledge Graph:** {kg_display}\n"
                response_text += f"**Search Available:** {'Yes' if status_info['search_available'] else 'No'}\n"
        else:
            response_text += f"**Status:** {status_info['status']}\n"

        return response_text

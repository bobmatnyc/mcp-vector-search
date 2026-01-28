"""Project management handlers for MCP vector search server."""

from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent

from ..core.exceptions import ProjectNotFoundError
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine


class ProjectHandlers:
    """Handlers for project management-related MCP tool operations."""

    def __init__(
        self,
        project_manager: ProjectManager,
        search_engine: SemanticSearchEngine | None,
        project_root: Path,
    ):
        """Initialize project handlers.

        Args:
            project_manager: Project manager instance
            search_engine: Semantic search engine instance (or None if not initialized)
            project_root: Project root directory
        """
        self.project_manager = project_manager
        self.search_engine = search_engine
        self.project_root = project_root

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

                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "total_chunks": stats.total_chunks,
                    "total_files": stats.total_files,
                    "index_size": (
                        f"{stats.index_size_mb:.2f} MB"
                        if hasattr(stats, "index_size_mb")
                        else "Unknown"
                    ),
                }
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

        if "total_chunks" in status_info:
            response_text += f"**Total Chunks:** {status_info['total_chunks']}\n"
            response_text += f"**Total Files:** {status_info['total_files']}\n"
            response_text += f"**Index Size:** {status_info['index_size']}\n"
        else:
            response_text += f"**Status:** {status_info['status']}\n"

        return response_text

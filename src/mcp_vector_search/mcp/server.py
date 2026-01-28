"""MCP server implementation for MCP Vector Search."""

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ServerCapabilities,
    TextContent,
)

from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.indexer import SemanticIndexer
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine
from ..core.watcher import FileWatcher
from .analysis_handlers import AnalysisHandlers
from .project_handlers import ProjectHandlers
from .search_handlers import SearchHandlers
from .tool_schemas import get_tool_schemas


class MCPVectorSearchServer:
    """MCP server for vector search functionality."""

    def __init__(
        self,
        project_root: Path | None = None,
        enable_file_watching: bool | None = None,
    ):
        """Initialize the MCP server.

        Args:
            project_root: Project root directory. If None, will auto-detect from:
                         1. PROJECT_ROOT or MCP_PROJECT_ROOT environment variable
                         2. Current working directory
            enable_file_watching: Enable file watching for automatic reindexing.
                                  If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
        """
        # Auto-detect project root from environment or current directory
        if project_root is None:
            # Priority 1: MCP_PROJECT_ROOT (new standard)
            # Priority 2: PROJECT_ROOT (legacy)
            # Priority 3: Current working directory
            env_project_root = os.getenv("MCP_PROJECT_ROOT") or os.getenv(
                "PROJECT_ROOT"
            )
            if env_project_root:
                project_root = Path(env_project_root).resolve()
                logger.info(f"Using project root from environment: {project_root}")
            else:
                project_root = Path.cwd()
                logger.info(f"Using current directory as project root: {project_root}")

        self.project_root = project_root
        self.project_manager = ProjectManager(self.project_root)
        self.search_engine: SemanticSearchEngine | None = None
        self.file_watcher: FileWatcher | None = None
        self.indexer: SemanticIndexer | None = None
        self.database: ChromaVectorDatabase | None = None
        self._initialized = False

        # Determine if file watching should be enabled
        if enable_file_watching is None:
            # Check environment variable, default to True
            env_value = os.getenv("MCP_ENABLE_FILE_WATCHING", "true").lower()
            self.enable_file_watching = env_value in ("true", "1", "yes", "on")
        else:
            self.enable_file_watching = enable_file_watching

        # Initialize handler instances (lazy initialization on first use)
        self._search_handlers: SearchHandlers | None = None
        self._analysis_handlers: AnalysisHandlers | None = None
        self._project_handlers: ProjectHandlers | None = None

    async def initialize(self) -> None:
        """Initialize the search engine and database."""
        if self._initialized:
            return

        try:
            # Run pending migrations first
            await self._run_migrations()

            # Load project configuration
            config = self.project_manager.load_config()

            # Setup embedding function
            embedding_function, _ = create_embedding_function(
                model_name=config.embedding_model
            )

            # Setup database
            self.database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )

            # Initialize database
            await self.database.__aenter__()

            # Setup search engine
            self.search_engine = SemanticSearchEngine(
                database=self.database, project_root=self.project_root
            )

            # Initialize handlers
            self._search_handlers = SearchHandlers(
                self.search_engine, self.project_root
            )
            self._analysis_handlers = AnalysisHandlers(self.project_root)
            self._project_handlers = ProjectHandlers(
                self.project_manager, self.search_engine, self.project_root
            )

            # Setup indexer for file watching
            if self.enable_file_watching:
                self.indexer = SemanticIndexer(
                    database=self.database,
                    project_root=self.project_root,
                    config=config,
                )

                # Setup file watcher
                self.file_watcher = FileWatcher(
                    project_root=self.project_root,
                    config=config,
                    indexer=self.indexer,
                    database=self.database,
                )

                # Start file watching
                await self.file_watcher.start()
                logger.info("File watching enabled for automatic reindexing")
            else:
                logger.info("File watching disabled")

            self._initialized = True
            logger.info(f"MCP server initialized for project: {self.project_root}")

        except ProjectNotFoundError:
            logger.error(f"Project not initialized at {self.project_root}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise

    async def _run_migrations(self) -> None:
        """Run pending migrations on startup.

        Migrations are run automatically but failures only log warnings
        to avoid blocking server startup.
        """
        try:
            from ..migrations import MigrationRunner
            from ..migrations.v1_2_2_codexembed import CodeXEmbedMigration

            runner = MigrationRunner(self.project_root)
            runner.register_migrations([CodeXEmbedMigration()])

            pending = runner.get_pending_migrations()
            if pending:
                logger.info(f"Running {len(pending)} pending migration(s)...")
                results = runner.run_pending_migrations()

                for result in results:
                    if result.status.value == "success":
                        logger.info(f"✓ Migration {result.migration_id} completed")
                    elif result.status.value == "failed":
                        logger.warning(
                            f"Migration {result.migration_id} failed: {result.message}"
                        )
                    # Skipped migrations are silently ignored
            else:
                logger.debug("No pending migrations")

        except Exception as e:
            # Don't block server startup on migration failures
            logger.warning(f"Migration check failed (non-fatal): {e}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Stop file watcher if running
        if self.file_watcher and self.file_watcher.is_running:
            logger.info("Stopping file watcher...")
            await self.file_watcher.stop()
            self.file_watcher = None

        # Cleanup database connection
        if self.database and hasattr(self.database, "__aexit__"):
            await self.database.__aexit__(None, None, None)
            self.database = None

        # Clear references
        self.search_engine = None
        self.indexer = None
        self._initialized = False
        logger.info("MCP server cleanup completed")

    def get_tools(self):
        """Get available MCP tools.

        Returns:
            List of Tool objects defining available MCP tools
        """
        return get_tool_schemas()

    def get_capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(tools={"listChanged": True}, logging={})

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls by delegating to appropriate handlers.

        Args:
            request: MCP tool call request

        Returns:
            CallToolResult from the appropriate handler
        """
        # Skip initialization for interpret_analysis (doesn't need project config)
        if request.params.name != "interpret_analysis" and not self._initialized:
            await self.initialize()

        try:
            tool_name = request.params.name
            args = request.params.arguments

            # Delegate to search handlers
            if tool_name == "search_code":
                return await self._search_handlers.handle_search_code(args)
            elif tool_name == "search_similar":
                return await self._search_handlers.handle_search_similar(args)
            elif tool_name == "search_context":
                return await self._search_handlers.handle_search_context(args)

            # Delegate to project handlers
            elif tool_name == "get_project_status":
                return await self._project_handlers.handle_get_project_status(args)
            elif tool_name == "index_project":
                return await self._project_handlers.handle_index_project(
                    args, self.cleanup, self.initialize
                )

            # Delegate to analysis handlers
            elif tool_name == "analyze_project":
                return await self._analysis_handlers.handle_analyze_project(args)
            elif tool_name == "analyze_file":
                return await self._analysis_handlers.handle_analyze_file(args)
            elif tool_name == "find_smells":
                return await self._analysis_handlers.handle_find_smells(args)
            elif tool_name == "get_complexity_hotspots":
                return await self._analysis_handlers.handle_get_complexity_hotspots(
                    args
                )
            elif tool_name == "check_circular_dependencies":
                return await self._analysis_handlers.handle_check_circular_dependencies(
                    args
                )
            elif tool_name == "interpret_analysis":
                return await self._analysis_handlers.handle_interpret_analysis(args)

            else:
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Unknown tool: {tool_name}")
                    ],
                    isError=True,
                )

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Tool execution failed: {str(e)}")
                ],
                isError=True,
            )


def create_mcp_server(
    project_root: Path | None = None, enable_file_watching: bool | None = None
) -> Server:
    """Create and configure the MCP server.

    Args:
        project_root: Project root directory. If None, will auto-detect.
        enable_file_watching: Enable file watching for automatic reindexing.
                              If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
    """
    server = Server("mcp-vector-search")
    mcp_server = MCPVectorSearchServer(project_root, enable_file_watching)

    @server.list_tools()
    async def handle_list_tools():
        """List available tools."""
        return mcp_server.get_tools()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None):
        """Handle tool calls."""
        # Create a mock request object for compatibility
        from types import SimpleNamespace

        mock_request = SimpleNamespace()
        mock_request.params = SimpleNamespace()
        mock_request.params.name = name
        mock_request.params.arguments = arguments or {}

        result = await mcp_server.call_tool(mock_request)

        # Return the content from the result
        return result.content

    # Store reference for cleanup
    server._mcp_server = mcp_server

    return server


async def run_mcp_server(
    project_root: Path | None = None, enable_file_watching: bool | None = None
) -> None:
    """Run the MCP server using stdio transport.

    Args:
        project_root: Project root directory. If None, will auto-detect.
        enable_file_watching: Enable file watching for automatic reindexing.
                              If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
    """
    server = create_mcp_server(project_root, enable_file_watching)

    # Create initialization options with proper capabilities
    init_options = InitializationOptions(
        server_name="mcp-vector-search",
        server_version="0.4.0",
        capabilities=ServerCapabilities(tools={"listChanged": True}, logging={}),
    )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(server, "_mcp_server"):
            logger.info("Performing server cleanup...")
            await server._mcp_server.cleanup()


if __name__ == "__main__":
    # Allow specifying project root as command line argument
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Check for file watching flag in command line args
    enable_file_watching = None
    if "--no-watch" in sys.argv:
        enable_file_watching = False
        sys.argv.remove("--no-watch")
    elif "--watch" in sys.argv:
        enable_file_watching = True
        sys.argv.remove("--watch")

    asyncio.run(run_mcp_server(project_root, enable_file_watching))

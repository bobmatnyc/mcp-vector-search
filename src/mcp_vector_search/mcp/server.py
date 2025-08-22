"""MCP server implementation for MCP Vector Search."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    ServerCapabilities,
    TextContent,
    Tool,
)
from mcp.server.models import InitializationOptions

from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.indexer import SemanticIndexer
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine
from ..core.watcher import FileWatcher


class MCPVectorSearchServer:
    """MCP server for vector search functionality."""

    def __init__(self, project_root: Optional[Path] = None, enable_file_watching: Optional[bool] = None):
        """Initialize the MCP server.
        
        Args:
            project_root: Project root directory. If None, will auto-detect.
            enable_file_watching: Enable file watching for automatic reindexing.
                                  If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
        """
        self.project_root = project_root or Path.cwd()
        self.project_manager = ProjectManager(self.project_root)
        self.search_engine: Optional[SemanticSearchEngine] = None
        self.file_watcher: Optional[FileWatcher] = None
        self.indexer: Optional[SemanticIndexer] = None
        self.database: Optional[ChromaVectorDatabase] = None
        self._initialized = False
        
        # Determine if file watching should be enabled
        if enable_file_watching is None:
            # Check environment variable, default to True
            env_value = os.getenv("MCP_ENABLE_FILE_WATCHING", "true").lower()
            self.enable_file_watching = env_value in ("true", "1", "yes", "on")
        else:
            self.enable_file_watching = enable_file_watching

    async def initialize(self) -> None:
        """Initialize the search engine and database."""
        if self._initialized:
            return

        try:
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
                database=self.database,
                project_root=self.project_root
            )
            
            # Setup indexer for file watching
            if self.enable_file_watching:
                self.indexer = SemanticIndexer(
                    database=self.database,
                    project_root=self.project_root,
                    file_extensions=config.file_extensions
                )
                
                # Setup file watcher
                self.file_watcher = FileWatcher(
                    project_root=self.project_root,
                    config=config,
                    indexer=self.indexer,
                    database=self.database
                )
                
                # Start file watching
                await self.file_watcher.start()
                logger.info(f"File watching enabled for automatic reindexing")
            else:
                logger.info(f"File watching disabled")
            
            self._initialized = True
            logger.info(f"MCP server initialized for project: {self.project_root}")
            
        except ProjectNotFoundError:
            logger.error(f"Project not initialized at {self.project_root}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Stop file watcher if running
        if self.file_watcher and self.file_watcher.is_running:
            logger.info("Stopping file watcher...")
            await self.file_watcher.stop()
            self.file_watcher = None
        
        # Cleanup database connection
        if self.database and hasattr(self.database, '__aexit__'):
            await self.database.__aexit__(None, None, None)
            self.database = None
        
        # Clear references
        self.search_engine = None
        self.indexer = None
        self._initialized = False
        logger.info("MCP server cleanup completed")

    def get_tools(self) -> List[Tool]:
        """Get available MCP tools."""
        tools = [
            Tool(
                name="search_code",
                description="Search for code using semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant code"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by file extensions (e.g., ['.py', '.js'])"
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter by function name"
                        },
                        "class_name": {
                            "type": "string",
                            "description": "Filter by class name"
                        },
                        "files": {
                            "type": "string",
                            "description": "Filter by file patterns (e.g., '*.py' or 'src/*.js')"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="search_similar",
                description="Find code similar to a specific file or function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to find similar code for"
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Optional function name within the file"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="search_context",
                description="Search for code based on contextual description",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Contextual description of what you're looking for"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Areas to focus on (e.g., ['security', 'authentication'])"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["description"]
                }
            ),
            Tool(
                name="get_project_status",
                description="Get project indexing status and statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="index_project",
                description="Index or reindex the project codebase",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force reindexing even if index exists",
                            "default": False
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to index (e.g., ['.py', '.js'])"
                        }
                    },
                    "required": []
                }
            )
        ]

        return tools

    def get_capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(
            tools={"listChanged": True},
            logging={}
        )

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        if not self._initialized:
            await self.initialize()

        try:
            if request.params.name == "search_code":
                return await self._search_code(request.params.arguments)
            elif request.params.name == "search_similar":
                return await self._search_similar(request.params.arguments)
            elif request.params.name == "search_context":
                return await self._search_context(request.params.arguments)
            elif request.params.name == "get_project_status":
                return await self._get_project_status(request.params.arguments)
            elif request.params.name == "index_project":
                return await self._index_project(request.params.arguments)
            else:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Unknown tool: {request.params.name}"
                    )],
                    isError=True
                )
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Tool execution failed: {str(e)}"
                )],
                isError=True
            )

    async def _search_code(self, args: Dict[str, Any]) -> CallToolResult:
        """Handle search_code tool call."""
        query = args.get("query", "")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)
        file_extensions = args.get("file_extensions")
        language = args.get("language")
        function_name = args.get("function_name")
        class_name = args.get("class_name")
        files = args.get("files")

        if not query:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Query parameter is required"
                )],
                isError=True
            )

        # Build filters
        filters = {}
        if file_extensions:
            filters["file_extension"] = {"$in": file_extensions}
        if language:
            filters["language"] = language
        if function_name:
            filters["function_name"] = function_name
        if class_name:
            filters["class_name"] = class_name
        if files:
            # Convert file pattern to filter (simplified)
            filters["file_pattern"] = files

        # Perform search
        results = await self.search_engine.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters
        )

        # Format results
        if not results:
            response_text = f"No results found for query: '{query}'"
        else:
            response_lines = [f"Found {len(results)} results for query: '{query}'\n"]
            
            for i, result in enumerate(results, 1):
                response_lines.append(f"## Result {i} (Score: {result.similarity_score:.3f})")
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                response_lines.append(result.content)
                response_lines.append("```\n")
            
            response_text = "\n".join(response_lines)

        return CallToolResult(
            content=[TextContent(type="text", text=response_text)]
        )

    async def _get_project_status(self, args: Dict[str, Any]) -> CallToolResult:
        """Handle get_project_status tool call."""
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
                    "index_size": f"{stats.index_size_mb:.2f} MB" if hasattr(stats, 'index_size_mb') else "Unknown"
                }
            else:
                status_info = {
                    "project_root": str(config.project_root),
                    "index_path": str(config.index_path),
                    "file_extensions": config.file_extensions,
                    "embedding_model": config.embedding_model,
                    "languages": config.languages,
                    "status": "Not indexed"
                }
            
            response_text = f"# Project Status\n\n"
            response_text += f"**Project Root:** {status_info['project_root']}\n"
            response_text += f"**Index Path:** {status_info['index_path']}\n"
            response_text += f"**File Extensions:** {', '.join(status_info['file_extensions'])}\n"
            response_text += f"**Embedding Model:** {status_info['embedding_model']}\n"
            response_text += f"**Languages:** {', '.join(status_info['languages'])}\n"
            
            if "total_chunks" in status_info:
                response_text += f"**Total Chunks:** {status_info['total_chunks']}\n"
                response_text += f"**Total Files:** {status_info['total_files']}\n"
                response_text += f"**Index Size:** {status_info['index_size']}\n"
            else:
                response_text += f"**Status:** {status_info['status']}\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )
            
        except ProjectNotFoundError:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first."
                )],
                isError=True
            )

    async def _index_project(self, args: Dict[str, Any]) -> CallToolResult:
        """Handle index_project tool call."""
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
                show_progress=False  # Disable progress for MCP
            )
            
            # Reinitialize search engine after indexing
            await self.cleanup()
            await self.initialize()
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Project indexing completed successfully!"
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Indexing failed: {str(e)}"
                )],
                isError=True
            )

    async def _search_similar(self, args: Dict[str, Any]) -> CallToolResult:
        """Handle search_similar tool call."""
        file_path = args.get("file_path", "")
        function_name = args.get("function_name")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)

        if not file_path:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="file_path parameter is required"
                )],
                isError=True
            )

        try:
            from pathlib import Path

            # Convert to Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                file_path_obj = self.project_root / file_path_obj

            if not file_path_obj.exists():
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"File not found: {file_path}"
                    )],
                    isError=True
                )

            # Run similar search
            results = await self.search_engine.search_similar(
                file_path=file_path_obj,
                function_name=function_name,
                limit=limit,
                similarity_threshold=similarity_threshold
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No similar code found for {file_path}"
                    )]
                )

            response_lines = [f"Found {len(results)} similar code snippets for {file_path}\n"]
            
            for i, result in enumerate(results, 1):
                response_lines.append(f"## Result {i} (Score: {result.similarity_score:.3f})")
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for similar search
                content_preview = result.content[:500] if len(result.content) > 500 else result.content
                response_lines.append(content_preview + ("..." if len(result.content) > 500 else ""))
                response_lines.append("```\n")
            
            result_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result_text
                )]
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Similar search failed: {str(e)}"
                )],
                isError=True
            )

    async def _search_context(self, args: Dict[str, Any]) -> CallToolResult:
        """Handle search_context tool call."""
        description = args.get("description", "")
        focus_areas = args.get("focus_areas")
        limit = args.get("limit", 10)

        if not description:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="description parameter is required"
                )],
                isError=True
            )

        try:
            # Perform context search
            results = await self.search_engine.search_by_context(
                context_description=description,
                focus_areas=focus_areas,
                limit=limit
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"No contextually relevant code found for: {description}"
                    )]
                )

            response_lines = [f"Found {len(results)} contextually relevant code snippets"]
            if focus_areas:
                response_lines[0] += f" (focus: {', '.join(focus_areas)})"
            response_lines[0] += f" for: {description}\n"
            
            for i, result in enumerate(results, 1):
                response_lines.append(f"## Result {i} (Score: {result.similarity_score:.3f})")
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(f"**Lines:** {result.start_line}-{result.end_line}")
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for context search
                content_preview = result.content[:500] if len(result.content) > 500 else result.content
                response_lines.append(content_preview + ("..." if len(result.content) > 500 else ""))
                response_lines.append("```\n")
            
            result_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result_text
                )]
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Context search failed: {str(e)}"
                )],
                isError=True
            )


def create_mcp_server(project_root: Optional[Path] = None, enable_file_watching: Optional[bool] = None) -> Server:
    """Create and configure the MCP server.
    
    Args:
        project_root: Project root directory. If None, will auto-detect.
        enable_file_watching: Enable file watching for automatic reindexing.
                              If None, checks MCP_ENABLE_FILE_WATCHING env var (default: True).
    """
    server = Server("mcp-vector-search")
    mcp_server = MCPVectorSearchServer(project_root, enable_file_watching)

    @server.list_tools()
    async def handle_list_tools() -> List[Tool]:
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


async def run_mcp_server(project_root: Optional[Path] = None, enable_file_watching: Optional[bool] = None) -> None:
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
        capabilities=ServerCapabilities(
            tools={"listChanged": True},
            logging={}
        )
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
        if hasattr(server, '_mcp_server'):
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

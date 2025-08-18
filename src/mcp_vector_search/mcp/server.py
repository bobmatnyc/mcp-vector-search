"""MCP server implementation for MCP Vector Search."""

import asyncio
import json
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
    TextContent,
    Tool,
)

from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine


class MCPVectorSearchServer:
    """MCP server for vector search functionality."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the MCP server.
        
        Args:
            project_root: Project root directory. If None, will auto-detect.
        """
        self.project_root = project_root or Path.cwd()
        self.project_manager = ProjectManager(self.project_root)
        self.search_engine: Optional[SemanticSearchEngine] = None
        self._initialized = False

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
            database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )
            
            # Initialize database
            await database.__aenter__()
            
            # Setup search engine
            self.search_engine = SemanticSearchEngine(
                database=database,
                project_root=self.project_root
            )
            
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
        if self.search_engine and hasattr(self.search_engine.database, '__aexit__'):
            await self.search_engine.database.__aexit__(None, None, None)
        self._initialized = False

    def get_tools(self) -> List[Tool]:
        """Get available MCP tools."""
        return [
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
                        }
                    },
                    "required": ["query"]
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

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        if not self._initialized:
            await self.initialize()

        try:
            if request.params.name == "search_code":
                return await self._search_code(request.params.arguments)
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
                response_lines.append(f"## Result {i} (Score: {result.score:.3f})")
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
                    "total_chunks": stats.get("total_chunks", 0),
                    "total_files": stats.get("total_files", 0),
                    "index_size": stats.get("index_size", "Unknown")
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


def create_mcp_server(project_root: Optional[Path] = None) -> Server:
    """Create and configure the MCP server."""
    server = Server("mcp-vector-search")
    mcp_server = MCPVectorSearchServer(project_root)
    
    @server.list_tools()
    async def list_tools(request: ListToolsRequest) -> ListToolsResult:
        """List available tools."""
        return ListToolsResult(tools=mcp_server.get_tools())
    
    @server.call_tool()
    async def call_tool(request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        return await mcp_server.call_tool(request)
    
    # Store reference for cleanup
    server._mcp_server = mcp_server
    
    return server


async def run_mcp_server(project_root: Optional[Path] = None) -> None:
    """Run the MCP server using stdio transport."""
    server = create_mcp_server(project_root)
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # Cleanup
        if hasattr(server, '_mcp_server'):
            await server._mcp_server.cleanup()


if __name__ == "__main__":
    # Allow specifying project root as command line argument
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(run_mcp_server(project_root))

"""MCP server implementation for MCP Vector Search."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ServerCapabilities,
    TextContent,
    Tool,
)

from ..analysis import (
    ProjectMetrics,
    SmellDetector,
    SmellSeverity,
)
from ..config.thresholds import ThresholdConfig
from ..core.database import ChromaVectorDatabase
from ..core.embeddings import create_embedding_function
from ..core.exceptions import ProjectNotFoundError
from ..core.indexer import SemanticIndexer
from ..core.project import ProjectManager
from ..core.search import SemanticSearchEngine
from ..core.watcher import FileWatcher
from ..parsers.registry import ParserRegistry


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
                database=self.database, project_root=self.project_root
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

    def get_tools(self) -> list[Tool]:
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
                            "description": "The search query to find relevant code",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by file extensions (e.g., ['.py', '.js'])",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Filter by function name",
                        },
                        "class_name": {
                            "type": "string",
                            "description": "Filter by class name",
                        },
                        "files": {
                            "type": "string",
                            "description": "Filter by file patterns (e.g., '*.py' or 'src/*.js')",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_similar",
                description="Find code similar to a specific file or function",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to find similar code for",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "Optional function name within the file",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0.0-1.0)",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="search_context",
                description="Search for code based on contextual description",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Contextual description of what you're looking for",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Areas to focus on (e.g., ['security', 'authentication'])",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["description"],
                },
            ),
            Tool(
                name="get_project_status",
                description="Get project indexing status and statistics",
                inputSchema={"type": "object", "properties": {}, "required": []},
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
                            "default": False,
                        },
                        "file_extensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to index (e.g., ['.py', '.js'])",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="analyze_project",
                description="Returns project-wide metrics summary",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "threshold_preset": {
                            "type": "string",
                            "description": "Threshold preset: 'strict', 'standard', or 'relaxed'",
                            "enum": ["strict", "standard", "relaxed"],
                            "default": "standard",
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Output format: 'summary' or 'detailed'",
                            "enum": ["summary", "detailed"],
                            "default": "summary",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="analyze_file",
                description="Returns file-level metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to analyze (relative or absolute)",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="find_smells",
                description="Returns list of code smells",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "smell_type": {
                            "type": "string",
                            "description": "Filter by smell type: 'Long Method', 'Deep Nesting', 'Long Parameter List', 'God Class', 'Complex Method'",
                            "enum": [
                                "Long Method",
                                "Deep Nesting",
                                "Long Parameter List",
                                "God Class",
                                "Complex Method",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "description": "Filter by severity level",
                            "enum": ["info", "warning", "error"],
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="get_complexity_hotspots",
                description="Returns top N most complex functions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of hotspots to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="check_circular_dependencies",
                description="Returns circular dependency cycles",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="interpret_analysis",
                description="Interpret analysis results with natural language explanations and recommendations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "analysis_json": {
                            "type": "string",
                            "description": "JSON string from analyze command with --include-context",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Focus area: 'summary', 'recommendations', or 'priorities'",
                            "enum": ["summary", "recommendations", "priorities"],
                            "default": "summary",
                        },
                        "verbosity": {
                            "type": "string",
                            "description": "Verbosity level: 'brief', 'normal', or 'detailed'",
                            "enum": ["brief", "normal", "detailed"],
                            "default": "normal",
                        },
                    },
                    "required": ["analysis_json"],
                },
            ),
        ]

        return tools

    def get_capabilities(self) -> ServerCapabilities:
        """Get server capabilities."""
        return ServerCapabilities(tools={"listChanged": True}, logging={})

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        # Skip initialization for interpret_analysis (doesn't need project config)
        if request.params.name != "interpret_analysis" and not self._initialized:
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
            elif request.params.name == "analyze_project":
                return await self._analyze_project(request.params.arguments)
            elif request.params.name == "analyze_file":
                return await self._analyze_file(request.params.arguments)
            elif request.params.name == "find_smells":
                return await self._find_smells(request.params.arguments)
            elif request.params.name == "get_complexity_hotspots":
                return await self._get_complexity_hotspots(request.params.arguments)
            elif request.params.name == "check_circular_dependencies":
                return await self._check_circular_dependencies(request.params.arguments)
            elif request.params.name == "interpret_analysis":
                return await self._interpret_analysis(request.params.arguments)
            else:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"Unknown tool: {request.params.name}"
                        )
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

    async def _search_code(self, args: dict[str, Any]) -> CallToolResult:
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
                content=[TextContent(type="text", text="Query parameter is required")],
                isError=True,
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
            filters=filters,
        )

        # Format results
        if not results:
            response_text = f"No results found for query: '{query}'"
        else:
            response_lines = [f"Found {len(results)} results for query: '{query}'\n"]

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                response_lines.append(result.content)
                response_lines.append("```\n")

            response_text = "\n".join(response_lines)

        return CallToolResult(content=[TextContent(type="text", text=response_text)])

    async def _get_project_status(self, args: dict[str, Any]) -> CallToolResult:
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

    async def _index_project(self, args: dict[str, Any]) -> CallToolResult:
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
                show_progress=False,  # Disable progress for MCP
            )

            # Reinitialize search engine after indexing
            await self.cleanup()
            await self.initialize()

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

    async def _search_similar(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_similar tool call."""
        file_path = args.get("file_path", "")
        function_name = args.get("function_name")
        limit = args.get("limit", 10)
        similarity_threshold = args.get("similarity_threshold", 0.3)

        if not file_path:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            from pathlib import Path

            # Convert to Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                file_path_obj = self.project_root / file_path_obj

            if not file_path_obj.exists():
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"File not found: {file_path}")
                    ],
                    isError=True,
                )

            # Run similar search
            results = await self.search_engine.search_similar(
                file_path=file_path_obj,
                function_name=function_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"No similar code found for {file_path}"
                        )
                    ]
                )

            response_lines = [
                f"Found {len(results)} similar code snippets for {file_path}\n"
            ]

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for similar search
                content_preview = (
                    result.content[:500]
                    if len(result.content) > 500
                    else result.content
                )
                response_lines.append(
                    content_preview + ("..." if len(result.content) > 500 else "")
                )
                response_lines.append("```\n")

            result_text = "\n".join(response_lines)

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Similar search failed: {str(e)}")
                ],
                isError=True,
            )

    async def _search_context(self, args: dict[str, Any]) -> CallToolResult:
        """Handle search_context tool call."""
        description = args.get("description", "")
        focus_areas = args.get("focus_areas")
        limit = args.get("limit", 10)

        if not description:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="description parameter is required")
                ],
                isError=True,
            )

        try:
            # Perform context search
            results = await self.search_engine.search_by_context(
                context_description=description, focus_areas=focus_areas, limit=limit
            )

            # Format results
            if not results:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"No contextually relevant code found for: {description}",
                        )
                    ]
                )

            response_lines = [
                f"Found {len(results)} contextually relevant code snippets"
            ]
            if focus_areas:
                response_lines[0] += f" (focus: {', '.join(focus_areas)})"
            response_lines[0] += f" for: {description}\n"

            for i, result in enumerate(results, 1):
                response_lines.append(
                    f"## Result {i} (Score: {result.similarity_score:.3f})"
                )
                response_lines.append(f"**File:** {result.file_path}")
                if result.function_name:
                    response_lines.append(f"**Function:** {result.function_name}")
                if result.class_name:
                    response_lines.append(f"**Class:** {result.class_name}")
                response_lines.append(
                    f"**Lines:** {result.start_line}-{result.end_line}"
                )
                response_lines.append("**Code:**")
                response_lines.append("```" + (result.language or ""))
                # Show more of the content for context search
                content_preview = (
                    result.content[:500]
                    if len(result.content) > 500
                    else result.content
                )
                response_lines.append(
                    content_preview + ("..." if len(result.content) > 500 else "")
                )
                response_lines.append("```\n")

            result_text = "\n".join(response_lines)

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Context search failed: {str(e)}")
                ],
                isError=True,
            )

    async def _analyze_project(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_project tool call."""
        threshold_preset = args.get("threshold_preset", "standard")
        output_format = args.get("output_format", "summary")

        try:
            # Load threshold configuration based on preset
            threshold_config = self._get_threshold_config(threshold_preset)

            # Run analysis using CLI analyze logic
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Analyze files
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception as e:
                    logger.debug(f"Failed to analyze {file_path}: {e}")
                    continue

            project_metrics.compute_aggregates()

            # Detect code smells
            smell_detector = SmellDetector(thresholds=threshold_config)
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Format response
            if output_format == "detailed":
                # Return full JSON output
                import json

                output = project_metrics.to_summary()
                output["smells"] = {
                    "total": len(all_smells),
                    "by_severity": {
                        "error": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.ERROR
                        ),
                        "warning": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.WARNING
                        ),
                        "info": sum(
                            1 for s in all_smells if s.severity == SmellSeverity.INFO
                        ),
                    },
                }
                response_text = json.dumps(output, indent=2)
            else:
                # Return summary
                summary = project_metrics.to_summary()
                response_lines = [
                    "# Project Analysis Summary\n",
                    f"**Project Root:** {summary['project_root']}",
                    f"**Total Files:** {summary['total_files']}",
                    f"**Total Functions:** {summary['total_functions']}",
                    f"**Total Classes:** {summary['total_classes']}",
                    f"**Average File Complexity:** {summary['avg_file_complexity']}\n",
                    "## Complexity Distribution",
                ]

                dist = summary["complexity_distribution"]
                for grade in ["A", "B", "C", "D", "F"]:
                    response_lines.append(f"- Grade {grade}: {dist[grade]} chunks")

                response_lines.extend(
                    [
                        "\n## Health Metrics",
                        f"- Average Health Score: {summary['health_metrics']['avg_health_score']:.2f}",
                        f"- Files Needing Attention: {summary['health_metrics']['files_needing_attention']}",
                        "\n## Code Smells",
                        f"- Total: {len(all_smells)}",
                        f"- Errors: {sum(1 for s in all_smells if s.severity == SmellSeverity.ERROR)}",
                        f"- Warnings: {sum(1 for s in all_smells if s.severity == SmellSeverity.WARNING)}",
                        f"- Info: {sum(1 for s in all_smells if s.severity == SmellSeverity.INFO)}",
                    ]
                )

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Project analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def _analyze_file(self, args: dict[str, Any]) -> CallToolResult:
        """Handle analyze_file tool call."""
        file_path_str = args.get("file_path", "")

        if not file_path_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="file_path parameter is required")
                ],
                isError=True,
            )

        try:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = self.project_root / file_path

            if not file_path.exists():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", text=f"File not found: {file_path_str}"
                        )
                    ],
                    isError=True,
                )

            # Analyze single file
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file

            parser_registry = ParserRegistry()
            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            file_metrics = await _analyze_file(file_path, parser_registry, collectors)

            if not file_metrics:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Unable to analyze file: {file_path_str}",
                        )
                    ],
                    isError=True,
                )

            # Detect smells
            smell_detector = SmellDetector()
            smells = smell_detector.detect_all(file_metrics, str(file_path))

            # Format response
            response_lines = [
                f"# File Analysis: {file_path.name}\n",
                f"**Path:** {file_path}",
                f"**Total Lines:** {file_metrics.total_lines}",
                f"**Code Lines:** {file_metrics.code_lines}",
                f"**Comment Lines:** {file_metrics.comment_lines}",
                f"**Functions:** {file_metrics.function_count}",
                f"**Classes:** {file_metrics.class_count}",
                f"**Methods:** {file_metrics.method_count}\n",
                "## Complexity Metrics",
                f"- Total Complexity: {file_metrics.total_complexity}",
                f"- Average Complexity: {file_metrics.avg_complexity:.2f}",
                f"- Max Complexity: {file_metrics.max_complexity}",
                f"- Health Score: {file_metrics.health_score:.2f}\n",
            ]

            if smells:
                response_lines.append(f"## Code Smells ({len(smells)})\n")
                for smell in smells[:10]:  # Show top 10
                    response_lines.append(
                        f"- [{smell.severity.value.upper()}] {smell.name}: {smell.description}"
                    )
                if len(smells) > 10:
                    response_lines.append(f"\n... and {len(smells) - 10} more")
            else:
                response_lines.append("## Code Smells\n- None detected")

            response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"File analysis failed: {str(e)}")
                ],
                isError=True,
            )

    async def _find_smells(self, args: dict[str, Any]) -> CallToolResult:
        """Handle find_smells tool call."""
        smell_type_filter = args.get("smell_type")
        severity_filter = args.get("severity")

        try:
            # Run full project analysis
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112 - intentional skip of unparseable files
                    continue

            # Detect all smells
            smell_detector = SmellDetector()
            all_smells = []
            for file_path, file_metrics in project_metrics.files.items():
                file_smells = smell_detector.detect_all(file_metrics, file_path)
                all_smells.extend(file_smells)

            # Apply filters
            filtered_smells = all_smells

            if smell_type_filter:
                filtered_smells = [
                    s for s in filtered_smells if s.name == smell_type_filter
                ]

            if severity_filter:
                severity_enum = SmellSeverity(severity_filter)
                filtered_smells = [
                    s for s in filtered_smells if s.severity == severity_enum
                ]

            # Format response
            if not filtered_smells:
                filter_desc = []
                if smell_type_filter:
                    filter_desc.append(f"type={smell_type_filter}")
                if severity_filter:
                    filter_desc.append(f"severity={severity_filter}")
                filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
                response_text = f"No code smells found{filter_str}"
            else:
                response_lines = [f"# Code Smells Found: {len(filtered_smells)}\n"]

                # Group by severity
                by_severity = {
                    "error": [
                        s for s in filtered_smells if s.severity == SmellSeverity.ERROR
                    ],
                    "warning": [
                        s
                        for s in filtered_smells
                        if s.severity == SmellSeverity.WARNING
                    ],
                    "info": [
                        s for s in filtered_smells if s.severity == SmellSeverity.INFO
                    ],
                }

                for severity_level in ["error", "warning", "info"]:
                    smells = by_severity[severity_level]
                    if smells:
                        response_lines.append(
                            f"## {severity_level.upper()} ({len(smells)})\n"
                        )
                        for smell in smells[:20]:  # Show top 20 per severity
                            response_lines.append(
                                f"- **{smell.name}** at `{smell.location}`"
                            )
                            response_lines.append(f"  {smell.description}")
                            if smell.suggestion:
                                response_lines.append(
                                    f"  *Suggestion: {smell.suggestion}*"
                                )
                            response_lines.append("")

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Smell detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Smell detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def _get_complexity_hotspots(self, args: dict[str, Any]) -> CallToolResult:
        """Handle get_complexity_hotspots tool call."""
        limit = args.get("limit", 10)

        try:
            # Run full project analysis
            from ..analysis import (
                CognitiveComplexityCollector,
                CyclomaticComplexityCollector,
                MethodCountCollector,
                NestingDepthCollector,
                ParameterCountCollector,
            )
            from ..cli.commands.analyze import _analyze_file, _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            collectors = [
                CognitiveComplexityCollector(),
                CyclomaticComplexityCollector(),
                NestingDepthCollector(),
                ParameterCountCollector(),
                MethodCountCollector(),
            ]

            project_metrics = ProjectMetrics(project_root=str(self.project_root))

            for file_path in files_to_analyze:
                try:
                    file_metrics = await _analyze_file(
                        file_path, parser_registry, collectors
                    )
                    if file_metrics and file_metrics.chunks:
                        project_metrics.files[str(file_path)] = file_metrics
                except Exception:  # nosec B112 - intentional skip of unparseable files
                    continue

            # Get top N complex files
            hotspots = project_metrics.get_hotspots(limit=limit)

            # Format response
            if not hotspots:
                response_text = "No complexity hotspots found"
            else:
                response_lines = [f"# Top {len(hotspots)} Complexity Hotspots\n"]

                for i, file_metrics in enumerate(hotspots, 1):
                    response_lines.extend(
                        [
                            f"## {i}. {Path(file_metrics.file_path).name}",
                            f"**Path:** `{file_metrics.file_path}`",
                            f"**Average Complexity:** {file_metrics.avg_complexity:.2f}",
                            f"**Max Complexity:** {file_metrics.max_complexity}",
                            f"**Total Complexity:** {file_metrics.total_complexity}",
                            f"**Functions:** {file_metrics.function_count}",
                            f"**Health Score:** {file_metrics.health_score:.2f}\n",
                        ]
                    )

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Hotspot detection failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Hotspot detection failed: {str(e)}")
                ],
                isError=True,
            )

    async def _check_circular_dependencies(
        self, args: dict[str, Any]
    ) -> CallToolResult:
        """Handle check_circular_dependencies tool call."""
        try:
            # Find analyzable files to build import graph
            from ..cli.commands.analyze import _find_analyzable_files

            parser_registry = ParserRegistry()
            files_to_analyze = _find_analyzable_files(
                self.project_root, None, None, parser_registry, None
            )

            if not files_to_analyze:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No analyzable files found in project",
                        )
                    ],
                    isError=True,
                )

            # Import circular dependency detection
            from ..analysis.collectors.coupling import build_import_graph

            # Build import graph for the project (reverse dependency graph)
            import_graph = build_import_graph(
                self.project_root, files_to_analyze, language="python"
            )

            # Convert to forward dependency graph for cycle detection
            # import_graph maps: module -> set of files that import it (reverse)
            # We need: file -> list of files it imports (forward)
            forward_graph: dict[str, list[str]] = {}

            # Build forward graph by reading imports from files
            for file_path in files_to_analyze:
                file_str = str(file_path.relative_to(self.project_root))
                if file_str not in forward_graph:
                    forward_graph[file_str] = []

                # For each module in import_graph, if this file imports it, add edge
                for module, importers in import_graph.items():
                    for importer in importers:
                        importer_str = str(
                            Path(importer).relative_to(self.project_root)
                            if Path(importer).is_absolute()
                            else importer
                        )
                        if importer_str == file_str:
                            # This file imports the module, add forward edge
                            if module not in forward_graph[file_str]:
                                forward_graph[file_str].append(module)

            # Detect circular dependencies using DFS
            def find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
                """Find all cycles in the import graph using DFS."""
                cycles = []
                visited = set()
                rec_stack = set()

                def dfs(node: str, path: list[str]) -> None:
                    visited.add(node)
                    rec_stack.add(node)
                    path.append(node)

                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            dfs(neighbor, path.copy())
                        elif neighbor in rec_stack:
                            # Found a cycle
                            try:
                                cycle_start = path.index(neighbor)
                                cycle = path[cycle_start:] + [neighbor]
                                # Normalize cycle representation to avoid duplicates
                                cycle_tuple = tuple(sorted(cycle))
                                if not any(
                                    tuple(sorted(c)) == cycle_tuple for c in cycles
                                ):
                                    cycles.append(cycle)
                            except ValueError:
                                pass

                    rec_stack.remove(node)

                for node in graph:
                    if node not in visited:
                        dfs(node, [])

                return cycles

            cycles = find_cycles(forward_graph)

            # Format response
            if not cycles:
                response_text = "No circular dependencies detected"
            else:
                response_lines = [f"# Circular Dependencies Found: {len(cycles)}\n"]

                for i, cycle in enumerate(cycles, 1):
                    response_lines.append(f"## Cycle {i}")
                    response_lines.append("```")
                    for j, node in enumerate(cycle):
                        if j < len(cycle) - 1:
                            response_lines.append(f"{node}")
                            response_lines.append("  ↓")
                        else:
                            response_lines.append(f"{node} (back to {cycle[0]})")
                    response_lines.append("```\n")

                response_text = "\n".join(response_lines)

            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )

        except Exception as e:
            logger.error(f"Circular dependency check failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Circular dependency check failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def _interpret_analysis(self, args: dict[str, Any]) -> CallToolResult:
        """Handle interpret_analysis tool call."""
        analysis_json_str = args.get("analysis_json", "")
        focus = args.get("focus", "summary")
        verbosity = args.get("verbosity", "normal")

        if not analysis_json_str:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="analysis_json parameter is required")
                ],
                isError=True,
            )

        try:
            import json

            from ..analysis.interpretation import AnalysisInterpreter, LLMContextExport

            # Parse JSON input
            analysis_data = json.loads(analysis_json_str)

            # Convert to LLMContextExport
            export = LLMContextExport(**analysis_data)

            # Create interpreter and generate interpretation
            interpreter = AnalysisInterpreter()
            interpretation = interpreter.interpret(
                export, focus=focus, verbosity=verbosity
            )

            return CallToolResult(
                content=[TextContent(type="text", text=interpretation)]
            )

        except json.JSONDecodeError as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Invalid JSON input: {str(e)}",
                    )
                ],
                isError=True,
            )
        except Exception as e:
            logger.error(f"Analysis interpretation failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Interpretation failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    def _get_threshold_config(self, preset: str) -> ThresholdConfig:
        """Get threshold configuration based on preset.

        Args:
            preset: Threshold preset ('strict', 'standard', or 'relaxed')

        Returns:
            ThresholdConfig instance
        """
        if preset == "strict":
            # Stricter thresholds
            config = ThresholdConfig()
            config.complexity.cognitive_a = 3
            config.complexity.cognitive_b = 7
            config.complexity.cognitive_c = 15
            config.complexity.cognitive_d = 20
            config.smells.long_method_lines = 30
            config.smells.high_complexity = 10
            config.smells.too_many_parameters = 3
            config.smells.deep_nesting_depth = 3
            return config
        elif preset == "relaxed":
            # More relaxed thresholds
            config = ThresholdConfig()
            config.complexity.cognitive_a = 7
            config.complexity.cognitive_b = 15
            config.complexity.cognitive_c = 25
            config.complexity.cognitive_d = 40
            config.smells.long_method_lines = 75
            config.smells.high_complexity = 20
            config.smells.too_many_parameters = 7
            config.smells.deep_nesting_depth = 5
            return config
        else:
            # Standard (default)
            return ThresholdConfig()


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
    async def handle_list_tools() -> list[Tool]:
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

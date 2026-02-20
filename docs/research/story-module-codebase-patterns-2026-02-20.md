# Story Module Implementation - Codebase Pattern Analysis

**Date**: 2026-02-20
**Purpose**: Document existing patterns for implementing the new `story` module
**Scope**: File structure, model patterns, CLI integration, MCP integration, rendering patterns

---

## 1. Project Structure Overview

```
src/mcp_vector_search/
├── analysis/                    # Code analysis modules
│   ├── collectors/             # Metric collectors (complexity, coupling, smells)
│   ├── reporters/              # Output formatters (markdown, console, SARIF)
│   ├── storage/                # Metric storage and trend tracking
│   ├── visualizer/             # D3.js visualization (HTML export)
│   └── baseline/               # Baseline management
├── cli/
│   ├── commands/               # Command implementations
│   │   ├── analyze.py          # analyze app (typer)
│   │   ├── visualize/          # visualize subcommand module
│   │   │   ├── cli.py
│   │   │   ├── templates/      # HTML templates
│   │   │   └── exporters/      # JSON/HTML exporters
│   │   ├── search.py
│   │   ├── chat.py
│   │   └── ...
│   └── main.py                 # Main CLI app
├── core/                       # Core functionality
│   ├── models.py               # Pydantic/dataclass models
│   ├── search.py               # SemanticSearchEngine
│   ├── llm_client.py           # LLMClient (OpenAI, OpenRouter, Bedrock)
│   ├── knowledge_graph.py      # KnowledgeGraph (KuzuDB)
│   └── ...
├── mcp/                        # MCP server
│   ├── server.py               # Main MCP server
│   ├── tool_schemas.py         # MCP tool definitions
│   ├── analysis_handlers.py   # Analysis tool handlers
│   ├── search_handlers.py     # Search tool handlers
│   └── wiki_handlers.py        # Wiki tool handlers
└── parsers/                    # Language parsers (tree-sitter)
```

**Key Insight**: Analysis-related features go in `analysis/`, CLI commands in `cli/commands/`, MCP handlers in `mcp/`, and data models in module-specific files.

---

## 2. Core Models Pattern (`core/models.py`)

### Import Style
```python
"""Data models for MCP Vector Search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
```

**Pattern**: Use `dataclasses` for internal data structures, `Pydantic BaseModel` for external API boundaries (MCP, CLI).

### Dataclass Pattern (CodeChunk)
```python
@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str
    chunk_type: str = "code"  # Default values after required fields
    function_name: str | None = None

    # Enhancement fields grouped by feature
    # Enhancement 1: Complexity scoring
    complexity_score: float = 0.0

    # Enhancement 7: Git blame metadata
    last_author: str | None = None
    last_modified: str | None = None
    commit_hash: str | None = None

    def __post_init__(self) -> None:
        """Initialize default values and generate IDs."""
        if self.imports is None:
            self.imports = []
        # Generate unique ID
        if self.chunk_id is None:
            import hashlib
            id_string = f"{self.file_path}:{self.chunk_type}:{self.start_line}"
            self.chunk_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]

    @property
    def id(self) -> str:
        """Generate unique ID for this chunk."""
        return f"{self.file_path}:{self.start_line}:{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "file_path": str(self.file_path),
            # ...
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            file_path=Path(data["file_path"]),
            # ...
        )
```

### Pydantic Pattern (SearchResult)
```python
class SearchResult(BaseModel):
    """Represents a search result with metadata."""

    content: str = Field(..., description="The matched code content")
    file_path: Path = Field(..., description="Path to the source file")
    start_line: int = Field(..., description="Starting line number")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    rank: int = Field(..., description="Result rank in search results")

    # Optional fields with defaults
    function_name: str | None = Field(default=None, description="Function name if applicable")
    context_before: list[str] = Field(default=[], description="Lines before the match")

    # Quality metrics (optional)
    cognitive_complexity: int | None = Field(default=None, description="Cognitive complexity score")
    code_smells: list[str] = Field(default=[], description="Detected code smells")

    class Config:
        arbitrary_types_allowed = True  # Allow Path type

    @property
    def line_count(self) -> int:
        """Get the number of lines in this result."""
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "content": self.content,
            "file_path": str(self.file_path),
            # ...
        }
        # Add optional fields conditionally
        if self.cognitive_complexity is not None:
            result["cognitive_complexity"] = self.cognitive_complexity
        return result
```

**Key Patterns**:
1. Use `Field(..., description="...")` for required fields
2. Use `Field(default=..., description="...")` for optional fields
3. Use `str | None` (PEP 604 union syntax) instead of `Optional[str]`
4. Add `Config` class for pydantic settings
5. Implement `@property` methods for computed fields
6. Implement `to_dict()` for JSON serialization
7. Group related fields with comments

---

## 3. Visualizer Schemas Pattern (`analysis/visualizer/schemas.py`)

### Export Metadata Pattern
```python
"""JSON export schema for structural code analysis results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExportMetadata(BaseModel):
    """Metadata about the export itself.

    Tracks version information, generation timestamp, and git context
    to enable historical comparison and tool compatibility checks.

    Attributes:
        version: Schema version (e.g., "1.0.0") for compatibility tracking
        generated_at: UTC timestamp when export was generated
        tool_version: mcp-vector-search version that generated the export
        project_root: Absolute path to project root directory
        git_commit: Git commit SHA if available (optional)
        git_branch: Git branch name if available (optional)
    """

    version: str = Field(
        default="1.0.0",
        description="Schema version for compatibility tracking"
    )
    generated_at: datetime = Field(
        description="UTC timestamp when analysis was performed"
    )
    tool_version: str = Field(
        description="Version of mcp-vector-search that generated this export"
    )
    project_root: str = Field(
        description="Absolute path to project root directory"
    )
    git_commit: str | None = Field(
        default=None,
        description="Git commit SHA (if available)"
    )
    git_branch: str | None = Field(
        default=None,
        description="Git branch name (if available)"
    )
```

### Nested Schema Pattern
```python
class FileDetail(BaseModel):
    """Complete metrics for a single file."""

    path: str = Field(description="Relative path from project root")
    language: str = Field(description="Programming language")
    lines_of_code: int = Field(ge=0, description="Total lines of code")

    # Aggregate complexity metrics
    cyclomatic_complexity: int = Field(ge=0, description="Sum of cyclomatic complexity")

    # Collections
    functions: list[FunctionMetrics] = Field(
        default_factory=list,
        description="Function-level metrics"
    )
    classes: list[ClassMetrics] = Field(
        default_factory=list,
        description="Class-level metrics"
    )
    smells: list[SmellLocation] = Field(
        default_factory=list,
        description="Detected code smells"
    )
```

### Root Export Schema Pattern
```python
class AnalysisExport(BaseModel):
    """Root schema for complete analysis export.

    Top-level container for all analysis data including metadata,
    summary statistics, file details, dependencies, and trends.

    Attributes:
        metadata: Export metadata and version information
        summary: Project-level summary statistics
        files: Detailed metrics for each analyzed file
        dependencies: Dependency graph and coupling analysis
        trends: Historical trend data (optional)
    """

    metadata: ExportMetadata = Field(description="Export metadata and versioning")
    summary: MetricsSummary = Field(description="Project-level summary statistics")
    files: list[FileDetail] = Field(
        default_factory=list,
        description="File-level metrics"
    )
    dependencies: DependencyGraph = Field(
        description="Dependency graph and coupling analysis"
    )
    trends: TrendData | None = Field(
        default=None,
        description="Historical trend data (optional)"
    )


def generate_json_schema() -> dict[str, Any]:
    """Generate JSON Schema for documentation and validation."""
    return AnalysisExport.model_json_schema()
```

**Key Patterns**:
1. Use `from __future__ import annotations` for forward references
2. Rich docstrings with `Attributes:` section
3. Nested schemas for complex data structures
4. `default_factory=list` for mutable defaults
5. Field validation with `ge=0` (greater than or equal)
6. Optional fields with `| None` and `default=None`
7. Root schema as top-level container
8. Helper function for JSON Schema generation

---

## 4. CLI Pattern (`cli/main.py`)

### App Creation
```python
"""Main CLI application for MCP Vector Search."""

import typer
from rich.console import Console

from ..cli.commands.analyze import analyze_app

console = Console()

# Create main Typer app
app = typer.Typer(
    name="mcp-vector-search",
    help="🔍 MCP Vector Search - Semantic Code Search CLI",
    add_completion=False,
    rich_markup_mode="rich",
)

# Register subcommands
app.add_typer(analyze_app, name="analyze", help="📈 Analyze code complexity")
```

### Command Implementation (`cli/commands/analyze.py`)
```python
"""Analyze command implementation."""

import typer
from rich.console import Console

console = Console()

# Create Typer app for analyze subcommand
analyze_app = typer.Typer(
    name="analyze",
    help="Analyze code complexity and quality",
    rich_markup_mode="rich",
)


@analyze_app.command("project")
def analyze_project(
    output: str = typer.Option(
        "console",
        "--output",
        "-o",
        help="Output format: console, json, markdown"
    ),
    threshold_preset: str = typer.Option(
        "standard",
        "--preset",
        "-p",
        help="Threshold preset: strict, standard, relaxed"
    ),
) -> None:
    """Analyze project-wide code quality metrics.

    Examples:
        mcp-vector-search analyze project
        mcp-vector-search analyze project --output markdown
        mcp-vector-search analyze project --preset strict
    """
    console.print("[bold blue]Analyzing project...[/bold blue]")
    # Implementation here
```

**Key Patterns**:
1. Create separate Typer apps for subcommands
2. Use `typer.Option()` with short/long flags
3. Rich help text with emoji
4. Docstrings with examples
5. Import console from rich
6. Register app in main.py with `app.add_typer()`

---

## 5. MCP Integration Pattern

### Tool Schema (`mcp/tool_schemas.py`)
```python
"""MCP tool schema definitions for vector search functionality."""

from mcp.types import Tool


def get_tool_schemas() -> list[Tool]:
    """Get all MCP tool schema definitions."""
    return [
        _get_analyze_project_schema(),
        _get_analyze_file_schema(),
        # ...
    ]


def _get_analyze_project_schema() -> Tool:
    """Get analyze_project tool schema."""
    return Tool(
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
    )
```

### MCP Server Integration (`mcp/server.py`)
```python
"""MCP server implementation for MCP Vector Search."""

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolResult, TextContent

from .analysis_handlers import AnalysisHandlers
from .tool_schemas import get_tool_schemas


class MCPVectorSearchServer:
    """MCP server for vector search functionality."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        # Initialize handlers (lazy initialization)
        self._analysis_handlers: AnalysisHandlers | None = None

    async def initialize(self) -> None:
        """Initialize the search engine and database."""
        # Initialize handlers
        self._analysis_handlers = AnalysisHandlers(self.project_root)

    def get_tools(self):
        """Get available MCP tools."""
        return get_tool_schemas()

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls by delegating to appropriate handlers."""
        tool_name = request.params.name
        args = request.params.arguments

        # Delegate to analysis handlers
        if tool_name == "analyze_project":
            return await self._analysis_handlers.handle_analyze_project(args)
        elif tool_name == "analyze_file":
            return await self._analysis_handlers.handle_analyze_file(args)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
                isError=True,
            )
```

### Handler Implementation (`mcp/analysis_handlers.py`)
```python
"""Analysis tool handlers for MCP server."""

from mcp.types import CallToolResult, TextContent
from pathlib import Path


class AnalysisHandlers:
    """Handlers for analysis MCP tools."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    async def handle_analyze_project(self, args: dict) -> CallToolResult:
        """Handle analyze_project tool call."""
        try:
            threshold_preset = args.get("threshold_preset", "standard")
            output_format = args.get("output_format", "summary")

            # Perform analysis
            # ...

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Analysis complete: {result_summary}"
                )],
                isError=False,
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Analysis failed: {str(e)}"
                )],
                isError=True,
            )
```

**Key Patterns**:
1. Tool schemas in separate file (`tool_schemas.py`)
2. Handlers in separate files by domain (`analysis_handlers.py`, `search_handlers.py`)
3. Server delegates to handlers (`server.py`)
4. Lazy initialization of handlers
5. Error handling with `CallToolResult(isError=True)`
6. Tool schemas follow JSON Schema format

---

## 6. LLM Client Interface (`core/llm_client.py`)

### Class Signature
```python
"""LLM client for intelligent code search using OpenAI, OpenRouter, or AWS Bedrock API."""

from typing import Literal, Any
from collections.abc import AsyncIterator

LLMProvider = Literal["openai", "openrouter", "bedrock"]
IntentType = Literal["find", "answer", "analyze"]


class LLMClient:
    """Client for LLM-powered intelligent search orchestration."""

    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "openrouter": "anthropic/claude-opus-4.5",
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        provider: LLMProvider | None = None,
        think: bool = False,
    ) -> None:
        """Initialize LLM client."""
        self.provider = provider or self._auto_detect_provider()
        self.model = model or self.DEFAULT_MODELS[self.provider]
        self.timeout = timeout

    async def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Make chat completion request to OpenAI, OpenRouter, or Bedrock API."""
        # Implementation

    async def generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate answer to user question using codebase context."""
        system_prompt = f"""You are a helpful code assistant analyzing a codebase.

Code Context:
{context}

Guidelines:
- Be concise but thorough in explanations
- Reference specific functions, classes, or files when relevant
- Use code examples from the context when helpful"""

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})

        response = await self._chat_completion(messages)
        return response["choices"][0]["message"]["content"]

    async def stream_chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream chat completion response chunk by chunk."""
        async for chunk in self._stream_impl(messages):
            yield chunk
```

**Key Patterns**:
1. Multi-provider support with auto-detection
2. Literal types for provider/intent validation
3. Async methods for all LLM calls
4. Streaming support with `AsyncIterator`
5. Default models per provider
6. System prompts with clear guidelines
7. Conversation history support

---

## 7. Search Engine Interface (`core/search.py`)

### Class Signature
```python
"""Semantic search engine for MCP Vector Search."""

from pathlib import Path
from typing import Any

from .database import VectorDatabase
from .models import SearchResult


class SemanticSearchEngine:
    """Semantic search engine for code search."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.3,
        enable_kg: bool = True,
    ) -> None:
        """Initialize semantic search engine."""
        self.database = database
        self.project_root = project_root
        self.similarity_threshold = similarity_threshold
        self.enable_kg = enable_kg

        # Initialize helper components
        self._query_processor = QueryProcessor(base_threshold=similarity_threshold)
        self._result_enhancer = ResultEnhancer()

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
        include_context: bool = True,
    ) -> list[SearchResult]:
        """Perform semantic search for code."""
        # Implementation

    async def search_similar(
        self,
        file_path: Path,
        function_name: str | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Find code similar to a specific function or file."""
        # Implementation

    async def get_search_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        db_stats = await self.database.get_stats()
        return {
            "total_chunks": db_stats.total_chunks,
            "languages": db_stats.languages,
            "project_root": str(self.project_root),
        }
```

**Key Patterns**:
1. Dependency injection (database, project_root)
2. Helper components initialized in `__init__`
3. Async search methods
4. Optional parameters with defaults
5. Return typed results (`list[SearchResult]`)
6. Statistics method for monitoring

---

## 8. HTML Template Pattern (`cli/commands/visualize/templates/base.py`)

### Self-Contained HTML Generation
```python
"""HTML template generation for the visualization."""

import time
from mcp_vector_search import __build__, __version__
from .scripts import get_all_scripts
from .styles import get_all_styles


def generate_html_template() -> str:
    """Generate the complete HTML template for visualization.

    Returns:
        Complete HTML string with embedded CSS and JavaScript
    """
    # Add timestamp for cache busting
    build_timestamp = int(time.time())

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Chunk Relationship Graph</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <!-- Build: {build_timestamp} -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
{get_all_styles()}
    </style>
</head>
<body>
    <div id="controls">
        <h1>🔍 Code Tree</h1>
        <div class="version-badge">v{__version__} (build {__build__})</div>
        <!-- Controls here -->
    </div>

    <div id="main-container">
        <svg id="graph"></svg>
    </div>

    <script>
{get_all_scripts()}
    </script>
</body>
</html>"""
    return html


def inject_data(html: str, data: dict) -> str:
    """Inject graph data into HTML template (optional for static export)."""
    return html
```

**Key Patterns**:
1. Self-contained HTML (all CSS/JS embedded)
2. Cache-busting with timestamp
3. External CDN libraries (D3.js, highlight.js)
4. Separate functions for styles and scripts
5. Version/build info in template
6. Optional data injection function

---

## 9. Visualize Server Pattern (`cli/commands/visualize/server.py`)

### FastAPI Server with Streaming
```python
"""HTTP server for visualization with streaming JSON support."""

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator

import orjson
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from rich.console import Console

console = Console()


def create_app(viz_dir: Path) -> FastAPI:
    """Create FastAPI application for visualization server."""
    app = FastAPI(title="MCP Vector Search Visualization")

    @app.get("/api/graph-status")
    async def graph_status() -> Response:
        """Get graph data generation status."""
        graph_file = viz_dir / "chunk-graph.json"
        if not graph_file.exists():
            return Response(
                content='{"ready": false, "size": 0}',
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )

        size = graph_file.stat().st_size
        is_ready = size > 100
        return Response(
            content=f'{{"ready": {str(is_ready).lower()}, "size": {size}}}',
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/graph")
    async def get_graph_data() -> Response:
        """Get graph data for D3 tree visualization."""
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found", "nodes": [], "links": []}',
                status_code=404,
                media_type="application/json",
            )

        with open(graph_file, "rb") as f:
            data = orjson.loads(f.read())

        return Response(
            content=orjson.dumps({
                "nodes": data.get("nodes", []),
                "links": data.get("links", [])
            }),
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/graph-data")
    async def stream_graph_data() -> StreamingResponse:
        """Stream chunk-graph.json in 100KB chunks."""
        graph_file = viz_dir / "chunk-graph.json"

        async def generate_chunks() -> AsyncGenerator[bytes, None]:
            chunk_size = 100 * 1024  # 100KB chunks
            with open(graph_file, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
                    await asyncio.sleep(0.01)

        return StreamingResponse(
            generate_chunks(),
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve index.html with no-cache headers."""
        return FileResponse(
            viz_dir / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


def start_visualization_server(port: int, viz_dir: Path, auto_open: bool = True) -> None:
    """Start HTTP server for visualization."""
    app = create_app(viz_dir)
    url = f"http://localhost:{port}"

    console.print(f"[green]✓[/green] Visualization server running at {url}")

    if auto_open:
        import webbrowser
        webbrowser.open(url)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
```

**Key Patterns**:
1. FastAPI for HTTP server
2. `orjson` for fast JSON serialization
3. Streaming responses for large data
4. No-cache headers for index.html
5. Status endpoint for loading indicators
6. Rich console for output
7. Auto-open browser option
8. Uvicorn as ASGI server

---

## 10. Markdown Reporter Pattern (`analysis/reporters/markdown.py`)

### Markdown Generation
```python
"""Markdown reporter for code analysis results."""

from datetime import datetime
from pathlib import Path


class MarkdownReporter:
    """Markdown reporter for generating analysis reports."""

    def generate_analysis_report(
        self,
        metrics: ProjectMetrics,
        smells: list[CodeSmell] | None = None,
        output_path: Path | None = None,
    ) -> str:
        """Generate full analysis report in markdown format.

        Returns:
            Path to generated markdown file
        """
        if output_path is None:
            output_path = Path.cwd()

        # Determine output directory
        if output_path.is_dir():
            output_dir = output_path
        else:
            output_dir = output_path.parent if output_path.parent.exists() else Path.cwd()

        output_file = output_dir / "mcp-vector-search-analysis.md"

        # Generate markdown content
        content = self._build_analysis_markdown(metrics, smells or [])

        # Write to file
        output_file.write_text(content, encoding="utf-8")

        return str(output_file)

    def _build_analysis_markdown(
        self,
        metrics: ProjectMetrics,
        smells: list[CodeSmell]
    ) -> str:
        """Build full analysis markdown content."""
        lines: list[str] = []

        # Header
        lines.append("# MCP Vector Search - Code Analysis Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Files analyzed**: {metrics.total_files}")
        lines.append(f"- **Total lines**: {metrics.total_lines:,}")
        lines.append("")

        # Complexity distribution
        lines.append("## Complexity Distribution")
        lines.append("")
        lines.append("| Grade | Description | Count | Percentage |")
        lines.append("|-------|------------|-------|------------|")

        for grade in ["A", "B", "C", "D", "F"]:
            count = distribution.get(grade, 0)
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            lines.append(f"| {grade} | {description} | {count} | {percentage:.1f}% |")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by mcp-vector-search analyze command*")
        lines.append("")

        return "\n".join(lines)
```

**Key Patterns**:
1. Class-based reporter
2. Build content in `_build_*` private method
3. Use `list[str]` to accumulate lines
4. Markdown tables with proper formatting
5. Timestamp in ISO format
6. Handle output_path edge cases
7. Return file path as string
8. Use `Path.write_text()` for file writing

---

## Key Takeaways for Story Module

### Recommended File Structure
```
src/mcp_vector_search/story/
├── __init__.py
├── models.py                # Pydantic models (CommitStory, etc.)
├── extractor.py             # Git/GitHub data extraction
├── analyzer.py              # Semantic analysis
├── synthesizer.py           # LLM synthesis
├── recipe.py                # GitStory recipe for synthesis
├── renderers/
│   ├── __init__.py
│   ├── json_renderer.py     # JSON export
│   ├── markdown_renderer.py # Markdown report
│   └── html_renderer.py     # D3.js timeline visualization
└── templates/
    ├── __init__.py
    ├── base.py              # HTML template generation
    ├── styles.py            # CSS for timeline
    └── scripts.py           # D3.js timeline scripts
```

### CLI Integration
```python
# cli/commands/story.py
story_app = typer.Typer(name="story", help="📖 Generate commit story narratives")

@story_app.command("generate")
def generate_story(...):
    """Generate commit story for a git range."""
    pass

# cli/main.py
from .commands.story import story_app
app.add_typer(story_app, name="story", help="📖 Generate commit story narratives")
```

### MCP Integration
```python
# mcp/tool_schemas.py
def _get_story_generate_schema() -> Tool:
    return Tool(
        name="story_generate",
        description="Generate narrative commit story for a git range",
        inputSchema={...},
    )

# mcp/story_handlers.py
class StoryHandlers:
    async def handle_story_generate(self, args: dict) -> CallToolResult:
        # Implementation
        pass

# mcp/server.py
self._story_handlers = StoryHandlers(self.project_root)
# In call_tool():
elif tool_name == "story_generate":
    return await self._story_handlers.handle_story_generate(args)
```

### Model Design
```python
# story/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

class CommitStoryMetadata(BaseModel):
    """Metadata about the story generation."""
    version: str = Field(default="1.0.0")
    generated_at: datetime
    git_range: str
    commit_count: int
    branch: str | None = None

class CommitChange(BaseModel):
    """A single code change in a commit."""
    file_path: Path
    change_type: str  # "added", "modified", "deleted"
    lines_added: int
    lines_deleted: int
    semantic_summary: str  # LLM-generated

class Commit(BaseModel):
    """A git commit with semantic analysis."""
    sha: str
    message: str
    author: str
    timestamp: datetime
    changes: list[CommitChange]
    semantic_theme: str  # LLM-classified
    narrative: str  # LLM-generated

class CommitStory(BaseModel):
    """Complete commit story with narrative."""
    metadata: CommitStoryMetadata
    commits: list[Commit]
    narrative_summary: str  # LLM-generated
    semantic_timeline: list[str]  # Key events
    character_analysis: dict[str, Any]  # Who did what
```

This document provides all the patterns you need to implement the story module following the existing codebase conventions. Let me know if you need clarification on any pattern!

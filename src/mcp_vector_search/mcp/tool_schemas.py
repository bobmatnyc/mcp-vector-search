"""MCP tool schema definitions for vector search functionality."""

from mcp.types import Tool


def get_tool_schemas() -> list[Tool]:
    """Get all MCP tool schema definitions.

    Returns:
        List of Tool objects defining available MCP tools
    """
    return [
        _get_search_code_schema(),
        _get_search_similar_schema(),
        _get_search_context_schema(),
        _get_project_status_schema(),
        _get_index_project_schema(),
        _get_analyze_project_schema(),
        _get_analyze_file_schema(),
        _get_find_smells_schema(),
        _get_complexity_hotspots_schema(),
        _get_circular_dependencies_schema(),
        _get_interpret_analysis_schema(),
    ]


def _get_search_code_schema() -> Tool:
    """Get search_code tool schema."""
    return Tool(
        name="search_code",
        description="Search codebase using natural language queries (text-to-code search). Use when you know what functionality you're looking for but not where it's implemented. Example: 'authentication middleware' or 'database connection pooling' to find relevant code.",
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
    )


def _get_search_similar_schema() -> Tool:
    """Get search_similar tool schema."""
    return Tool(
        name="search_similar",
        description="Find code snippets similar to a specific file or function (code-to-code similarity). Use when looking for duplicate code, similar patterns, or related implementations. Example: 'Find functions similar to auth_handler.py' to discover related authentication code.",
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
    )


def _get_search_context_schema() -> Tool:
    """Get search_context tool schema."""
    return Tool(
        name="search_context",
        description="Search for code using rich contextual descriptions with optional focus areas. Use when you need broader context around specific concerns. Example: 'code handling user sessions' with focus_areas=['security', 'authentication'] to find session management with security emphasis.",
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
    )


def _get_project_status_schema() -> Tool:
    """Get get_project_status tool schema."""
    return Tool(
        name="get_project_status",
        description="Get project indexing status and statistics",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


def _get_index_project_schema() -> Tool:
    """Get index_project tool schema."""
    return Tool(
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
    )


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


def _get_analyze_file_schema() -> Tool:
    """Get analyze_file tool schema."""
    return Tool(
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
    )


def _get_find_smells_schema() -> Tool:
    """Get find_smells tool schema."""
    return Tool(
        name="find_smells",
        description="Identify code quality issues, anti-patterns, bad practices, and technical debt. Detects Long Methods, Deep Nesting, Long Parameter Lists, God Classes, and Complex Methods. Use when assessing code quality, finding refactoring opportunities, or identifying maintainability issues.",
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
    )


def _get_complexity_hotspots_schema() -> Tool:
    """Get get_complexity_hotspots tool schema."""
    return Tool(
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
    )


def _get_circular_dependencies_schema() -> Tool:
    """Get check_circular_dependencies tool schema."""
    return Tool(
        name="check_circular_dependencies",
        description="Returns circular dependency cycles",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


def _get_interpret_analysis_schema() -> Tool:
    """Get interpret_analysis tool schema."""
    return Tool(
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
    )

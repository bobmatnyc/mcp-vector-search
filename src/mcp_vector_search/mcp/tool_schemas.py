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
        _get_embed_chunks_schema(),
        _get_analyze_project_schema(),
        _get_analyze_file_schema(),
        _get_find_smells_schema(),
        _get_complexity_hotspots_schema(),
        _get_circular_dependencies_schema(),
        _get_interpret_analysis_schema(),
        _get_save_report_schema(),
        _get_review_repository_schema(),
        _get_review_pull_request_schema(),
        _get_wiki_generate_schema(),
        _get_kg_build_schema(),
        _get_kg_stats_schema(),
        _get_kg_query_schema(),
        _get_story_generate_schema(),
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
                "project_name": {
                    "type": "string",
                    "description": "Filter by monorepo subproject name",
                },
                "files": {
                    "type": "string",
                    "description": "Filter by file patterns (e.g., '*.py' or 'src/*.js')",
                },
                "diversity": {
                    "type": "number",
                    "description": "MMR diversity parameter (0.0=pure relevance, 1.0=max diversity)",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "use_mmr": {
                    "type": "boolean",
                    "description": "Enable MMR diversity filtering (default: true)",
                    "default": True,
                },
                "expand": {
                    "type": "boolean",
                    "description": "Enable query expansion with code synonyms (default: true)",
                    "default": True,
                },
                "use_rerank": {
                    "type": "boolean",
                    "description": "Enable cross-encoder reranking for higher precision (default: true)",
                    "default": True,
                },
                "rerank_top_n": {
                    "type": "integer",
                    "description": "Number of candidates to retrieve before reranking (default: 50)",
                    "default": 50,
                    "minimum": 10,
                    "maximum": 200,
                },
                "search_mode": {
                    "type": "string",
                    "description": "Search mode: 'vector' (semantic only), 'bm25' (keyword only), or 'hybrid' (combined, default)",
                    "enum": ["vector", "bm25", "hybrid"],
                    "default": "hybrid",
                },
                "hybrid_alpha": {
                    "type": "number",
                    "description": "Weight for vector search in hybrid mode (0.0-1.0, default 0.7). 1.0=pure vector, 0.0=pure BM25",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
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
        description="Index or reindex the project codebase. IMPORTANT: Always call get_project_status first to check if an index already exists. Only call this tool if: (1) No index exists (total_chunks: 0), OR (2) User explicitly requests reindexing with force=true. This operation can take 2-5 minutes for large projects.",
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


def _get_embed_chunks_schema() -> Tool:
    """Get embed_chunks tool schema."""
    return Tool(
        name="embed_chunks",
        description="Generate embeddings for indexed code chunks. Only embeds chunks that haven't been embedded yet (incremental). Use fresh=true to re-embed everything. Run index_project first to chunk files, then embed_chunks to generate embeddings.",
        inputSchema={
            "type": "object",
            "properties": {
                "fresh": {
                    "type": "boolean",
                    "description": "Clear vectors and re-embed all chunks from scratch",
                    "default": False,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of chunks per embedding batch (default: 512)",
                    "default": 512,
                    "minimum": 100,
                    "maximum": 10000,
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


def _get_save_report_schema() -> Tool:
    """Get save_report tool schema."""
    return Tool(
        name="save_report",
        description="Save analysis or search results as a markdown file for documentation",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Markdown content to save",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["analysis", "search", "smells", "hotspots", "custom"],
                    "default": "custom",
                    "description": "Type of report for filename generation",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional custom output path (file or directory)",
                },
                "filename_prefix": {
                    "type": "string",
                    "description": "Optional prefix for auto-generated filename",
                },
            },
            "required": ["content"],
        },
    )


def _get_review_repository_schema() -> Tool:
    """Get review_repository tool schema."""
    return Tool(
        name="review_repository",
        description="""Run an AI-powered code review on the indexed codebase using vector search and LLM analysis.

**What it does:**
- Performs specialized code reviews (security, architecture, performance)
- Uses targeted vector search queries to find relevant code
- Analyzes code with LLM to identify issues and provide recommendations
- Returns structured findings with severity levels, categories, and recommendations

**Review types:**
- **security**: Finds vulnerabilities (SQL injection, XSS, authentication issues, etc.)
- **architecture**: Identifies design issues (tight coupling, missing abstractions, etc.)
- **performance**: Detects performance problems (N+1 queries, inefficient algorithms, etc.)

**Output format:**
Returns structured findings with:
- title: Brief description of the issue
- description: Detailed explanation
- severity: critical, high, medium, low, info
- file_path: Location of the issue
- start_line/end_line: Precise location in file
- category: Issue category (e.g., "SQL Injection", "Tight Coupling")
- recommendation: How to fix the issue
- confidence: Confidence score (0.0-1.0)
- cwe_id: Common Weakness Enumeration ID (for security issues)

**Example usage:**
{"review_type": "security"}  # Review entire codebase for security issues
{"review_type": "architecture", "scope": "src/auth"}  # Review auth module
{"review_type": "performance", "max_chunks": 50}  # Analyze more code
{"review_type": "security", "changed_only": true}  # Only review changed files

**Example response:**
{
  "status": "success",
  "review_type": "security",
  "scope": "entire project",
  "findings": [
    {
      "title": "SQL Injection Vulnerability",
      "description": "User input is concatenated directly into SQL query",
      "severity": "critical",
      "file_path": "src/db/users.py",
      "start_line": 42,
      "end_line": 45,
      "category": "SQL Injection",
      "recommendation": "Use parameterized queries or an ORM",
      "confidence": 0.95,
      "cwe_id": "CWE-89"
    }
  ],
  "summary": "Found 1 security issue(s):\\n  - CRITICAL: 1",
  "context_chunks_used": 25,
  "kg_relationships_used": 12,
  "model_used": "claude-3-5-sonnet-20241022",
  "duration_seconds": 8.3
}""",
        inputSchema={
            "type": "object",
            "properties": {
                "review_type": {
                    "type": "string",
                    "enum": ["security", "architecture", "performance"],
                    "description": "Type of review to perform",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional path scope (e.g., 'src/auth', 'core/')",
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Maximum code chunks to analyze (default: 30)",
                    "default": 30,
                    "minimum": 10,
                    "maximum": 100,
                },
                "changed_only": {
                    "type": "boolean",
                    "description": "Only review files changed in git (uncommitted changes)",
                    "default": False,
                },
            },
            "required": ["review_type"],
        },
    )


def _get_wiki_generate_schema() -> Tool:
    """Get wiki_generate tool schema."""
    return Tool(
        name="wiki_generate",
        description="Generate codebase wiki/ontology showing hierarchical concept organization with LLM-powered semantic grouping",
        inputSchema={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force regeneration, ignoring cache",
                    "default": False,
                },
                "ttl_hours": {
                    "type": "integer",
                    "description": "Cache TTL in hours (default: 24)",
                    "minimum": 1,
                    "default": 24,
                },
                "no_llm": {
                    "type": "boolean",
                    "description": "Skip LLM semantic grouping (flat ontology only)",
                    "default": False,
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "summary"],
                    "description": "Output format (json for full ontology, summary for overview)",
                    "default": "summary",
                },
            },
            "required": [],
        },
    )


def _get_kg_build_schema() -> Tool:
    """Get kg_build tool schema."""
    return Tool(
        name="kg_build",
        description="""Build or rebuild the Knowledge Graph from indexed code.

**What it does:**
- Extracts code entities (functions, classes, modules) and relationships (calls, imports, inheritance)
- Creates documentation-to-code mappings (DOCUMENTS, REFERENCES)
- Extracts authorship from git history (AUTHORED, PART_OF)
- Builds queryable graph for code navigation and exploration

**When to use:**
- After initial project indexing
- After significant code changes
- When KG queries return empty results
- To refresh relationships after refactoring

**Performance:**
- ~2-3 minutes for 10K files (batch optimized)
- Use skip_documents for faster builds (25-30% faster)
- Progress tracking available via CLI

**Example usage:**
{"force": false}  # Build only if not exists
{"force": true}  # Force rebuild
{"force": true, "skip_documents": true}  # Fast rebuild
{"limit": 100}  # Test with 100 chunks

**Example response:**
{
  "status": "success",
  "statistics": {
    "entities": 3243,
    "relationships": {
      "calls": 3944,
      "imports": 3243,
      "inherits": 289,
      "contains": 5821
    }
  }
}""",
        inputSchema={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force rebuild even if graph exists. Default: false",
                    "default": False,
                },
                "skip_documents": {
                    "type": "boolean",
                    "description": "Skip DOCUMENTS extraction for faster build (25-30% faster). Default: false",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Limit chunks to process (for testing). Default: no limit",
                },
            },
            "required": [],
        },
    )


def _get_kg_stats_schema() -> Tool:
    """Get kg_stats tool schema."""
    return Tool(
        name="kg_stats",
        description="""Get Knowledge Graph statistics and health.

**What it returns:**
- Entity counts: code entities, doc sections, tags, persons, projects
- Relationship counts: calls, imports, inherits, contains, documents, authored, etc.
- Database size and location
- Health status indicators

**When to use:**
- Check if KG is built and populated
- Verify KG health after indexing
- Monitor KG size for large repos
- Validate before running queries
- Troubleshoot KG issues

**Example usage:**
{}  # No parameters required

**Example response:**
{
  "status": "success",
  "statistics": {
    "total_entities": 15609,
    "code_entities": 3243,
    "doc_sections": 10483,
    "tags": 1060,
    "persons": 4,
    "projects": 1,
    "database_path": "/path/.mcp-vector-search/knowledge_graph/kg.db",
    "relationships": {
      "calls": 3944,
      "imports": 3243,
      "inherits": 289,
      "contains": 5821,
      "documents": 2864,
      "authored": 1276
    }
  }
}""",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )


def _get_kg_query_schema() -> Tool:
    """Get kg_query tool schema."""
    return Tool(
        name="kg_query",
        description="""Query entity relationships in the Knowledge Graph.

**What it does:**
- Finds entities related to a given entity
- Supports filtering by relationship type
- Returns relationship metadata and paths
- Traverses graph within specified hops

**Relationship types:**
- **calls**: Functions called BY this entity
- **called_by**: Functions that CALL this entity
- **imports**: Modules imported BY this entity
- **imported_by**: Modules that IMPORT this entity
- **inherits**: Classes this entity INHERITS FROM (parents)
- **inherited_by**: Classes that INHERIT FROM this entity (children)
- **contains**: Entities CONTAINED BY this entity (module contains functions)
- **contained_by**: Entities that CONTAIN this entity

**Query patterns:**
1. **Call graph discovery**: Find all functions a method calls
2. **Impact analysis**: Find what calls a function (callers)
3. **Inheritance hierarchy**: Find parent/child classes
4. **Module structure**: Find what a module contains
5. **Dependency analysis**: Find import relationships
6. **General exploration**: Find all related entities (no relationship filter)

**Example queries:**
1. Find all functions called by search_code:
   {"entity": "search_code", "relationship": "calls"}

2. Find what calls search_code (impact analysis):
   {"entity": "search_code", "relationship": "called_by"}

3. Find parent classes of VectorDatabase:
   {"entity": "VectorDatabase", "relationship": "inherits"}

4. Find child classes of BaseModel:
   {"entity": "BaseModel", "relationship": "inherited_by"}

5. Find all related entities (no filter):
   {"entity": "SemanticSearchEngine"}

6. Limit results:
   {"entity": "search", "relationship": "calls", "limit": 10}

**Example response:**
{
  "status": "success",
  "query": {
    "entity": "search_code",
    "relationship": "calls"
  },
  "results": [
    {
      "name": "_embed_query",
      "type": "function",
      "direction": "calls",
      "file_path": "src/core/search.py"
    },
    {
      "name": "_filter_results",
      "type": "function",
      "direction": "calls",
      "file_path": "src/core/search.py"
    }
  ],
  "count": 2
}

**Empty results:**
{
  "status": "success",
  "query": {"entity": "unknown_function"},
  "results": [],
  "message": "No related entities found for 'unknown_function'"
}""",
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity name or ID to query (e.g., 'search_code', 'VectorDatabase', 'SemanticSearchEngine')",
                },
                "relationship": {
                    "type": "string",
                    "enum": [
                        "calls",
                        "called_by",
                        "imports",
                        "imported_by",
                        "inherits",
                        "inherited_by",
                        "contains",
                        "contained_by",
                    ],
                    "description": "Filter by relationship type. Omit to return all relationships.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["entity"],
        },
    )


def _get_review_pull_request_schema() -> Tool:
    """Get review_pull_request tool schema."""
    return Tool(
        name="review_pull_request",
        description="""Review a pull request/merge request using full codebase context.

**What it does:**
- Analyzes code changes between two git refs (branches, commits, tags)
- Uses vector search to find related code (callers, similar patterns, dependencies)
- Uses knowledge graph to identify impact and relationships
- Applies customizable review instructions from user config
- Returns structured, actionable comments with severity levels

**Key Features:**
- **Context-Aware**: For each changed file, finds callers, similar code, tests, dependencies
- **Customizable**: Applies user-defined review standards from .mcp-vector-search/review-instructions.yaml
- **Structured Output**: Returns inline comments (file + line) and overall PR comments
- **Severity Levels**: critical, high, medium, low, info with blocking semantics
- **Verdict**: approve, request_changes, or comment based on findings

**Review Focus:**
- Security vulnerabilities (SQL injection, XSS, hardcoded secrets)
- Code quality (error handling, input validation, test coverage)
- Style consistency with similar codebase patterns
- Architecture issues (coupling, separation of concerns)
- Performance problems (N+1 queries, inefficient algorithms)

**Example usage:**
{"base_ref": "main", "head_ref": "HEAD"}  # Review current changes vs main
{"base_ref": "develop", "head_ref": "feature-branch"}  # Review feature branch
{"base_ref": "abc123", "head_ref": "def456"}  # Review specific commits
{"base_ref": "main", "head_ref": "HEAD", "custom_instructions": "..."}  # Custom rules

**Example response:**
{
  "status": "success",
  "summary": "Found 2 security issues and 3 style improvements",
  "verdict": "request_changes",
  "overall_score": 0.65,
  "comments": [
    {
      "file_path": "src/auth.py",
      "line_number": 42,
      "comment": "SQL injection vulnerability: use parameterized queries",
      "severity": "critical",
      "category": "security",
      "suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
      "is_blocking": true
    },
    {
      "file_path": null,
      "line_number": null,
      "comment": "Consider adding integration tests for auth flow",
      "severity": "info",
      "category": "quality",
      "suggestion": null,
      "is_blocking": false
    }
  ],
  "blocking_issues": 2,
  "warnings": 0,
  "suggestions": 3,
  "context_files_used": 15,
  "kg_relationships_used": 8
}""",
        inputSchema={
            "type": "object",
            "properties": {
                "base_ref": {
                    "type": "string",
                    "description": "Base git ref to compare against (e.g., 'main', 'develop', commit hash)",
                    "default": "main",
                },
                "head_ref": {
                    "type": "string",
                    "description": "Head git ref to review (default: 'HEAD')",
                    "default": "HEAD",
                },
                "custom_instructions": {
                    "type": "string",
                    "description": "Override default review instructions with custom text (multi-line string)",
                },
            },
            "required": [],
        },
    )


def _get_story_generate_schema() -> Tool:
    """Get story_generate tool schema."""
    return Tool(
        name="story_generate",
        description="Generate a development narrative from git history enriched with semantic code analysis. Produces a StoryIndex JSON artifact with extraction data, semantic analysis, and optional LLM-generated narrative.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Path to the project root directory. Defaults to current project.",
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'json', 'markdown', 'html', or 'all'",
                    "enum": ["json", "markdown", "html", "all"],
                    "default": "json",
                },
                "max_commits": {
                    "type": "integer",
                    "description": "Maximum number of commits to analyze",
                    "default": 200,
                },
                "max_issues": {
                    "type": "integer",
                    "description": "Maximum number of issues to fetch",
                    "default": 100,
                },
                "use_llm": {
                    "type": "boolean",
                    "description": "Whether to use LLM for narrative generation",
                    "default": True,
                },
                "model": {
                    "type": "string",
                    "description": "LLM model to use (optional, uses default if not specified)",
                },
            },
            "required": [],
        },
    )

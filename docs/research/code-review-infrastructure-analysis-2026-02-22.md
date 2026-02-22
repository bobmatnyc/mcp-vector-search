# Code Review Infrastructure Analysis
**Date:** 2026-02-22
**Project:** /Users/masa/Projects/mcp-vector-search
**Purpose:** Understand existing analysis and chat systems to build AI-powered code review feature

---

## Executive Summary

The codebase has a robust analysis framework and sophisticated chat system that can be leveraged for building a new AI-powered code review feature. Key infrastructure exists for:

1. **Structural code analysis** with multiple collectors and smell detection
2. **LLM-powered chat** with tool calling and iterative refinement (30 iterations)
3. **SARIF 2.1.0 reporting** for IDE/CI integration
4. **Knowledge graph** for relationship exploration
5. **Multi-provider LLM support** (OpenAI, OpenRouter, Bedrock)

---

## 1. Existing Analysis System

### Location: `src/mcp_vector_search/cli/commands/analyze.py`

### Subcommands
```python
# Main command: analyze (runs both complexity + dead-code)
# Lines 31-103: Main callback with quick mode option

# Subcommand 1: complexity (lines 105-385)
mcp-vector-search analyze complexity [OPTIONS]

# Subcommand 2: dead-code (lines 1156-1282)
mcp-vector-search analyze dead-code [OPTIONS]

# Subcommand 3: engineers (lines 1421-1426)
mcp-vector-search analyze engineers [OPTIONS]
```

### Analysis Structure
```python
# Core components
run_analysis() → async function (lines 412-821)
├─ Collectors (complexity, smells, etc.)
├─ Output formats: console, json, sarif, markdown
├─ Git integration (changed-only, baseline comparison)
├─ Baseline management (save, compare, list)
├─ Metrics store (historical tracking)
└─ Quality gates (fail-on-smell with severity threshold)

# Key parameters
project_root: Path
quick_mode: bool  # Cognitive + cyclomatic only
output_format: str  # console | json | sarif | markdown
fail_on_smell: bool  # Exit code 1 for CI/CD
severity_threshold: str  # info | warning | error | none
changed_only: bool  # Git integration
baseline: str | None  # Branch comparison
```

### Analysis Collectors (lines 466-489)
```python
# Quick mode (2 collectors)
- CognitiveComplexityCollector()
- CyclomaticComplexityCollector()

# Full mode (5 collectors)
- CognitiveComplexityCollector()
- CyclomaticComplexityCollector()
- NestingDepthCollector()
- ParameterCountCollector()
- MethodCountCollector()
```

### Analysis Modules (`src/mcp_vector_search/analysis/`)
```
analysis/
├── __init__.py                    # Exports core collectors
├── metrics.py                     # ChunkMetrics, FileMetrics, ProjectMetrics
├── dead_code.py                   # Dead code detection (confidence-based)
├── code_quality.py                # Quality assessment
├── interpretation.py              # Enhanced JSON export with LLM context
├── entry_points.py                # Entry point detection
├── debt.py                        # Technical debt tracking
├── trends.py                      # Trend analysis
├── collectors/
│   ├── base.py                    # Base collector interface
│   ├── complexity.py              # Complexity collectors
│   ├── smells.py                  # SmellDetector, CodeSmell, SmellSeverity
│   ├── coupling.py                # Coupling metrics
│   ├── cohesion.py                # Cohesion metrics
│   └── halstead.py                # Halstead metrics
├── reporters/
│   ├── console.py                 # Rich console output
│   ├── sarif.py                   # SARIF 2.1.0 reporter
│   └── markdown.py                # Markdown reports
├── baseline/
│   ├── manager.py                 # Baseline storage/loading
│   └── comparator.py              # Baseline comparison
├── storage/
│   ├── metrics_store.py           # SQLite metrics database
│   ├── trend_tracker.py           # Historical trend analysis
│   └── schema.py                  # Database schema
└── visualizer/
    ├── d3_data.py                 # D3.js data transformation
    ├── exporter.py                # Visualization export
    ├── html_report.py             # Interactive HTML reports
    └── schemas.py                 # Visualization schemas
```

---

## 2. Code Smell Detection

### Location: `src/mcp_vector_search/analysis/collectors/smells.py`

### SmellSeverity Enum (lines 25-40)
```python
class SmellSeverity(Enum):
    INFO = "info"       # Informational
    WARNING = "warning" # Should be addressed
    ERROR = "error"     # Requires immediate attention
```

### CodeSmell Dataclass (lines 43-70)
```python
@dataclass
class CodeSmell:
    name: str                  # Human-readable name
    description: str           # Detailed description
    severity: SmellSeverity    # Severity level
    location: str              # "file:line" or "file:line-range"
    metric_value: float        # Actual metric value
    threshold: float           # Threshold exceeded
    suggestion: str = ""       # Fix suggestion
```

### SmellDetector Class (lines 73-326)

**Detection Methods:**
```python
detect(metrics: ChunkMetrics, file_path: str, start_line: int) → list[CodeSmell]
    # Lines 110-217: Chunk-level smell detection
    # Detects: Long Method, Deep Nesting, Long Parameter List, Complex Method

detect_god_class(file_metrics: FileMetrics, file_path: str) → list[CodeSmell]
    # Lines 219-266: File-level smell detection
    # Detects: God Class (too many methods + lines)

detect_all(file_metrics: FileMetrics, file_path: str) → list[CodeSmell]
    # Lines 268-300: Detect all smells in file

get_smell_summary(smells: list[CodeSmell]) → dict[str, int]
    # Lines 302-325: Generate statistics summary
```

**Supported Smells:**
1. **Long Method** (lines 131-164): `lines > 50 OR cognitive_complexity > 15`
2. **Deep Nesting** (lines 166-181): `max_nesting_depth > 4`
3. **Long Parameter List** (lines 183-198): `parameter_count > 5`
4. **Complex Method** (lines 200-216): `cyclomatic_complexity > 10`
5. **God Class** (lines 219-266): `method_count > 20 AND lines > 500`

---

## 3. SARIF Support

### Location: `src/mcp_vector_search/analysis/reporters/sarif.py`

### SARIFReporter Class (lines 30-378)

**Key Methods:**
```python
generate_sarif(smells: list[CodeSmell], base_path: Path | None) → dict[str, Any]
    # Lines 57-109: Generate SARIF 2.1.0 compliant document
    # Returns: Full SARIF document with tool metadata, rules, results

write_sarif(smells: list[CodeSmell], output_path: Path, base_path: Path | None, indent: int)
    # Lines 111-147: Write SARIF to file with pretty-printing
```

**SARIF Document Structure:**
```python
{
    "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
    "version": "2.1.0",
    "runs": [{
        "tool": {
            "driver": {
                "name": "MCP Vector Search",
                "version": "1.0.3",
                "informationUri": "https://github.com/bobmatnyc/mcp-vector-search",
                "rules": [...]  # Built from unique smells
            }
        },
        "results": [...],  # Converted from CodeSmell objects
        "invocations": [...]
    }]
}
```

**Helper Methods:**
```python
_severity_to_level(severity: SmellSeverity) → str
    # Lines 148-172: Map to SARIF levels (error, warning, note)

_smell_to_rule_id(smell_name: str) → str
    # Lines 174-194: Convert to kebab-case (e.g., "Long Method" → "long-method")

_build_rules(smells: list[CodeSmell]) → list[dict[str, Any]]
    # Lines 196-241: Generate unique rules from smells

_smell_to_result(smell: CodeSmell, base_path: Path | None) → dict[str, Any]
    # Lines 268-345: Convert CodeSmell to SARIF result with location

_compute_fingerprint(smell: CodeSmell) → str
    # Lines 347-377: Generate SHA-256 hash for deduplication
```

---

## 4. Chat Tool System

### Location: `src/mcp_vector_search/cli/commands/chat.py`

### Tool Registration (lines 709-882)

**Tool Definition Format:**
```python
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "Tool description for LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description",
                    "default": "default_value"
                }
            },
            "required": ["required_params"]
        }
    }
}
```

### Existing Tools (lines 715-881)

1. **search_code** (lines 716-737)
   - Query: string, Limit: int (default: 5, max: 15)
   - Semantic search for relevant code

2. **read_file** (lines 739-754)
   - file_path: string
   - Read full file content

3. **write_markdown** (lines 756-775)
   - filename: string, content: string
   - Write reports to reports/ directory

4. **analyze_code** (lines 777-792)
   - focus: string (summary | complexity | smells | all)
   - Get code quality metrics

5. **web_search** (lines 794-809)
   - query: string
   - Search web for documentation

6. **list_files** (lines 811-826)
   - pattern: string (glob pattern)
   - List files matching pattern

7. **deep_search** (lines 828-853)
   - query: string, file_paths: array, limit: int
   - Search within specific files

8. **query_knowledge_graph** (lines 855-880)
   - entity_name: string, relationship_type: string, max_hops: int
   - Explore code relationships

### Tool Execution (lines 884-971)

**Tool Dispatcher:**
```python
async def _execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    search_engine: Any,
    database: Any,
    project_root: Path,
    config: Any,
    session: EnhancedChatSession | None = None
) → str
```

**Tool Implementations:**
```python
_tool_search_code(query, limit, search_engine, database, ...) → str
    # Lines 973-1053: Execute semantic search, format results
    # Includes KG context for top 3 results

_tool_read_file(file_path, project_root) → str
    # Lines 1055-1086: Read file with security checks, 100KB limit

_tool_write_markdown(filename, content, project_root) → str
    # Lines 1088-1110: Write to reports/ directory

_tool_analyze_code(focus, project_root, config) → str
    # Lines 1112-1191: Run analysis, format based on focus

_tool_web_search(query) → str
    # Lines 1193-1239: DuckDuckGo instant answers API

_tool_list_files(pattern, project_root) → str
    # Lines 1241-1269: Glob pattern matching

_tool_deep_search(query, file_paths, limit, ...) → str
    # Lines 1271-1338: Search within specific files

_tool_query_knowledge_graph(entity_name, relationship_type, max_hops, ...) → str
    # Lines 1340-1437: Query KG for relationships
```

### Tool Calling Loop (lines 1439-1590)

**Query Processing:**
```python
async def _process_query(
    query: str,
    llm_client: LLMClient,
    search_engine: Any,
    database: Any,
    session: EnhancedChatSession,
    project_root: Path,
    config: Any,
    verbose_tools: bool = False
) → None
```

**Iteration Logic:**
```python
max_iterations = 30  # Increased from 15 for deeper exploration (line 1470)

for _iteration in range(max_iterations):
    # 1. Call LLM with tools
    response = await llm_client.chat_with_tools(messages, tools)

    # 2. Check for tool calls
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        # Execute tools and add results to messages
        for tool_call in tool_calls:
            result = await _execute_tool(...)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result
            })
    else:
        # No more tool calls → final response
        # Display response and return
        break

# If max iterations reached: synthesize response from collected data
```

**Synthesis on Max Iterations (lines 1554-1589):**
```python
synthesis_prompt = {
    "role": "user",
    "content": "You've reached the maximum number of tool calls. Based on all the information gathered from the tools above, please provide the best possible answer to the original question. Summarize what you found and note if anything is incomplete."
}
messages.append(synthesis_prompt)

# Final call without tools to force text response
final_response = await llm_client.chat_with_tools(messages, tools=[])
```

### Session Management (lines 66-256)

**EnhancedChatSession Class:**
```python
RECENT_EXCHANGES_TO_KEEP = 5  # Keep last 5 pairs (line 76)

# Core methods
set_task(description: str) → None
update_task_status(status: str) → None
add_message(role: str, content: str) → None
add_tool_message(message: dict[str, Any]) → None
add_search_summary(tool_name: str, query: str, result_count: int) → None
get_messages() → list[dict[str, Any]]
    # Returns: [system, history_summary?, task_context?, search_history?, ...recent_messages]
```

**Compaction Strategy (lines 138-191):**
```python
def _compact_history() → None:
    # Compact on 6th pair (keep 5)
    # 1. Find first user/assistant pair
    # 2. Summarize exchange (150 chars each)
    # 3. Append to history_summary
    # 4. Remove compacted messages from self.messages
```

---

## 5. LLM Client

### Location: `src/mcp_vector_search/core/llm_client.py`

### LLMClient Class (lines 22-1031)

**Supported Providers:**
```python
LLMProvider = Literal["openai", "openrouter", "bedrock"]

# Default models
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openrouter": "anthropic/claude-opus-4.5",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

# Thinking models (--think flag)
THINKING_MODELS = {
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-opus-4.5",
    "bedrock": "anthropic.claude-opus-4-20250514-v1:0"
}
```

**Initialization (lines 58-162):**
```python
def __init__(
    self,
    api_key: str | None = None,           # Deprecated
    model: str | None = None,             # Override default model
    timeout: float = TIMEOUT_SECONDS,     # 30.0 default
    provider: LLMProvider | None = None,  # Explicit or auto-detect
    openai_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    think: bool = False                   # Use advanced model
) → None
```

**Auto-detection Priority:**
1. Explicit `provider` parameter
2. Bedrock (if AWS credentials available)
3. OpenRouter (if OPENROUTER_API_KEY set)
4. OpenAI (if OPENAI_API_KEY set)

### Core Methods

**Tool Calling (lines 937-1030):**
```python
async def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]]
) → dict[str, Any]
    # POST to API endpoint with tool definitions
    # Returns: Response with tool_calls or final message
    # NOTE: Bedrock tool calling not yet implemented
```

**Streaming (lines 704-882):**
```python
async def stream_chat_completion(
    messages: list[dict[str, str]]
) → AsyncIterator[str]
    # Stream response chunks for real-time display
    # Parses SSE format: "data: {json}"
    # Yields: Text content chunks
```

**Standard Chat (lines 338-423):**
```python
async def _chat_completion(
    messages: list[dict[str, str]]
) → dict[str, Any]
    # Standard chat completion (no tools)
    # Routes to _bedrock_chat_completion for Bedrock
    # Returns: OpenAI-format response
```

**Bedrock Support (lines 424-517):**
```python
async def _bedrock_chat_completion(
    messages: list[dict[str, str]]
) → dict[str, Any]
    # Uses boto3 converse API
    # Converts messages to Bedrock format
    # Returns: OpenAI-compatible response
```

### Additional Methods

```python
async def generate_search_queries(natural_language_query: str, limit: int) → list[str]
    # Lines 201-260: Generate targeted search queries from NL

async def analyze_and_rank_results(original_query: str, search_results: dict, top_n: int) → list[dict]
    # Lines 262-336: Analyze search results and select most relevant

async def detect_intent(query: str) → IntentType
    # Lines 646-702: Classify intent as "find", "answer", or "analyze"

async def generate_answer(query: str, context: str, conversation_history: list) → str
    # Lines 884-935: Generate answer using codebase context
```

---

## 6. Knowledge Graph Query

### Location: `src/mcp_vector_search/core/knowledge_graph.py`

### KnowledgeGraph Class (lines 145-200+)

**Initialization:**
```python
def __init__(self, db_path: Path):
    self.db_path = db_path
    self.db = None  # kuzu.Database
    self.conn = None  # kuzu.Connection
    self._initialized = False
    self._kuzu_lock = threading.Lock()  # Thread safety
```

**Core Node Types:**
```python
@dataclass
class CodeEntity:
    id: str                    # Unique identifier (chunk_id)
    name: str                  # Function/class name
    entity_type: str           # file, class, function, module
    file_path: str             # Source file path
    commit_sha: str | None     # Git commit for temporal tracking

@dataclass
class CodeRelationship:
    source_id: str
    target_id: str
    relationship_type: str     # calls, imports, inherits, contains, references, documents, follows, has_tag, demonstrates, links_to
    commit_sha: str | None     # Git commit
    weight: float = 1.0        # Relationship strength
```

**Query Methods (inferred from chat tool):**
```python
# From chat.py _tool_query_knowledge_graph (lines 1340-1437)

async def find_entity_by_name(entity_name: str) → str | None
    # Find entity ID by name

async def get_call_graph(entity_id: str) → list[dict]
    # Get call relationships (calls, called_by)

async def get_inheritance_tree(entity_id: str) → list[dict]
    # Get inheritance relationships (inherits, inherited_by)

async def find_related(entity_id: str, max_hops: int) → list[dict]
    # Get related entities within N hops
```

**Relationship Types:**
- **calls**: Function A calls function B
- **imports**: Module A imports module B
- **inherits**: Class A extends class B
- **contains**: File F contains class/function C
- **references**: Code references another entity
- **documents**: Documentation describes code
- **follows**: Sequential relationship
- **has_tag**: Tagged with topic/category
- **demonstrates**: Example demonstrates concept
- **links_to**: Cross-reference link

---

## 7. Models/Data Structures

### Location: `src/mcp_vector_search/core/models.py`

### SearchResult (lines 1-100+)
```python
class SearchResult(BaseModel):
    file_path: Path              # Absolute path to file
    content: str                 # Chunk content
    location: str                # Human-readable location
    similarity_score: float      # Vector similarity (0-1)
    start_line: int              # Starting line number
    end_line: int                # Ending line number
    function_name: str | None    # Function name if applicable
    class_name: str | None       # Class name if applicable
    chunk_type: str              # function, class, method, etc.
    language: str | None         # Programming language
    context_before: str | None   # Lines before chunk
    context_after: str | None    # Lines after chunk
    # Additional fields for ranking
    bm25_score: float | None     # BM25 keyword score
    rerank_score: float | None   # Cross-encoder rerank score
    mmr_score: float | None      # MMR diversity score
```

### Analysis Output Structures

**FileMetrics (from analysis/metrics.py):**
```python
@dataclass
class FileMetrics:
    file_path: str
    chunks: list[ChunkMetrics]
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    function_count: int
    class_count: int
    method_count: int
    # Computed aggregates
    avg_complexity: float
    max_complexity: float
    total_complexity: float
```

**ChunkMetrics (from analysis/metrics.py):**
```python
@dataclass
class ChunkMetrics:
    cognitive_complexity: int
    cyclomatic_complexity: int
    max_nesting_depth: int
    parameter_count: int
    lines_of_code: int
```

**ProjectMetrics (from analysis/metrics.py):**
```python
@dataclass
class ProjectMetrics:
    project_root: str
    files: dict[str, FileMetrics]
    # Aggregated statistics
    total_files: int
    total_functions: int
    average_complexity: float
    hotspots: list[ComplexityHotspot]
```

---

## 8. How to Wire in New Functionality

### Adding a New Analysis Subcommand

**1. Create new subcommand in analyze.py:**
```python
@analyze_app.command(name="review")
def code_review(
    project_root: Path | None = typer.Option(None, ...),
    review_type: str = typer.Option("security", ...),
    output_format: str = typer.Option("console", ...),
    output_file: Path | None = typer.Option(None, ...),
) -> None:
    """Perform AI-powered code review."""
    asyncio.run(run_code_review(
        project_root=project_root or Path.cwd(),
        review_type=review_type,
        output_format=output_format,
        output_file=output_file,
    ))

async def run_code_review(...) -> None:
    # Implement review logic
    pass
```

**2. Register as tool in chat system (chat.py):**
```python
# Add to _get_tools() around line 715
{
    "type": "function",
    "function": {
        "name": "review_code",
        "description": "Perform AI-powered code review on specific files or patterns",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files to review"
                },
                "review_type": {
                    "type": "string",
                    "description": "Type of review: security, architecture, performance, general",
                    "default": "general"
                }
            },
            "required": ["file_paths"]
        }
    }
}

# Add to _execute_tool() around line 906
elif tool_name == "review_code":
    return await _tool_review_code(
        arguments.get("file_paths", []),
        arguments.get("review_type", "general"),
        search_engine,
        database,
        project_root,
        config,
    )

# Implement tool function
async def _tool_review_code(
    file_paths: list[str],
    review_type: str,
    search_engine: Any,
    database: Any,
    project_root: Path,
    config: Any,
) -> str:
    """Execute review_code tool."""
    # 1. Load files
    # 2. Get vector context via search_engine
    # 3. Use LLMClient to perform review
    # 4. Format results
    return formatted_review_results
```

### Adding a New Reporter

**1. Create reporter in analysis/reporters/:**
```python
# analysis/reporters/review.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class ReviewReporter:
    """Generate code review reports."""

    def generate_report(
        self,
        review_results: list[ReviewFinding],
        base_path: Path | None = None
    ) -> dict[str, Any]:
        """Generate review report."""
        # Format review findings
        pass

    def write_report(
        self,
        review_results: list[ReviewFinding],
        output_path: Path,
        format: str = "markdown"
    ) -> None:
        """Write review report to file."""
        if format == "sarif":
            self._write_sarif(review_results, output_path)
        elif format == "markdown":
            self._write_markdown(review_results, output_path)
        else:
            self._write_console(review_results)
```

**2. Create data structures:**
```python
@dataclass
class ReviewFinding:
    """A code review finding."""
    issue_type: str             # security, performance, architecture, style
    severity: str               # critical, high, medium, low, info
    location: str               # file:line
    title: str                  # Short description
    description: str            # Detailed explanation
    recommendation: str         # How to fix
    code_snippet: str | None    # Relevant code
    references: list[str]       # Related findings or documentation
```

### Leveraging Existing Infrastructure

**Use LLMClient for AI analysis:**
```python
from ...core.llm_client import LLMClient

llm_client = LLMClient(
    openrouter_api_key=api_key,
    model="anthropic/claude-opus-4.5",
    timeout=120.0
)

review_prompt = f"""Review this code for {review_type} issues:

{code_content}

Provide specific findings with line numbers, severity, and recommendations."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": review_prompt}
]

response = await llm_client._chat_completion(messages)
review_results = parse_review_response(response)
```

**Use SemanticSearchEngine for context:**
```python
from ...core.search import SemanticSearchEngine

# Find related code
related_results = await search_engine.search(
    query="authentication patterns",
    limit=10,
    include_context=True
)

# Get call graph from KG
if search_engine._kg:
    entity_id = await search_engine._kg.find_entity_by_name("process_login")
    callers = await search_engine._kg.get_call_graph(entity_id)
```

**Reuse SARIF reporter:**
```python
from ...analysis.reporters.sarif import SARIFReporter
from ...analysis.collectors.smells import CodeSmell, SmellSeverity

# Convert review findings to CodeSmell format
smells = [
    CodeSmell(
        name=finding.issue_type,
        description=finding.description,
        severity=SmellSeverity.ERROR if finding.severity == "critical" else SmellSeverity.WARNING,
        location=finding.location,
        metric_value=0.0,
        threshold=0.0,
        suggestion=finding.recommendation
    )
    for finding in review_findings
]

# Generate SARIF
reporter = SARIFReporter()
reporter.write_sarif(smells, output_path="review.sarif", base_path=project_root)
```

---

## 9. Implementation Blueprint for Code Review Feature

### Phase 1: Core Review Engine

**1. Create review module:** `src/mcp_vector_search/analysis/review.py`
```python
@dataclass
class ReviewEngine:
    """AI-powered code review engine."""
    llm_client: LLMClient
    search_engine: SemanticSearchEngine
    project_root: Path

    async def review_files(
        self,
        file_paths: list[Path],
        review_type: str = "general"
    ) -> list[ReviewFinding]:
        """Perform review on files."""
        # 1. Load files
        # 2. Get vector context via semantic search
        # 3. Query KG for relationships
        # 4. Generate review with LLM
        # 5. Parse and structure findings
        pass
```

**2. Define review prompt templates:** `src/mcp_vector_search/analysis/review_prompts.py`
```python
SECURITY_REVIEW_PROMPT = """Review this code for security vulnerabilities...
Focus on:
- SQL injection, XSS, CSRF
- Authentication/authorization flaws
- Input validation
- Cryptographic weaknesses
..."""

ARCHITECTURE_REVIEW_PROMPT = """Review code architecture...
Focus on:
- SOLID principles
- Design patterns
- Coupling/cohesion
- Layer separation
..."""

PERFORMANCE_REVIEW_PROMPT = """Review code performance...
Focus on:
- Algorithmic complexity
- Database query optimization
- Caching opportunities
- Memory usage
..."""
```

### Phase 2: CLI Integration

**Add subcommand to analyze.py:**
```python
@analyze_app.command(name="review")
def code_review(
    files: list[str] = typer.Argument(..., help="Files to review"),
    review_type: str = typer.Option("general", "--type", "-t"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("console", "--format", "-f"),
) -> None:
    """AI-powered code review."""
    asyncio.run(run_code_review(
        file_paths=[Path(f) for f in files],
        review_type=review_type,
        output_path=output,
        output_format=format,
    ))
```

### Phase 3: Chat Tool Integration

**Add review tool to _get_tools():**
```python
{
    "type": "function",
    "function": {
        "name": "review_code",
        "description": "Perform AI code review with security, architecture, or performance focus",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "review_type": {"type": "string", "enum": ["security", "architecture", "performance", "general"]}
            },
            "required": ["file_paths"]
        }
    }
}
```

### Phase 4: SARIF Output

**Extend SARIF reporter for review findings:**
```python
# analysis/reporters/review_sarif.py

class ReviewSARIFReporter(SARIFReporter):
    """SARIF reporter for code review findings."""

    def generate_review_sarif(
        self,
        findings: list[ReviewFinding],
        base_path: Path | None = None
    ) -> dict[str, Any]:
        """Generate SARIF with review-specific rules."""
        # Convert ReviewFinding to CodeSmell format
        # Add review-specific rules (CWE references, OWASP categories)
        # Generate SARIF 2.1.0 document
        pass
```

---

## Summary Table: Key APIs and Signatures

| Component | Key Method | Signature | Location |
|-----------|-----------|-----------|----------|
| **Analysis** | `run_analysis()` | `async (project_root, quick_mode, output_format, ...) → None` | analyze.py:412 |
| **SmellDetector** | `detect()` | `(metrics: ChunkMetrics, file_path, start_line) → list[CodeSmell]` | smells.py:110 |
| **SmellDetector** | `detect_all()` | `(file_metrics: FileMetrics, file_path) → list[CodeSmell]` | smells.py:268 |
| **SARIFReporter** | `generate_sarif()` | `(smells: list[CodeSmell], base_path) → dict[str, Any]` | sarif.py:57 |
| **SARIFReporter** | `write_sarif()` | `(smells, output_path, base_path, indent) → None` | sarif.py:111 |
| **LLMClient** | `chat_with_tools()` | `async (messages, tools) → dict[str, Any]` | llm_client.py:937 |
| **LLMClient** | `_chat_completion()` | `async (messages) → dict[str, Any]` | llm_client.py:338 |
| **SemanticSearchEngine** | `search()` | `async (query, limit, filters, ...) → list[SearchResult]` | search.py:128 |
| **KnowledgeGraph** | `find_entity_by_name()` | `async (entity_name) → str \| None` | knowledge_graph.py (inferred) |
| **KnowledgeGraph** | `get_call_graph()` | `async (entity_id) → list[dict]` | knowledge_graph.py (inferred) |
| **Chat Tool Execution** | `_execute_tool()` | `async (tool_name, arguments, ...) → str` | chat.py:884 |
| **Tool Loop** | `_process_query()` | `async (query, llm_client, ...) → None` | chat.py:1439 |

---

## Next Steps

To implement code review feature:

1. ✅ **Understand infrastructure** (this document)
2. ⏭️ **Define ReviewFinding data structure** similar to CodeSmell
3. ⏭️ **Create ReviewEngine class** with vector-enhanced context
4. ⏭️ **Write review prompt templates** for different review types
5. ⏭️ **Add CLI command** `mcp-vector-search analyze review`
6. ⏭️ **Integrate as chat tool** `review_code()`
7. ⏭️ **Extend SARIF reporter** for review findings
8. ⏭️ **Add tests** for review functionality

---

## File References

- `src/mcp_vector_search/cli/commands/analyze.py` - Main analysis CLI
- `src/mcp_vector_search/cli/commands/chat.py` - Chat REPL with tools
- `src/mcp_vector_search/core/llm_client.py` - Multi-provider LLM client
- `src/mcp_vector_search/core/search.py` - Semantic search engine
- `src/mcp_vector_search/core/knowledge_graph.py` - Kuzu knowledge graph
- `src/mcp_vector_search/core/models.py` - SearchResult data structure
- `src/mcp_vector_search/analysis/collectors/smells.py` - Smell detection
- `src/mcp_vector_search/analysis/reporters/sarif.py` - SARIF 2.1.0 reporter
- `src/mcp_vector_search/analysis/metrics.py` - Metrics data structures

---

*Analysis complete. Infrastructure well-suited for code review feature. Proceed with implementation.*

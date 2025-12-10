# LLM Controller Architecture Analysis

**Date**: December 10, 2024
**Scope**: Investigation of existing LLM controller architecture for structural analysis integration
**Related**: [Structural Code Analysis Project](../projects/structural-code-analysis.md), Issue #38

---

## Executive Summary

The mcp-vector-search project has a well-structured LLM orchestration layer with:
1. **Intent-based routing** distinguishing between "find" (search) and "answer" (QA) modes
2. **Tool-based architecture** enabling LLMs to execute searches and read files
3. **Multiple interfaces**: CLI (`chat` command), MCP server, and programmatic API
4. **Extensible prompt system** with inline prompt definitions in `LLMClient`

**Key Finding**: The architecture is ready for a new "analyze" path with minimal changes. The system already supports tool calling, streaming responses, and contextual prompts.

---

## Architecture Overview

### 1. LLM Controller Location

**Primary Controller**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/llm_client.py`

**Key Components**:
```
src/mcp_vector_search/
├── core/
│   └── llm_client.py          # Central LLM orchestration (752 lines)
├── cli/
│   └── commands/
│       └── chat.py            # Chat command with intent routing (1263 lines)
└── mcp/
    └── server.py              # MCP server implementation (600+ lines)
```

### 2. Intent Detection System

**Current Intents**:
```python
IntentType = Literal["find", "answer"]
```

**Location**: `core/llm_client.py:18`

**Detection Method**: LLM-based classification using a dedicated prompt

```python
async def detect_intent(self, query: str) -> IntentType:
    """Detect user intent from query.

    Returns:
        Intent type: "find" or "answer"
    """
```

**Prompt Structure** (lines 488-496):
```
You are a code search intent classifier. Classify the user's query into ONE of these categories:

1. "find" - User wants to locate/search for something in the codebase
   Examples: "where is X", "find the function that", "show me the code for", "locate X"

2. "answer" - User wants an explanation/answer about the codebase
   Examples: "what does this do", "how does X work", "explain the architecture", "why is X used"

Return ONLY the word "find" or "answer" with no other text.
```

### 3. Routing Implementation

**Router Location**: `cli/commands/chat.py:326-434`

**Function**: `run_chat_with_intent()`

**Flow**:
```python
async def run_chat_with_intent(...):
    # 1. Initialize LLM client
    intent_client = LLMClient(...)

    # 2. Detect intent
    intent = await intent_client.detect_intent(query)

    # 3. Route based on intent
    if intent == "find":
        await run_chat_search(...)  # Lines 887-1113
    else:  # "answer"
        await run_chat_answer(...)  # Lines 436-563
```

### 4. Search Path ("find" intent)

**Entry Point**: `cli/commands/chat.py:887-1113`

**Implementation Flow**:
```python
async def run_chat_search(...):
    # Step 1: Generate multiple search queries from natural language
    search_queries = await llm_client.generate_search_queries(query, limit=3)

    # Step 2: Execute each search query
    search_results = {}
    for search_query in search_queries:
        results = await search_engine.search(...)
        search_results[search_query] = results

    # Step 3: Have LLM analyze and rank results
    ranked_results = await llm_client.analyze_and_rank_results(
        original_query=query,
        search_results=search_results,
        top_n=limit,
    )

    # Step 4: Display results with explanations
    await _display_rich_results(ranked_results, query)
```

**Prompts Used**:
- `generate_search_queries()` (lines 156-172): Converts NL to targeted search queries
- `analyze_and_rank_results()` (lines 224-244): Ranks results by relevance

### 5. Answer Path ("answer" intent)

**Entry Point**: `cli/commands/chat.py:436-563`

**Implementation Flow**:
```python
async def run_chat_answer(...):
    # Initialize with "thinking mode" (advanced model)
    llm_client = LLMClient(..., think=True)

    # Initialize chat session with system prompt
    system_prompt = """You are a helpful code assistant..."""
    session = ChatSession(system_prompt)

    # Process initial query
    await _process_answer_query(...)

    # Interactive loop for follow-up questions
    while True:
        user_input = console.input("You: ")
        await _process_answer_query(...)
```

**Tool-Based Architecture** (lines 589-647):
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the codebase for relevant code snippets",
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full content of a specific file",
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the codebase matching a pattern",
            "parameters": {...}
        }
    }
]
```

**Agentic Loop** (lines 792-884):
- LLM decides which tools to call
- Tools execute and return results
- LLM processes results and decides next action
- Loop continues until LLM provides final answer

### 6. MCP Server Integration

**Location**: `mcp/server.py`

**Tool Definitions** (lines 157-294):
```python
def get_tools(self) -> list[Tool]:
    return [
        Tool(name="search_code", ...),
        Tool(name="search_similar", ...),
        Tool(name="search_context", ...),
        Tool(name="get_project_status", ...),
        Tool(name="index_project", ...),
    ]
```

**Dispatcher** (lines 300-333):
```python
async def call_tool(self, request: CallToolRequest) -> CallToolResult:
    if request.params.name == "search_code":
        return await self._search_code(request.params.arguments)
    elif request.params.name == "search_similar":
        return await self._search_similar(request.params.arguments)
    # ... etc
```

**Pattern**: Each tool has:
1. Schema definition in `get_tools()`
2. Handler method (`_search_code()`, `_search_similar()`, etc.)
3. Result formatting as markdown

---

## Prompt Management

### Current Approach: Inline Prompts

All prompts are defined inline as Python f-strings within methods:

**Example 1: Intent Detection** (lines 488-496):
```python
system_prompt = """You are a code search intent classifier..."""
```

**Example 2: Search Query Generation** (lines 156-172):
```python
system_prompt = """You are a code search expert..."""
```

**Example 3: Result Ranking** (lines 224-244):
```python
system_prompt = """You are a code search expert. Your task is to analyze search results..."""
```

**Example 4: Answer Mode System Prompt** (lines 501-508):
```python
system_prompt = """You are a helpful code assistant analyzing a codebase. Answer questions based on provided code context.

Guidelines:
- Be concise but thorough
- Reference specific functions, classes, or files
- Use code examples when helpful
- If context is insufficient, say so
- Use markdown formatting"""
```

**Example 5: Tool-Based Answer Prompt** (lines 650-664):
```python
system_prompt = """You are a helpful code assistant with access to search tools. Use these tools to find and analyze code in the codebase.

Available tools:
- search_code: Search for relevant code using semantic search
- read_file: Read the full content of a specific file
- list_files: List files matching a pattern

Guidelines:
1. Use search_code to find relevant code snippets
2. Use read_file when you need to see the full file context
3. Use list_files to understand the project structure
4. Make multiple searches if needed to gather enough context
5. After gathering sufficient information, provide your analysis

Always base your answers on actual code from the tools. If you can't find relevant code, say so."""
```

### Advantages of Current Approach
- ✅ Prompts colocated with usage
- ✅ Easy to iterate during development
- ✅ No external file management
- ✅ Type-safe with Python f-strings

### Limitations
- ❌ No versioning of prompts
- ❌ Difficult to A/B test variants
- ❌ No centralized prompt registry
- ❌ Hard to analyze prompt performance

---

## Integration Points for "Analyze" Path

### Option 1: Extend Intent Detection

**Modification**: `core/llm_client.py:18`

```python
# BEFORE
IntentType = Literal["find", "answer"]

# AFTER
IntentType = Literal["find", "answer", "analyze"]
```

**Update Intent Detection Prompt** (lines 488-496):
```python
system_prompt = """You are a code search intent classifier. Classify the user's query into ONE of these categories:

1. "find" - User wants to locate/search for something in the codebase
   Examples: "where is X", "find the function that", "show me the code for"

2. "answer" - User wants an explanation/answer about the codebase
   Examples: "what does this do", "how does X work", "explain the architecture"

3. "analyze" - User wants structural analysis or quality metrics
   Examples: "analyze complexity", "show me code smells", "which files are most complex"

Return ONLY the word "find", "answer", or "analyze" with no other text."""
```

**Update Router** (`chat.py:326-434`):
```python
async def run_chat_with_intent(...):
    intent = await intent_client.detect_intent(query)

    if intent == "find":
        await run_chat_search(...)
    elif intent == "answer":
        await run_chat_answer(...)
    else:  # "analyze"
        await run_chat_analyze(...)  # NEW
```

### Option 2: Add New MCP Tool

**Location**: `mcp/server.py`

**Add to `get_tools()` (after line 291)**:
```python
Tool(
    name="analyze_code",
    description="Analyze code structure and quality metrics",
    inputSchema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to file or directory to analyze"
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Metrics to compute: ['complexity', 'nesting', 'coupling', etc.]"
            },
            "format": {
                "type": "string",
                "enum": ["json", "markdown", "sarif"],
                "description": "Output format",
                "default": "markdown"
            }
        },
        "required": ["file_path"]
    }
)
```

**Add Handler** (after line 598):
```python
async def _analyze_code(self, args: dict[str, Any]) -> CallToolResult:
    """Handle analyze_code tool call."""
    file_path = args.get("file_path", "")
    metrics = args.get("metrics", ["complexity", "nesting"])
    output_format = args.get("format", "markdown")

    # Import analysis module (Phase 1)
    from ..analysis.collectors.complexity import CognitiveComplexityCollector
    from ..analysis.reporters.console import ConsoleReporter

    # Run analysis
    results = await self._run_structural_analysis(
        file_path=file_path,
        metrics=metrics
    )

    # Format output
    if output_format == "markdown":
        response_text = self._format_analysis_markdown(results)
    elif output_format == "json":
        response_text = json.dumps(results, indent=2)
    else:  # sarif
        response_text = self._format_analysis_sarif(results)

    return CallToolResult(
        content=[TextContent(type="text", text=response_text)]
    )
```

**Update Dispatcher** (lines 300-333):
```python
async def call_tool(self, request: CallToolRequest) -> CallToolResult:
    # ... existing cases ...
    elif request.params.name == "analyze_code":
        return await self._analyze_code(request.params.arguments)
    # ...
```

### Option 3: New CLI Command (Parallel to Chat)

**Location**: Create `cli/commands/analyze.py`

**Pattern**: Similar to `chat.py` but focused on analysis

```python
@analyze_app.callback(invoke_without_command=True)
def analyze_main(
    ctx: typer.Context,
    path: Path | None = typer.Argument(None, help="File or directory to analyze"),
    quick: bool = typer.Option(False, "--quick", help="Quick analysis (Tier 1 metrics only)"),
    metrics: list[str] = typer.Option(None, "--metric", "-m", help="Specific metrics to compute"),
    format: str = typer.Option("console", "--format", "-f", help="Output format"),
) -> None:
    """🔍 Analyze code structure and quality metrics."""
    # Delegate to LLM-based analysis or direct metric computation
```

---

## Recommended Architecture for "Analyze" Path

### Phase 1 (Current Sprint): Direct Analysis

**NO LLM involvement** - just expose metrics via CLI and MCP

```
User → CLI/MCP → MetricCollectors → Reporters → Output
```

**Rationale**:
- Metrics are deterministic (no need for LLM interpretation yet)
- Faster iteration on metric accuracy
- Lower token costs during development

### Phase 2-4: Hybrid Approach

**Add LLM interpretation layer** for complex queries

```
User → Intent Detection → Route:
  ├─ "analyze complexity of auth module" → LLM + MetricCollectors
  ├─ "find files with high coupling" → LLM + Search + Metrics
  └─ "explain why this function is complex" → LLM + Metrics + Context
```

### Phase 5 (Issue #38): Full LLM Integration

**Location**: `core/llm_client.py`

**Add Methods**:
```python
async def interpret_metrics(
    self,
    metrics: dict[str, Any],
    context: str,
) -> str:
    """Generate human-readable interpretation of analysis results.

    Args:
        metrics: Computed metrics (complexity, coupling, etc.)
        context: Code context for analysis

    Returns:
        Natural language explanation of findings
    """
    system_prompt = """You are a code quality expert. Interpret structural metrics and provide actionable recommendations.

Metrics schema:
- cognitive_complexity: int (threshold: 15)
- cyclomatic_complexity: int (threshold: 10)
- nesting_depth: int (threshold: 4)
- parameter_count: int (threshold: 5)
- method_count: int (threshold: 20)

Guidelines:
- Explain what the metrics mean in plain language
- Identify the top 3 issues to address
- Suggest specific refactoring patterns
- Prioritize by impact (complexity > nesting > parameters)
"""

    user_prompt = f"""Analyze these metrics:

{json.dumps(metrics, indent=2)}

Code context:
{context}

Provide analysis:"""

    response = await self._chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    return response["choices"][0]["message"]["content"]
```

**Usage in Answer Mode** (add to tools list):
```python
{
    "type": "function",
    "function": {
        "name": "analyze_metrics",
        "description": "Compute structural quality metrics for a file or function",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "function_name": {"type": "string"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "enum": ["complexity", "nesting", "coupling", "smells"]
                }
            },
            "required": ["file_path"]
        }
    }
}
```

---

## Code Smell Detection Integration

**Existing Pattern**: Chat Answer mode uses tool calling

**Proposed**: Add `detect_smells` as a tool

```python
async def execute_detect_smells(file_path: str, smell_types: list[str]) -> str:
    """Execute detect_smells tool."""
    from ..analysis.collectors.smells import SmellDetector

    detector = SmellDetector()
    results = await detector.analyze_file(
        file_path=project_root / file_path,
        smell_types=smell_types or ["long_method", "deep_nesting", "god_class"]
    )

    # Format results for LLM consumption
    if not results:
        return f"No code smells found in {file_path}"

    smell_descriptions = []
    for smell in results:
        smell_descriptions.append(
            f"[{smell.severity}] {smell.type} at lines {smell.start_line}-{smell.end_line}\n"
            f"  Issue: {smell.description}\n"
            f"  Suggestion: {smell.fix_suggestion}\n"
        )

    return "\n".join(smell_descriptions)
```

**LLM Usage**:
```
User: "Why is the authentication function hard to understand?"

LLM: *calls detect_smells(file_path="auth.py", function_name="authenticate")*

Tool Result:
[WARNING] long_method at lines 45-120
  Issue: Function has 75 lines (threshold: 50)
  Suggestion: Extract token validation into separate function

[ERROR] deep_nesting at lines 67-89
  Issue: Nesting depth of 6 (threshold: 4)
  Suggestion: Early return pattern to reduce nesting

LLM Response:
The `authenticate()` function is hard to understand for two main reasons:

1. **Length**: At 75 lines, it's doing too much. The token validation logic (lines 67-89) could be extracted into a separate `validate_token()` function.

2. **Deep Nesting**: The 6 levels of nesting around the token validation make the control flow difficult to follow. Using early returns would flatten this...
```

---

## Performance Considerations

### Current System
- **Intent Detection**: Single LLM call (~500 tokens)
- **Search Mode**: 2 LLM calls (query generation + ranking)
- **Answer Mode**: Variable (depends on tool calls, typically 3-5 LLM calls)

### Proposed "Analyze" Path

**Phase 1** (No LLM):
- Zero LLM calls
- Pure metric computation
- Latency: <100ms for single file, <5s for full project

**Phase 5** (With LLM):
- Intent detection: 1 call
- Metric interpretation: 1 call
- Optional follow-up: 1-3 calls
- Total: 2-5 LLM calls

**Optimization**: Cache metric computations
```python
# During indexing
await database.add_metadata({
    "chunk_id": chunk.id,
    "metrics": {
        "cognitive_complexity": 12,
        "cyclomatic_complexity": 8,
        "nesting_depth": 3
    }
})

# During analysis query
results = await search_engine.search_with_filters({
    "max_complexity": 15,
    "max_nesting": 4
})
```

---

## Related Existing Code

### Parser Integration Points

**Location**: `parsers/python.py`, `parsers/javascript.py`, etc.

**Pattern**: All parsers extend `BaseParser` and implement:
```python
async def parse(self, content: str) -> list[CodeChunk]:
    tree = self.parser.parse(bytes(content, "utf8"))
    # AST traversal happens here
    return chunks
```

**Metric Collection Opportunity**: Add metric collectors to this traversal

```python
async def parse(self, content: str) -> list[CodeChunk]:
    tree = self.parser.parse(bytes(content, "utf8"))

    # Existing chunk extraction
    chunks = self._extract_chunks(tree)

    # NEW: Metric collection (Phase 1)
    if self.collect_metrics:  # config flag
        for chunk in chunks:
            chunk.metrics = await self._collect_metrics(chunk)

    return chunks
```

### Database Schema Extension

**Location**: `core/database.py`

**Current Metadata**:
```python
metadata = {
    "file_path": str,
    "language": str,
    "function_name": Optional[str],
    "class_name": Optional[str],
    "start_line": int,
    "end_line": int,
}
```

**Proposed Extension** (Issue #9):
```python
metadata = {
    # ... existing fields ...
    "metrics": {
        "cognitive_complexity": int,
        "cyclomatic_complexity": int,
        "nesting_depth": int,
        "parameter_count": int,
        "method_count": int,  # for class chunks
    },
    "quality_grade": str,  # "A", "B", "C", "D", "F"
    "has_smells": bool,
}
```

---

## Migration Path

### Immediate (Current Implementation)

1. ✅ Intent detection working with 2 intents
2. ✅ Tool-based architecture in place
3. ✅ MCP server exposing search tools
4. ✅ Prompt management via inline strings

### Phase 1 (v0.17.0 - Current Sprint)

**Goal**: Metric computation WITHOUT LLM

**Changes**:
- Create `analysis/` module structure
- Add metric collectors
- Create `analyze --quick` CLI command
- Extend ChromaDB metadata schema
- Add console reporter

**LLM Impact**: ZERO (no LLM integration yet)

### Phase 5 (v0.21.0 - Search Integration)

**Goal**: LLM-powered analysis queries

**Changes**:
- Add `IntentType = "analyze"` to intent detection
- Create `run_chat_analyze()` function
- Add `analyze_metrics` tool for Answer mode
- Add `interpret_metrics()` to LLMClient
- Expose `analyze_code` MCP tool

**LLM Impact**:
- New intent type in classification prompt
- New system prompts for metric interpretation
- Tool calling for quality-aware search

---

## Conclusions

### Architecture Readiness: ✅ EXCELLENT

The existing LLM controller architecture is:
- **Modular**: Clear separation between intent detection, routing, and execution
- **Extensible**: Adding a third intent is a 10-line change
- **Tool-based**: Already supports dynamic tool calling in Answer mode
- **Multi-interface**: CLI, MCP, and programmatic access all supported

### Recommended Approach

**Phase 1**: Build metric infrastructure WITHOUT LLM integration
- Faster iteration on metric accuracy
- No token costs during development
- Simpler testing (deterministic outputs)

**Phase 5**: Add LLM interpretation layer
- Natural language queries for quality analysis
- Tool-based access to metrics in Answer mode
- Quality-aware search ranking

### Key Integration Points

1. **Intent Detection**: Extend `IntentType` to include "analyze"
2. **Routing**: Add `run_chat_analyze()` parallel to search/answer
3. **MCP Tools**: Add `analyze_code` tool to server
4. **Prompts**: Create metric interpretation prompts
5. **Tools**: Expose metric collectors as callable tools

### Next Steps

1. Implement Phase 1 metric collectors (Issues #2-7)
2. Integrate with indexer (Issue #8)
3. Create CLI command (Issue #10)
4. Defer LLM integration to Phase 5 (Issue #38)

---

## Appendix: File Manifest

### Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `core/llm_client.py` | 752 | LLM orchestration and API calls |
| `cli/commands/chat.py` | 1263 | Chat command with intent routing |
| `mcp/server.py` | 600+ | MCP server tool dispatcher |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `detect_intent()` | `llm_client.py:476` | Classify query intent |
| `run_chat_with_intent()` | `chat.py:326` | Route to search/answer mode |
| `run_chat_search()` | `chat.py:887` | Execute search workflow |
| `run_chat_answer()` | `chat.py:436` | Execute QA workflow |
| `call_tool()` | `server.py:300` | MCP tool dispatcher |

### Prompts Identified

| Prompt | Location | Purpose |
|--------|----------|---------|
| Intent classification | `llm_client.py:488` | Detect find vs answer |
| Query generation | `llm_client.py:156` | NL → search queries |
| Result ranking | `llm_client.py:224` | Analyze search results |
| Answer system | `chat.py:501` | Code assistant guidelines |
| Tool usage | `chat.py:650` | Agentic tool calling |

---

**Research Complete**: December 10, 2024
**Findings Valid For**: v1.0.0 (commit 47e0ff3)
**Next Review**: After Phase 1 implementation

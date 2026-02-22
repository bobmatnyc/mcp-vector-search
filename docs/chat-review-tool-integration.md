# Chat System: `review_code` Tool Integration

**Date:** 2026-02-22
**Status:** ✅ Completed

## Summary

Added `review_code` tool to the chat system, enabling users to request AI-powered code reviews directly in conversation.

## What Was Added

### 1. Tool Definition

Added to `_get_tools()` in `chat.py`:

```python
{
    "type": "function",
    "function": {
        "name": "review_code",
        "description": "Run an AI-powered code review. Supports security, architecture, and performance review types. Uses vector search and knowledge graph for context.",
        "parameters": {
            "type": "object",
            "properties": {
                "review_type": {
                    "type": "string",
                    "enum": ["security", "architecture", "performance"],
                    "description": "Type of review to perform"
                },
                "scope": {
                    "type": "string",
                    "description": "Optional path to scope the review (e.g., 'src/auth', 'core/')"
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Maximum code chunks to analyze (default: 20)",
                    "default": 20
                }
            },
            "required": ["review_type"]
        }
    }
}
```

### 2. Tool Handler

Implemented `_tool_review_code()` async function:

- Creates `ReviewEngine` with search engine, knowledge graph (if available), LLM client, and project root
- Validates review type enum (security, architecture, performance)
- Runs review with scope filter and max_chunks limit
- Formats findings with severity emojis (🔴 CRITICAL, 🟠 HIGH, 🟡 MEDIUM, 🔵 LOW, ⚪ INFO)
- Returns structured markdown for chat context:
  - File paths with line numbers
  - CWE IDs (for security findings)
  - Confidence scores
  - Recommendations
  - Related files
  - Summary with chunk/KG stats and duration

### 3. Tool Execution

Updated `_execute_tool()`:

- Added `llm_client` parameter (needed for ReviewEngine)
- Added case for `"review_code"` tool
- Passes LLM client to handler

Updated `_process_query()`:

- Passes `llm_client` to `_execute_tool()`

### 4. System Prompt Updates

- Updated intro message: "Review code for security, architecture, and performance issues"
- Updated tool usage guidelines: "Use review_code for AI-powered security, architecture, or performance reviews"

### 5. Tests

Created `tests/test_chat_review_tool.py`:

- `test_review_code_tool_registered()` - Verifies tool is registered with correct structure
- `test_review_code_tool_in_tool_list()` - Ensures tool appears after `query_knowledge_graph`

## Usage Examples

Users can now say things like:

```
"do a security review of the auth module"
"review architecture of the core package"
"find performance issues in the database code"
"security review of src/api with max 30 chunks"
```

The LLM will invoke the tool, and findings will be formatted in the chat context for follow-up questions.

## Output Format

```markdown
## Security Review Results

Found 3 finding(s):

### 🔴 CRITICAL: SQL Injection in execute_query()
- **File**: `src/core/db.py:42-55`
- **Category**: SQL Injection
- **CWE**: CWE-89
- **Confidence**: 95%
- **Description**: User input is directly interpolated into SQL query string
- **Recommendation**: Use parameterized queries instead of string formatting
- **Related files**: `src/models.py`, `src/api/handler.py`

### 🟠 HIGH: Missing Authentication Check
...

---
**Summary**: Analyzed 20 code chunks, 8 KG relationships. Review took 6.2s.
```

## Integration Points

- **ReviewEngine** (`mcp_vector_search.analysis.review.engine`): Orchestrates review pipeline
- **Vector Search**: Gathers relevant code chunks using targeted queries
- **Knowledge Graph**: Provides relationship context (if available)
- **LLM Client**: Analyzes code and extracts structured findings

## Architecture Decision

**Why in chat vs. dedicated command?**

- Chat system provides iterative refinement: Users can ask follow-up questions about findings
- LLM can correlate findings across multiple reviews
- Natural for exploratory analysis workflows
- Complements existing `analyze review` CLI command for batch/CI use cases

## Files Modified

- `src/mcp_vector_search/cli/commands/chat.py`: Tool definition, handler, execution
- `tests/test_chat_review_tool.py`: Integration tests

## Test Results

- ✅ All existing tests pass (1408 passed, 108 skipped)
- ✅ New tests verify tool registration and structure
- ✅ No regressions in chat system

## Next Steps

1. **User Testing**: Collect feedback on review output format and verbosity
2. **SARIF Export**: Add optional SARIF format export for CI/CD integration (task #20)
3. **Review History**: Consider caching reviews to avoid duplicate analysis
4. **Custom Rules**: Allow users to define project-specific review rules

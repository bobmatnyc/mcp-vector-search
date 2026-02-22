# `analyze review` CLI Command Implementation

## Summary

Added `mcp-vector-search analyze review [type]` CLI command for AI-powered code review using vector search, knowledge graphs, and LLM analysis.

## Implementation

### Location
- **File**: `src/mcp_vector_search/cli/commands/analyze.py`
- **Function**: `review_code()` (lines 1426-1530)
- **Helper Functions**: `run_review_analysis()`, formatting functions for console/JSON/SARIF/markdown output

### Command Structure

```bash
mcp-vector-search analyze review [REVIEW_TYPE] [OPTIONS]
```

**Review Types:**
- `security` - Security vulnerability analysis (SQL injection, XSS, hardcoded secrets)
- `architecture` - Architecture and design issues (God classes, tight coupling, circular dependencies)
- `performance` - Performance problems (N+1 queries, inefficient algorithms, missing caching)

**Options:**
- `--format/-f` - Output format: `console` (default), `json`, `sarif`, `markdown`
- `--output/-o` - Output file path (stdout for console, auto-named for others)
- `--path` - Scope review to specific path/directory
- `--max-chunks` - Maximum code chunks to analyze (default: 30, range: 10-100)
- `--verbose/-v` - Show detailed progress
- `--project-root/-p` - Project root directory (auto-detected if not specified)

### Pipeline

1. **Initialize Components**
   - Load project configuration
   - Initialize search engine with LanceDB
   - Initialize LLM client (OpenRouter/OpenAI/Bedrock)
   - Optionally load knowledge graph (graceful degradation if unavailable)

2. **Vector Search for Relevant Code**
   - Execute targeted search queries per review type (6 queries for security, 5 for architecture/performance)
   - Merge and deduplicate results
   - Limit to `--max-chunks` most relevant chunks

3. **Knowledge Graph Context** (optional)
   - Query KG for entity relationships
   - Gather 1-2 hop connections for context

4. **LLM Analysis**
   - Format code chunks + KG relationships as context
   - Use specialized review prompts (from `src/mcp_vector_search/analysis/review/prompts.py`)
   - Parse JSON findings from LLM response

5. **Format Output**
   - **Console**: Rich formatted table with color-coded severity, file locations, recommendations
   - **JSON**: Structured findings with metadata
   - **SARIF 2.1.0**: GitHub-compatible format for CI/CD integration
   - **Markdown**: Human-readable report with sections per severity

### Output Format Examples

#### Console
```
🔍 Security Review — entire project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found 7 findings (2 critical, 3 high, 2 medium)

🔴 CRITICAL: SQL Injection in execute_query()
   src/core/db.py:42-55 | CWE-89 | Confidence: 95%
   → Use parameterized queries instead of string formatting

🔴 CRITICAL: Hardcoded API Key
   src/config.py:12 | CWE-798 | Confidence: 98%
   → Move to environment variables

🟠 HIGH: Missing Input Validation
   src/api/handler.py:88-95 | CWE-20 | Confidence: 85%
   → Add input sanitization before processing

Summary: Analyzed 30 code chunks, 12 KG relationships
Review completed in 8.3s using openrouter:anthropic/claude-sonnet-4
```

#### SARIF (for GitHub)
```json
{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "MCP Vector Search - Code Review",
        "version": "1.0.0",
        "rules": [...]
      }
    },
    "results": [
      {
        "ruleId": "sql-injection",
        "level": "error",
        "message": {...},
        "locations": [...]
      }
    ]
  }]
}
```

### Integration Points

**Depends on:**
- `ReviewEngine` - Orchestrates review pipeline (vector search → KG → LLM)
- `SemanticSearchEngine` - Vector search for relevant code chunks
- `LLMClient` - LLM API calls (OpenRouter/OpenAI/Bedrock)
- `KnowledgeGraph` - Optional entity relationship context

**Used by:**
- CLI: `mcp-vector-search analyze review [type]`
- Future: Chat REPL tool integration

### Error Handling

- **Project not initialized**: Clear error message with `mcp-vector-search init` suggestion
- **Invalid review type**: Lists valid options (security, architecture, performance)
- **Invalid format**: Lists valid formats (console, json, sarif, markdown)
- **Missing dependencies**: Graceful degradation for optional components (KG)
- **LLM errors**: Logged with context, returns empty findings list

## Usage Examples

### Basic Security Review
```bash
mcp-vector-search analyze review security
```

### Scoped Architecture Review
```bash
mcp-vector-search analyze review architecture --path src/auth
```

### Export to SARIF for CI/CD
```bash
mcp-vector-search analyze review security --format sarif --output findings.sarif
```

### Performance Review with Verbose Output
```bash
mcp-vector-search analyze review performance --max-chunks 50 --verbose
```

## Testing

- **Unit tests**: Existing tests pass (pytest coverage)
- **Command registration**: Verified via `--help` output
- **End-to-end**: Requires LLM API key and indexed project

## Future Enhancements

1. **Task #19**: Integrate review into chat system as tool
2. **Task #20**: Add review-specific SARIF rules and test end-to-end
3. Support for custom review types via configuration
4. Incremental review (only changed files)
5. Review quality scoring and confidence thresholds
6. Integration with GitHub Actions workflow

## Documentation

- **CLI Help**: `mcp-vector-search analyze review --help`
- **Code**: `src/mcp_vector_search/cli/commands/analyze.py` (lines 1426-2007)
- **Review Engine**: `src/mcp_vector_search/analysis/review/engine.py`
- **Prompts**: `src/mcp_vector_search/analysis/review/prompts.py`

## Performance

- **Search**: ~2-5s for 30 chunks (depends on index size)
- **LLM Analysis**: ~5-15s (depends on model and chunk count)
- **Total**: ~8-20s per review (typical)

## Dependencies

- **Required**: LanceDB, LLM API key (OpenRouter/OpenAI/Bedrock)
- **Optional**: Knowledge graph (graceful degradation)

## Known Limitations

1. LLM response parsing assumes JSON format (markdown code blocks supported)
2. SARIF output requires findings (fails gracefully if none found)
3. Knowledge graph integration is optional (not all projects have it)
4. Review quality depends on LLM model capabilities

## Related Files

- `src/mcp_vector_search/cli/commands/analyze.py` - CLI command implementation
- `src/mcp_vector_search/analysis/review/engine.py` - Review orchestration
- `src/mcp_vector_search/analysis/review/models.py` - Data structures
- `src/mcp_vector_search/analysis/review/prompts.py` - LLM prompts
- `src/mcp_vector_search/analysis/reporters/sarif.py` - SARIF formatting patterns

# MCP Vector Search PR/MR Review Skill

## Overview
Comprehensive PR/MR review skill for mcp-vector-search projects that provides:
- Context-aware code review using vector search and knowledge graph
- Specific, actionable feedback with exact file locations
- HOW-TO guidance for PR/MR workflows
- Performance and architecture recommendations
- Test coverage validation

## Usage
```
/mcp-vector-search-pr-review [--baseline=main] [--focus=security|performance|architecture]
```

## HOW-TO: Creating Effective PRs/MRs

### 1. Pre-Review Setup
```bash
# Ensure project is indexed for context
mvs index_project --force-reindex

# Build knowledge graph for dependency analysis
mvs kg_build

# Check current status
mvs kg_stats
```

### 2. PR Preparation Workflow
```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. After making changes, get contextual review
mvs analyze review-pr --baseline main --format github-json

# 3. Run focused analysis on changed files
mvs search_code "your changed functionality" --limit 10

# 4. Check for architectural impacts
mvs kg_query "functions calling YourChangedClass"

# 5. Validate with repository-wide security scan
mvs review_repository --review-type security --changed-only
```

## Review Process

When `/mcp-vector-search-pr-review` is invoked:

1. **Context Analysis** - Use vector search to find related code patterns
2. **Dependency Impact** - Query knowledge graph for affected components
3. **Test Coverage** - Verify test completeness for changes
4. **Performance Review** - Check for potential bottlenecks
5. **Security Scan** - Validate secure coding patterns
6. **Actionable Feedback** - Provide specific suggestions with locations

## Review Template

### Context Discovery
- Use `search_code` to find similar patterns in codebase
- Use `kg_query` to understand dependency relationships
- Use `analyze_file` for complexity and quality metrics

### Feedback Format
Provide specific, actionable suggestions using GitHub-style format:

**Good Examples:**
```suggestion
# In src/analysis/review/engine.py, line 245
def _gather_code_chunks(self, review_type, scope, max_chunks, file_filter=None):
    # Add input validation
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if file_filter and not isinstance(file_filter, list):
        raise TypeError("file_filter must be a list")
```

**Avoid Generic Advice:**
- ❌ "Consider improving error handling"
- ✅ "Add try-catch in engine.py:156 for SearchError when calling search_engine.search()"

### Performance Focus Areas
- Vector database operations efficiency
- Knowledge graph query optimization
- Large file processing patterns
- Memory usage in indexing operations
- Concurrent processing bottlenecks

### Security Focus Areas
- Input validation in MCP tool handlers
- Path traversal prevention in file operations
- LLM response sanitization
- Configuration file security
- Dependency vulnerability checks

### Architecture Focus Areas
- Single responsibility in handler classes
- Proper error propagation patterns
- Clean separation between core/mcp layers
- Extension point design
- Testing strategy completeness

## Integration Points

### With mcp-vector-search Tools
- `review_pull_request` - For diff-based analysis
- `review_repository` - For comprehensive scans
- `search_code` - For pattern discovery
- `kg_query` - For impact analysis
- `analyze_file` - For quality metrics

### With Git Workflow
- Automatic baseline detection (main, develop, master)
- Changed file identification via git diff
- Commit message validation
- Branch naming convention checks

## Configuration

### Custom Review Instructions
Create `.mcp-vector-search/review-instructions.yaml`:
```yaml
language_standards:
  python:
    - Use type hints for all public methods
    - Follow PEP 8 naming conventions
    - Add docstrings for complex functions

scope_standards:
  mcp_handlers:
    - Validate all input parameters
    - Use proper error types (CallToolResult)
    - Include comprehensive logging

  core_modules:
    - Maintain backwards compatibility
    - Add comprehensive unit tests
    - Document performance characteristics

style_preferences:
  - Prefer composition over inheritance
  - Use dependency injection for testing
  - Minimize coupling between modules

custom_review_focus:
  - Vector search performance impacts
  - Knowledge graph consistency
  - MCP tool interface compliance
  - Error handling completeness
```

## Output Format

### Summary
- Overview of changes analyzed
- Context files consulted
- Key recommendations count
- Risk assessment (LOW/MEDIUM/HIGH)

### Detailed Feedback
```json
{
  "overall_assessment": "APPROVE_WITH_SUGGESTIONS",
  "risk_level": "LOW",
  "suggestions": [
    {
      "file": "src/mcp/review_handlers.py",
      "line": 45,
      "type": "performance",
      "priority": "medium",
      "description": "Consider caching search_engine initialization",
      "suggestion": "Store search_engine as instance variable to avoid repeated initialization",
      "code_block": "```python\nif not hasattr(self, '_cached_search_engine'):\n    self._cached_search_engine = create_search_engine()\nreturn self._cached_search_engine\n```"
    }
  ],
  "context_used": {
    "similar_patterns": 3,
    "dependency_analysis": "8 related functions found",
    "test_coverage": "85% of changed lines covered"
  }
}
```

## Best Practices

### For Reviewers
1. Always run context analysis first
2. Focus on specific, actionable feedback
3. Reference similar patterns in codebase
4. Validate suggestions with vector search
5. Consider architectural impact via KG

### For Authors
1. Index project before creating PR
2. Run self-review with vector search context
3. Check for similar implementations
4. Validate test coverage
5. Document performance implications

## Installation

This skill is automatically installed with mcp-vector-search and available in any project with `.mcp-vector-search/` configuration.

```bash
pip install mcp-vector-search[skills]
# Skill available as: /mcp-vector-search-pr-review
```

---
**Version**: 1.0.0
**Compatible**: mcp-vector-search 3.0.14+
**Author**: MCP Vector Search Team
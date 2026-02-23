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

**Key Feature: Branch Modification Selection**
- Only reviews files modified in your branch vs. baseline
- Uses `git diff` to identify exact changes
- Provides GitHub-style ```suggestion blocks for improvements
- Focuses on changed lines, not entire codebase

**Examples:**
```bash
# Review current branch vs main
/mcp-vector-search-pr-review --baseline=main

# Security-focused review of changes only
/mcp-vector-search-pr-review --baseline=develop --focus=security

# Performance review for feature branch
/mcp-vector-search-pr-review --baseline=main --focus=performance
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

1. **Branch Analysis** - Identify all modified files in the branch/PR
2. **Context Discovery** - Use vector search to find similar patterns for changed code
3. **Impact Analysis** - Query knowledge graph for affected downstream components
4. **Modification-Focused Review** - Review ONLY the changed lines and their immediate context
5. **Actionable Suggestions** - Provide GitHub-style code suggestions for improvements

## Review Template: Branch Modification Focus

### Step 1: Changed Files Selection
```bash
# Get modified files only
git diff --name-only origin/main...HEAD
mvs review_pull_request --baseline main --changed-only
```

### Step 2: Context Discovery (Per Changed File)
```bash
# For each modified file, find related patterns
mvs search_code "similar functionality to YourChangedFile"
mvs kg_query "functions called by YourModifiedFunction"
```

### Step 3: In-Place Code Suggestions

**✅ CORRECT: GitHub-Style Suggestions**
```suggestion
def _parse_findings_json(self, llm_response: str) -> list[ReviewFinding]:
    # Add input validation for empty/null responses
    if not llm_response or not llm_response.strip():
        logger.warning("Empty LLM response received")
        return []

    # Clean up common JSON issues before parsing
    json_str = self._clean_json_string(llm_response)
```

```suggestion
async def handle_review_repository(self, args: dict[str, Any]) -> CallToolResult:
    try:
        # Validate review_type parameter early
        review_type_str = args.get("review_type", "security")
        if review_type_str not in ["security", "architecture", "performance"]:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Invalid review_type: {review_type_str}"
                )],
                isError=True,
            )
```

**❌ AVOID: Generic Line References**
- "Add validation in line 42"
- "Consider improving error handling"
- "Optimize this function"

**✅ PREFERRED: Specific Modifications**
- Show exact code changes in ```suggestion blocks
- Focus on the actual changed lines only
- Provide concrete code improvements

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

## Output Format: Branch-Focused Review

### Summary
- **Files Modified**: List of changed files in branch
- **Lines Changed**: Total additions/deletions
- **Context Analysis**: Similar patterns found via vector search
- **Impact Assessment**: Downstream effects via knowledge graph
- **Suggestions**: Count of actionable improvements for modified code only

### Branch Modification Review Format
```json
{
  "branch_summary": {
    "base_branch": "main",
    "modified_files": ["src/mcp/review_handlers.py", "tests/test_handlers.py"],
    "lines_added": 45,
    "lines_removed": 12,
    "overall_assessment": "APPROVE_WITH_SUGGESTIONS"
  },
  "file_suggestions": [
    {
      "file": "src/mcp/review_handlers.py",
      "modified_lines": "45-67",
      "suggestion_type": "performance",
      "priority": "medium",
      "github_suggestion": "```suggestion\n# Cache search_engine to avoid repeated initialization\nif not hasattr(self, '_search_engine_cache'):\n    embedding_function, _ = create_embedding_function(config.embedding_model)\n    self._search_engine_cache = SemanticSearchEngine(\n        database=database,\n        project_root=self.project_root\n    )\nreturn self._search_engine_cache\n```",
      "rationale": "Modified initialization code creates new search_engine on each call"
    }
  ],
  "context_analysis": {
    "similar_patterns_found": 3,
    "related_functions": ["create_database", "initialize_handlers"],
    "test_coverage_impact": "2 new test methods needed for modified functions"
  }
}
```

### Key Principles
- **Only Review Changed Code** - Don't suggest improvements to unmodified files
- **Use Git Context** - Compare against baseline branch (main/develop)
- **GitHub Suggestions** - All code improvements as ```suggestion blocks
- **Specific Over Generic** - "Add null check in line 45" not "improve validation"

## Best Practices: Branch Modification Focus

### For Reviewers
1. **Start with git diff** - Always identify exact changed files/lines first
2. **Use --changed-only flags** - `mvs review_repository --changed-only`
3. **Context per modification** - Run vector search for each changed function/class
4. **GitHub suggestions only** - All feedback as ```suggestion code blocks
5. **Impact analysis** - Use KG to find downstream effects of changes

### Critical Review Commands
```bash
# Get the exact scope of changes
git diff --name-only origin/main...HEAD
git diff --stat origin/main...HEAD

# Review only the modified files
mvs review_pull_request --baseline main --format github-json
mvs review_repository --review-type security --changed-only

# Find context for each changed component
mvs search_code "similar to YourModifiedClass" --limit 5
mvs kg_query "functions calling YourChangedMethod"
```

### For Authors (Pre-PR Checklist)
1. **Self-review changed files** - `mvs review_pull_request --baseline main`
2. **Check similar patterns** - `mvs search_code "YourNewPattern"`
3. **Validate dependencies** - `mvs kg_query "impact of YourChanges"`
4. **Test coverage** - Ensure tests cover all modified functionality
5. **Performance check** - Use `mvs analyze_file` on heavily modified files

### Anti-Patterns to Avoid
❌ **Don't review entire files** - Only focus on changed lines
❌ **Don't give generic advice** - "improve error handling"
❌ **Don't reference line numbers** - Use ```suggestion blocks instead
❌ **Don't suggest unrelated improvements** - Only review what changed

✅ **Do focus on branch modifications**
✅ **Do use GitHub-style suggestions**
✅ **Do provide specific code improvements**
✅ **Do consider impact of changes via KG**

## Installation

This skill is automatically installed with mcp-vector-search and available in any project with `.mcp-vector-search/` configuration.

```bash
pip install mcp-vector-search[skills]
# Skill available as: /mcp-vector-search-pr-review
```

---
**Version**: 1.1.0
**Compatible**: mcp-vector-search 3.0.15+
**Author**: MCP Vector Search Team
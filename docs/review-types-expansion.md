# Code Review System Expansion

## Summary

Added three new review types to the AI-powered code review system, expanding from 3 to 6 review types.

## New Review Types

### 1. Quality Review (`quality`)

**Focus**: Code quality and maintainability

**Key Areas**:
- Code duplication / DRY violations
- Function/method length and complexity (> 50 lines, > 10 cyclomatic)
- Naming conventions (unclear variable names, misleading names)
- Magic numbers and hardcoded values
- Dead code (unreachable, unused variables)
- Overly complex boolean logic
- Missing error handling

**Search Queries**:
- `"duplicate code repeated logic"`
- `"long method function lines"`
- `"magic number hardcoded value"`
- `"dead code unreachable"`
- `"complex boolean condition"`
- `"error handling exception"`

### 2. Testing Review (`testing`)

**Focus**: Test coverage and quality

**Key Areas**:
- Missing tests for public functions/methods
- Edge case coverage (null/empty inputs, boundary values)
- Test isolation issues (tests depending on each other)
- Missing error path tests
- Brittle tests (hardcoded values, no mocks)
- Missing integration tests for critical paths
- Test function naming clarity

**Search Queries**:
- `"test function method coverage"`
- `"edge case boundary null empty"`
- `"mock fixture setup teardown"`
- `"assert expect verify test"`
- `"integration test e2e"`
- `"pytest unittest spec test"`

### 3. Documentation Review (`documentation`)

**Focus**: Documentation quality

**Key Areas**:
- Missing docstrings on public functions/classes
- Outdated or misleading comments
- Missing parameter/return type documentation
- No module-level documentation
- TODO/FIXME comments that should be resolved
- Missing examples in docstrings for complex APIs

**Search Queries**:
- `"docstring documentation comment"`
- `"TODO FIXME HACK"`
- `"parameter return type annotation"`
- `"example usage demo"`
- `"module package description"`
- `"api endpoint documentation"`

## Files Modified

### 1. `src/mcp_vector_search/analysis/review/models.py`
- Added `QUALITY`, `TESTING`, `DOCUMENTATION` to `ReviewType` enum

### 2. `src/mcp_vector_search/analysis/review/prompts.py`
- Added `QUALITY_REVIEW_PROMPT` (3228 chars)
- Added `TESTING_REVIEW_PROMPT` (3217 chars)
- Added `DOCUMENTATION_REVIEW_PROMPT` (3131 chars)
- Updated `REVIEW_PROMPTS` dict to include new types

### 3. `src/mcp_vector_search/analysis/review/engine.py`
- Added search queries for `QUALITY`, `TESTING`, `DOCUMENTATION` to `REVIEW_SEARCH_QUERIES` dict

### 4. `src/mcp_vector_search/cli/commands/analyze.py`
- Updated `review_code()` function help text to list all 6 review types

## Usage Examples

### Quality Review
```bash
# Review code quality across entire project
uv run mcp-vector-search analyze review quality

# Review quality in specific module
uv run mcp-vector-search analyze review quality --path src/core

# Export quality findings to JSON
uv run mcp-vector-search analyze review quality --format json --output quality-report.json
```

### Testing Review
```bash
# Review test coverage
uv run mcp-vector-search analyze review testing

# Review testing with changed files only
uv run mcp-vector-search analyze review testing --changed-only

# Multiple review types at once
uv run mcp-vector-search analyze review --types testing,quality
```

### Documentation Review
```bash
# Review documentation quality
uv run mcp-vector-search analyze review documentation

# Review docs in API module
uv run mcp-vector-search analyze review documentation --path src/api

# Generate markdown report
uv run mcp-vector-search analyze review documentation --format markdown --output docs-review.md
```

## Integration Notes

### Quality Review Integration
- The quality review can leverage existing `SmellDetector` data if available
- Cyclomatic complexity analysis from code smells feed into quality findings
- Dead code detection complements existing static analysis tools

### Testing Review Integration
- Searches for existing test files to understand coverage gaps
- Can identify missing test files for production code
- Suggests specific test cases based on function complexity

### Documentation Review Integration
- Checks for docstring patterns (Google/NumPy/Sphinx style)
- Identifies TODO/FIXME comments that may indicate technical debt
- Validates parameter/return type documentation against type hints

## Testing

All tests pass:
```bash
uv run pytest tests/ -x -q
# 1472 passed, 147 skipped in 35.83s
```

Custom verification script:
```bash
uv run python scripts/test_new_review_types.py
# ✅ All tests passed!
```

## LOC Delta

**Summary**:
- **Added**: 451 lines (3 new prompts + search queries + enum values)
- **Modified**: 2 lines (CLI help text)
- **Removed**: 0 lines
- **Net Change**: +451 lines

**Breakdown by file**:
- `models.py`: +3 lines (enum values)
- `prompts.py`: +429 lines (3 prompts @ ~143 lines each + dict entries)
- `engine.py`: +18 lines (search query arrays)
- `analyze.py`: +1 line (help text update)

**Phase**: Enhancement (expanding existing feature with new capabilities)

## Next Steps

1. **Add Examples to Docs**: Update main README with quality/testing/documentation review examples
2. **Prompt Tuning**: Based on real-world usage, refine prompts for accuracy
3. **Integration Testing**: Run reviews on large codebases to validate effectiveness
4. **Benchmark Performance**: Measure review speed and LLM token usage for new types
5. **Add Review Presets**: Create preset combinations (e.g., `--types quality,testing` for TDD workflow)

## Related Issues

This enhancement enables:
- Pre-commit quality checks (quality + testing reviews)
- Documentation audits before releases (documentation review)
- Comprehensive code health reports (all 6 review types)
- Team-specific review workflows (combine multiple types)

# Auto-Discovery Implementation Summary

## Overview

Updated `src/mcp_vector_search/analysis/review/instructions.py` to automatically discover repository standards files instead of requiring manual configuration.

## Changes Made

### 1. Added Auto-Discovery Feature

**New Method: `discover_and_load()`**
- Scans repository for standards files in priority order
- Merges standards from multiple sources
- Returns structured `ReviewInstructions` dataclass

**Discovery Priority (highest to lowest):**
1. `.mcp-vector-search/review-instructions.yaml` — explicit override
2. `.github/PULL_REQUEST_TEMPLATE.md` — PR requirements
3. `CONTRIBUTING.md` or `docs/CONTRIBUTING.md` — code standards
4. `DEVELOPMENT.md` or `docs/DEVELOPMENT.md` — dev workflow
5. `STYLE_GUIDE.md` or `docs/style*.md` — explicit style rules
6. `pyproject.toml` — extract tool config (ruff, mypy, black)
7. `.editorconfig` — formatting rules (indent, line length, etc.)

### 2. New Data Structures

**`ReviewInstructions` Dataclass:**
```python
@dataclass
class ReviewInstructions:
    language_standards: list[str]
    scope_standards: list[str]
    style_preferences: list[str]
    custom_review_focus: list[str]
    sources_found: list[str]  # Which files were used
    has_custom_config: bool   # Whether explicit config exists

    def to_prompt_text(self) -> str:
        """Format for LLM prompt with source attribution."""
```

### 3. Extraction Methods

**`_extract_from_markdown(content, source)`**
- Extracts bullet points and numbered lists
- Looks for sections with keywords: "code style", "standards", "requirements", etc.
- Formats as: `"{Section}: {rule text}"`

**`_extract_from_pyproject()`**
- Reads `pyproject.toml` with `tomllib`
- Extracts from `[tool.ruff]`: line-length, target-version, lint rules
- Extracts from `[tool.mypy]`: type checking requirements
- Extracts from `[tool.black]`: line-length
- Formats rules in readable form

**`_extract_from_editorconfig()`**
- Parses `.editorconfig` INI format
- Extracts per-file-type rules: indent_style, indent_size, max_line_length
- Prioritizes Python-specific rules
- Skips generic "all files" rules to avoid duplication

### 4. Backward Compatibility

**Updated `format_for_prompt()` method:**
- Now internally calls `discover_and_load()`
- Maintains same output format
- Existing code continues to work without changes

**Example:**
```python
# Old API (still works)
loader = InstructionsLoader(project_root)
formatted = loader.format_for_prompt()

# New API (recommended)
loader = InstructionsLoader(project_root)
instructions = loader.discover_and_load()
print(f"Sources: {instructions.sources_found}")
```

### 5. Deduplication

Added deduplication logic to remove exact duplicate rules within each category while preserving order.

## Example Output for This Project

**Sources Found:**
- `pyproject.toml`
- `.editorconfig`

**Extracted Standards (19 total):**

From `pyproject.toml` (12 rules):
- Maximum line length: 88
- Target Python version: py311
- Enable pycodestyle errors checks
- Enable pycodestyle warnings checks
- Enable pyflakes checks
- Enable import sorting (isort) checks
- Enable flake8-bugbear checks
- Enable flake8-comprehensions checks
- Enable pyupgrade (modern Python idioms) checks
- Enable pep8-naming conventions checks
- Type check all function definitions
- Use strict equality checks

From `.editorconfig` (7 rules):
- Python files: Use space for indentation
- Python files: Indent size 4
- Python files: Maximum line length 88
- YAML files: Use space for indentation
- YAML files: Indent size 2
- Markdown files: Use space for indentation
- Markdown files: Indent size 2

## Testing

**Verification:**
```bash
# Run all tests
uv run pytest tests/ -x -q

# Test auto-discovery manually
uv run python -c "
from pathlib import Path
from mcp_vector_search.analysis.review.instructions import InstructionsLoader

loader = InstructionsLoader(Path('.'))
instructions = loader.discover_and_load()
print('Sources:', instructions.sources_found)
print('Total rules:', sum([
    len(instructions.language_standards),
    len(instructions.scope_standards),
    len(instructions.style_preferences),
    len(instructions.custom_review_focus)
]))
"
```

**Results:**
- ✅ All 1407 tests pass
- ✅ Backward compatibility maintained
- ✅ Auto-discovery works for this project's files
- ✅ Deduplication works correctly

## Benefits

1. **Zero Configuration**: Works out of the box with existing repo files
2. **Consistency**: Uses standards already defined in your repository
3. **Maintainability**: Update standards in one place (pyproject.toml, .editorconfig)
4. **Flexibility**: Override with explicit config when needed
5. **Transparency**: Shows which files were used as sources
6. **Backward Compatible**: Existing code continues to work

## Documentation Added

1. `docs/auto-discovered-standards-example.md` - Usage examples and extracted standards
2. `docs/auto-discovery-implementation-summary.md` - This document

## Future Enhancements (Optional)

1. Add caching to avoid re-parsing files on every call
2. Support more file types (`.pre-commit-config.yaml`, `tox.ini`, etc.)
3. Add weights to prioritize certain sources over others
4. Support regex-based extraction for more flexible pattern matching
5. Add CLI command to preview auto-discovered standards

## Migration Guide

**No migration needed!** The feature is backward compatible.

**Optional: Migrate to new API for better visibility:**

```python
# Before (still works)
loader = InstructionsLoader(project_root)
loader.load()
formatted = loader.format_for_prompt()

# After (recommended for new code)
loader = InstructionsLoader(project_root)
instructions = loader.discover_and_load()
print(f"Using standards from: {', '.join(instructions.sources_found)}")
formatted = instructions.to_prompt_text()
```

## LOC Delta

```
LOC Delta:
- Added: ~380 lines (new methods, dataclass, extraction logic)
- Removed: 0 lines (backward compatible)
- Net Change: +380 lines
- Purpose: Auto-discovery feature with zero configuration required
```

## Related Files

- `src/mcp_vector_search/analysis/review/instructions.py` - Main implementation
- `src/mcp_vector_search/analysis/review/pr_engine.py` - Consumer of instructions
- `docs/auto-discovered-standards-example.md` - Usage documentation

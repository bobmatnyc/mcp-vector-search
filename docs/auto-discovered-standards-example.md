# Auto-Discovered Review Standards Example

This document shows an example of the review standards that were automatically discovered from this project's repository files.

## Discovery Process

The `InstructionsLoader` automatically scans the following files in priority order:

1. `.mcp-vector-search/review-instructions.yaml` (explicit config - highest priority)
2. `.github/PULL_REQUEST_TEMPLATE.md`
3. `CONTRIBUTING.md` or `docs/CONTRIBUTING.md`
4. `DEVELOPMENT.md` or `docs/DEVELOPMENT.md`
5. `STYLE_GUIDE.md` or `docs/style*.md`
6. `pyproject.toml` (tool config)
7. `.editorconfig`

## Example: This Project's Auto-Discovered Standards

### Sources Found
- `pyproject.toml`
- `.editorconfig`

### Extracted Standards

#### From pyproject.toml
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

#### From .editorconfig
- Python files: Use space for indentation
- Python files: Indent size 4
- Python files: Maximum line length 88
- YAML files: Use space for indentation
- YAML files: Indent size 2
- Markdown files: Use space for indentation
- Markdown files: Indent size 2

## Usage

These standards are automatically loaded when running PR/MR reviews:

```python
from pathlib import Path
from mcp_vector_search.analysis.review.instructions import InstructionsLoader

# Auto-discover and load standards
loader = InstructionsLoader(Path("/path/to/project"))
instructions = loader.discover_and_load()

# View what was found
print(f"Sources: {instructions.sources_found}")
print(f"Has custom config: {instructions.has_custom_config}")

# Format for LLM prompt
prompt_text = instructions.to_prompt_text()
```

## Backward Compatibility

The existing API continues to work:

```python
loader = InstructionsLoader(project_root)
formatted = loader.format_for_prompt()
```

This now internally uses the auto-discovery system but maintains the same output format.

## Creating Custom Config

If you want to override the auto-discovered standards, create:

```
.mcp-vector-search/review-instructions.yaml
```

This file takes highest priority and will override all auto-discovered standards.

Example content:

```yaml
language_standards:
  - "Use snake_case for all Python variables and functions"
  - "All public methods must have docstrings with Args/Returns sections"
  - "Maximum function length: 50 lines"

scope_standards:
  - "All database queries must use parameterized statements"
  - "No hardcoded credentials or API keys"

style_preferences:
  - "Prefer composition over inheritance"
  - "Use type hints for all function signatures"

custom_review_focus:
  - "Check for proper async/await usage in concurrent code"
  - "Ensure test coverage for all new functionality"
```

## Benefits

1. **Zero Configuration**: Works out of the box with existing repo files
2. **Consistency**: Uses the same standards already defined in your repo
3. **Maintainability**: Update standards in one place (pyproject.toml, .editorconfig)
4. **Flexibility**: Override with explicit config when needed
5. **Transparency**: Shows which files were used as sources

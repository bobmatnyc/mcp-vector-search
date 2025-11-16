# Code Quality & Linting

## 🎯 Code Quality Standards

MCP Vector Search maintains high code quality through automated linting, formatting, and type checking. This document outlines the tools and standards used.

---

## 🛠️ Tools & Configuration

### Pre-commit Hooks
All code quality checks run automatically on commit via pre-commit hooks.

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files

# Update hook versions
uv run pre-commit autoupdate
```

### Configuration Files
- **`.pre-commit-config.yaml`** - Pre-commit hook configuration
- **`pyproject.toml`** - Tool configurations (black, ruff, mypy)

---

## 🔧 Linting Tools

### 1. Ruff - Fast Python Linter
**Purpose**: Code style, error detection, import sorting

```bash
# Run ruff linting
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Check specific rules
uv run ruff check src/ --select E,W,F
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests

[tool.ruff.isort]
known-first-party = ["mcp_vector_search"]
```

### 2. Black - Code Formatter
**Purpose**: Consistent code formatting

```bash
# Format code
uv run black src/ tests/

# Check formatting without changes
uv run black src/ tests/ --check

# Show diff
uv run black src/ tests/ --diff
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### 3. MyPy - Type Checking
**Purpose**: Static type analysis

```bash
# Run type checking
uv run mypy src/

# Check specific file
uv run mypy src/mcp_vector_search/core/indexer.py

# Generate type coverage report
uv run mypy src/ --html-report mypy-report/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "chromadb.*",
    "sentence_transformers.*",
    "tree_sitter.*",
]
ignore_missing_imports = true
```

---

## 📋 Code Style Guidelines

### 1. Python Style (PEP 8 + Black)
```python
# Good: Clear, typed function
async def index_file(
    file_path: Path,
    parser: BaseParser,
    database: VectorDatabase,
) -> List[CodeChunk]:
    """Index a single file and return generated chunks."""
    content = await read_file_async(file_path)
    chunks = parser.parse(content)
    await database.store_chunks(chunks)
    return chunks

# Bad: Unclear, untyped function
def index_file(file_path, parser, database):
    content = open(file_path).read()
    chunks = parser.parse(content)
    database.store_chunks(chunks)
    return chunks
```

### 2. Import Organization
```python
# Standard library imports
import asyncio
from pathlib import Path
from typing import List, Optional

# Third-party imports
import typer
from rich.console import Console

# Local imports
from mcp_vector_search.core.models import CodeChunk
from mcp_vector_search.parsers.base import BaseParser
```

### 3. Docstring Style (Google Format)
```python
def search_similar(
    query: str,
    limit: int = 10,
    threshold: float = 0.7,
) -> List[SearchResult]:
    """Search for code similar to the given query.
    
    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        threshold: Minimum similarity score (0.0 to 1.0).
    
    Returns:
        List of search results sorted by similarity score.
    
    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.
        DatabaseError: If vector database is unavailable.
    
    Example:
        >>> results = search_similar("authentication logic", limit=5)
        >>> print(f"Found {len(results)} results")
    """
```

### 4. Error Handling
```python
# Good: Specific exceptions with context
try:
    chunks = await parser.parse(content)
except ParseError as e:
    logger.error(f"Failed to parse {file_path}: {e}")
    raise IndexingError(f"Cannot index {file_path}") from e

# Bad: Bare except
try:
    chunks = parser.parse(content)
except:
    pass
```

---

## 🧪 Quality Metrics

### Code Coverage
```bash
# Run tests with coverage
uv run pytest --cov=src/mcp_vector_search --cov-report=html

# View coverage report
open htmlcov/index.html

# Coverage requirements
# - Minimum: 80% overall coverage
# - Core modules: 90% coverage
# - CLI commands: 70% coverage (UI testing complexity)
```

### Type Coverage
```bash
# Generate type coverage report
uv run mypy src/ --html-report mypy-report/

# Requirements:
# - 95%+ type coverage for core modules
# - 90%+ type coverage for CLI modules
# - 100% type coverage for public APIs
```

### Complexity Metrics
```bash
# Check cyclomatic complexity with radon
pip install radon
radon cc src/ -a

# Guidelines:
# - Functions: Complexity < 10
# - Classes: Complexity < 15
# - Modules: Average complexity < 5
```

---

## 🔍 IDE Integration

### VS Code Settings
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.typeChecking": "strict",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm Settings
1. **File → Settings → Tools → External Tools**
2. Add tools for ruff, black, mypy
3. **Code → Reformat Code** (Ctrl+Alt+L)
4. **Code → Optimize Imports** (Ctrl+Alt+O)

---

## 🚨 Common Issues & Fixes

### Import Sorting Issues
```bash
# Fix import order
uv run ruff check src/ --select I --fix

# Or use isort directly
uv run isort src/ tests/
```

### Type Checking Errors
```python
# Common fixes:

# 1. Add type annotations
def process_file(path):  # Bad
def process_file(path: Path) -> None:  # Good

# 2. Handle Optional types
def get_parser(lang: str) -> BaseParser:  # Bad - might return None
def get_parser(lang: str) -> Optional[BaseParser]:  # Good

# 3. Use type guards
if parser is not None:
    result = parser.parse(content)  # MyPy knows parser is not None
```

### Line Length Issues
```python
# Bad: Long line
result = some_very_long_function_name(argument1, argument2, argument3, argument4, argument5)

# Good: Multi-line
result = some_very_long_function_name(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5,
)
```

---

## 📊 Quality Gates

### Pre-commit Requirements
All commits must pass:
- ✅ Ruff linting (no errors)
- ✅ Black formatting
- ✅ MyPy type checking
- ✅ Import sorting

### CI/CD Requirements
All PRs must pass:
- ✅ All pre-commit checks
- ✅ Test suite (pytest)
- ✅ 80%+ code coverage
- ✅ Documentation builds
- ✅ No security vulnerabilities

### Release Requirements
All releases must have:
- ✅ 90%+ type coverage
- ✅ 85%+ code coverage
- ✅ All quality checks passing
- ✅ Documentation updated
- ✅ Changelog updated

---

## 🔧 Development Workflow

### Daily Development
```bash
# 1. Start development
uv sync
uv run pre-commit install

# 2. Make changes
# ... edit code ...

# 3. Check quality before commit
uv run pre-commit run --all-files

# 4. Commit (hooks run automatically)
git add .
git commit -m "feat: add new feature"
```

### Before Pull Request
```bash
# Run full quality check
./scripts/dev-test.sh

# Check coverage
uv run pytest --cov=src/mcp_vector_search --cov-fail-under=80

# Update documentation if needed
# ... update docs ...
```

---

## 📚 Resources

### Documentation
- **[Black Documentation](https://black.readthedocs.io/)**
- **[Ruff Documentation](https://docs.astral.sh/ruff/)**
- **[MyPy Documentation](https://mypy.readthedocs.io/)**
- **[Pre-commit Documentation](https://pre-commit.com/)**

### Style Guides
- **[PEP 8](https://peps.python.org/pep-0008/)** - Python style guide
- **[PEP 257](https://peps.python.org/pep-0257/)** - Docstring conventions
- **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**

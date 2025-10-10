# Project Organization Standard

**MCP Vector Search - File Organization Rules**

> Version: 1.0.0
> Last Updated: 2025-10-09
> Framework: Python CLI with MCP Integration

## Table of Contents
- [Directory Structure](#directory-structure)
- [File Placement Rules](#file-placement-rules)
- [Naming Conventions](#naming-conventions)
- [Framework-Specific Rules](#framework-specific-rules)
- [Migration Guide](#migration-guide)

---

## Directory Structure

### Standard Layout

```
mcp-vector-search/
├── .claude/                    # Claude MPM configuration (git-ignored)
├── .claude-mpm/               # Claude MPM state (git-ignored)
├── .github/                   # GitHub Actions and workflows
├── .venv/                     # Python virtual environment (git-ignored)
├── docs/                      # ALL documentation
│   ├── reference/            # Reference documentation (this file)
│   ├── developer/            # Developer guides and APIs
│   ├── architecture/         # Architecture decisions and diagrams
│   ├── performance/          # Performance analysis and benchmarks
│   ├── analysis/             # Analysis reports
│   ├── debugging/            # Debugging guides
│   ├── prd/                  # Product requirements
│   └── technical/            # Technical specifications
├── examples/                  # Usage examples and sample code
├── scripts/                   # Development and utility scripts
│   ├── debug/                # Debug scripts (debug_*.py)
│   ├── test/                 # Manual test scripts (test_*.py)
│   └── setup/                # Setup and installation scripts
├── src/                       # Source code (Python package)
│   └── mcp_vector_search/    # Main package
│       ├── cli/              # CLI layer (Typer commands)
│       ├── core/             # Core business logic
│       ├── parsers/          # Language parsers
│       ├── mcp/              # MCP server integration
│       └── config/           # Configuration management
├── tests/                     # ALL test files
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
├── tmp/                       # Temporary files (git-ignored)
├── dist/                      # Build artifacts (git-ignored)
├── .gitignore                 # Git ignore rules
├── CLAUDE.md                  # Claude Code instructions (root only)
├── README.md                  # Project overview (root only)
├── LICENSE                    # License file (root only)
├── pyproject.toml            # Python project configuration
├── Makefile                  # Build and development tasks
└── uv.lock                   # Dependency lock file
```

---

## File Placement Rules

### 🔴 Critical Rules (MUST FOLLOW)

#### Root Directory Files (Maximum Clarity)
**Only these files should be in the root directory:**

| File | Purpose | Must Exist |
|------|---------|-----------|
| `README.md` | Project overview, quick start | ✅ Yes |
| `CLAUDE.md` | Claude Code/MPM instructions | ✅ Yes |
| `LICENSE` | License information | ✅ Yes |
| `pyproject.toml` | Python package configuration | ✅ Yes |
| `Makefile` | Build system and tasks | ✅ Yes |
| `uv.lock` | Dependency lock file | ✅ Yes |
| `pytest.ini` | Pytest configuration | Optional |
| `.gitignore` | Git ignore patterns | ✅ Yes |
| `.editorconfig` | Editor configuration | Optional |
| `.pre-commit-config.yaml` | Pre-commit hooks | Optional |
| `.python-version` | Python version pinning | Optional |

**❌ NEVER place these in root:**
- Debug scripts (`debug_*.py`)
- Test files (`test_*.py`)
- Documentation files (except README.md, CLAUDE.md, LICENSE)
- Utility scripts (`.sh`, helper scripts)
- Temporary files
- Build artifacts

#### Documentation Files (`docs/`)
**All documentation must be in `docs/` with proper categorization:**

| Category | Location | File Types | Examples |
|----------|----------|-----------|----------|
| Reference Docs | `docs/reference/` | Standards, guides | `PROJECT_ORGANIZATION.md` |
| Developer Guides | `docs/developer/` | APIs, testing | `API.md`, `TESTING.md` |
| Architecture | `docs/architecture/` | Design docs | `REINDEXING_WORKFLOW.md` |
| Performance | `docs/performance/` | Benchmarks | `CONNECTION_POOLING.md` |
| Analysis | `docs/analysis/` | Reports | `SEARCH_ANALYSIS_REPORT.md` |
| Technical | `docs/technical/` | Specifications | `SIMILARITY_CALCULATION_FIX.md` |
| Product | `docs/prd/` | Requirements | `mcp_vector_search_prd.md` |

**Special Documentation (Root Level Only):**
- `README.md` - Project introduction and quick start
- `CLAUDE.md` - Claude Code/MPM instructions (linked to `docs/reference/`)
- `LICENSE` - License file

**✅ Correct:**
```
docs/developer/CONTRIBUTING.md    # Developer contribution guide
docs/reference/PROJECT_ORGANIZATION.md  # This file
docs/architecture/REINDEXING_WORKFLOW.md  # Architecture doc
```

**❌ Incorrect:**
```
DEVELOPER.md                      # Should be in docs/developer/
INSTALL.md                        # Should be in docs/reference/
MCP_SETUP.md                      # Should be in docs/reference/
```

#### Test Files (`tests/`)
**All test files must be in `tests/` directory:**

| Test Type | Location | Naming Pattern |
|-----------|----------|----------------|
| Unit Tests | `tests/unit/` | `test_*.py` |
| Integration | `tests/integration/` | `test_*.py` |
| End-to-End | `tests/e2e/` | `test_*.py` |
| Fixtures | `tests/conftest.py` | `conftest.py` |

**❌ Never place test files in root directory**
```
# WRONG - test files in root
test_debug.py
test_index.py
test_patch.py

# CORRECT - test files in tests/
tests/unit/test_debug.py
tests/integration/test_index.py
tests/unit/test_patch.py
```

#### Debug & Utility Scripts (`scripts/`)
**All debug and utility scripts must be in `scripts/` with subcategories:**

| Script Type | Location | Examples |
|-------------|----------|----------|
| Debug Scripts | `scripts/debug/` | `debug_parser.py`, `debug_search.py` |
| Test Scripts | `scripts/test/` | `test_both_patches.py` |
| Setup Scripts | `scripts/setup/` | `setup-alias.sh` |
| Utility Scripts | `scripts/utils/` | Helper scripts |

**✅ Correct:**
```
scripts/debug/debug_parser.py
scripts/debug/debug_search_detailed.py
scripts/setup/setup-alias.sh
```

**❌ Incorrect:**
```
debug_parser.py              # Should be in scripts/debug/
debug_search.py              # Should be in scripts/debug/
setup-alias.sh               # Should be in scripts/setup/
```

#### Source Code (`src/`)
**Python package follows standard src-layout:**

```
src/mcp_vector_search/
├── __init__.py           # Package initialization, version
├── cli/                  # CLI layer (user interface)
│   ├── __init__.py
│   ├── main.py          # Typer app entry point
│   ├── commands/        # CLI command implementations
│   ├── output.py        # Rich output formatting
│   └── didyoumean.py    # Command suggestions
├── core/                 # Core business logic
│   ├── __init__.py
│   ├── indexer.py       # Code indexing
│   ├── search.py        # Search algorithms
│   ├── database.py      # Vector database
│   ├── project.py       # Project management
│   └── embeddings.py    # Text embeddings
├── parsers/             # Language parsers
│   ├── __init__.py
│   ├── base.py          # Base parser interface
│   ├── registry.py      # Parser registry
│   ├── python.py        # Python parser
│   └── ...              # Other language parsers
├── mcp/                 # MCP server integration
│   ├── __init__.py
│   └── server.py        # MCP server implementation
└── config/              # Configuration
    ├── __init__.py
    ├── settings.py      # Settings models
    └── defaults.py      # Default values
```

---

## Naming Conventions

### File Naming Standards

#### Python Files
- **Modules**: `snake_case.py` (e.g., `auto_indexer.py`)
- **Tests**: `test_*.py` (e.g., `test_search.py`)
- **Debug Scripts**: `debug_*.py` (e.g., `debug_parser.py`)

#### Documentation Files
- **General Docs**: `SCREAMING_CASE.md` (e.g., `README.md`, `CONTRIBUTING.md`)
- **Technical Docs**: `SCREAMING_CASE.md` (e.g., `API.md`, `TESTING.md`)
- **Descriptive Docs**: `lowercase_with_underscores.md` (e.g., `mcp_integration.md`)

#### Shell Scripts
- **Setup Scripts**: `kebab-case.sh` (e.g., `setup-alias.sh`)
- **Utility Scripts**: `kebab-case.sh` (e.g., `mcp-vector-search.sh`)

#### Directories
- **Package Directories**: `snake_case/` (e.g., `mcp_vector_search/`)
- **Doc Categories**: `lowercase/` (e.g., `developer/`, `reference/`)

### Import Conventions
```python
# Absolute imports from src/
from mcp_vector_search.core.search import VectorSearch
from mcp_vector_search.parsers.python import PythonParser
from mcp_vector_search.cli.output import console

# Never use relative imports across packages
# ❌ from ..core.search import VectorSearch
```

---

## Framework-Specific Rules

### Python CLI Application (Typer + Rich)

#### Entry Points
```python
# pyproject.toml
[project.scripts]
mcp-vector-search = "mcp_vector_search.cli.main:app"
```

#### CLI Command Structure
```
src/mcp_vector_search/cli/
├── main.py              # Typer app, command routing
└── commands/            # Command implementations
    ├── search.py        # Search commands
    ├── index.py         # Index commands
    ├── config.py        # Config commands
    └── mcp.py           # MCP commands
```

### MCP Server Integration

#### MCP Files Organization
```
src/mcp_vector_search/mcp/
├── __init__.py
└── server.py            # MCP server implementation

# MCP tool implementations use core modules
# from mcp_vector_search.core.search import VectorSearch
```

#### Configuration Files
```
.mcp-vector-search/      # Per-project config (git-ignored)
├── config.json          # Project settings
└── chroma_db/          # Vector database
```

### Testing Structure

#### Test Organization
```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Unit tests
│   ├── test_search.py
│   ├── test_indexer.py
│   └── test_parsers/
│       ├── test_python.py
│       ├── test_javascript.py
│       └── ...
├── integration/         # Integration tests
│   ├── test_cli.py
│   └── test_mcp.py
└── e2e/                # End-to-end tests
    └── test_workflow.py
```

---

## Migration Guide

### Moving Files to Correct Locations

#### Step 1: Identify Misplaced Files
```bash
# Find root-level files that should be moved
find . -maxdepth 1 -type f \( -name "*.py" -o -name "*.md" -o -name "*.sh" \)
```

#### Step 2: Create Target Directories
```bash
mkdir -p scripts/debug
mkdir -p scripts/test
mkdir -p scripts/setup
mkdir -p docs/reference
```

#### Step 3: Move Files with Git History Preservation
```bash
# Use git mv to preserve history
git mv debug_parser.py scripts/debug/
git mv test_index.py tests/unit/
git mv DEVELOPER.md docs/developer/
```

#### Step 4: Update Imports and References
```python
# Update any imports in code
# Old: from debug_parser import ...
# New: from scripts.debug.debug_parser import ...
```

#### Step 5: Validate Build
```bash
make test          # Run all tests
make quality       # Check code quality
```

### Common Migration Patterns

#### Debug Scripts (Root → scripts/debug/)
```bash
git mv debug_parser.py scripts/debug/
git mv debug_search.py scripts/debug/
git mv debug_search_v2.py scripts/debug/
git mv debug_search_detailed.py scripts/debug/
git mv debug_cli.py scripts/debug/
```

#### Test Files (Root → tests/unit/)
```bash
git mv test_debug.py tests/unit/
git mv test_index.py tests/unit/
git mv test_patch.py tests/unit/
git mv test_force.py tests/unit/
git mv test_both_patches.py tests/unit/
git mv test_empty_index.py tests/unit/
git mv test_with_files.py tests/unit/
```

#### Documentation (Root → docs/reference/)
```bash
git mv DEVELOPER.md docs/developer/DEVELOPER.md
git mv INSTALL.md docs/reference/INSTALLATION.md
git mv MCP_SETUP.md docs/reference/MCP_SETUP.md
git mv ENGINEER_TASK.md docs/reference/ENGINEER_TASK.md
git mv INSTALL_COMMAND_ENHANCEMENTS.md docs/reference/INSTALL_COMMAND_ENHANCEMENTS.md
```

#### Setup Scripts (Root → scripts/setup/)
```bash
git mv setup-alias.sh scripts/setup/
git mv mcp-vector-search.sh scripts/setup/
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-09 | Initial organization standard created |

---

## References

- [CLAUDE.md](../../CLAUDE.md) - Claude Code instructions
- [STRUCTURE.md](../STRUCTURE.md) - Detailed structure documentation
- [CONTRIBUTING.md](../developer/CONTRIBUTING.md) - Contribution guidelines

---

**🔴 This standard must be followed for all new files and enforced during code reviews.**
**📚 Link this document from CLAUDE.md for Claude Code/MPM awareness.**

# MCP Vector Search - Project Instructions

## Project Overview

**CLI-first semantic code search with MCP (Model Context Protocol) integration**

This project provides intelligent code search capabilities using vector embeddings and semantic similarity, with deep integration into Claude Desktop and other MCP-compatible tools.

## Project Information
- **Path**: /Users/masa/Projects/mcp-vector-search
- **Language**: Python 3.11+
- **Framework**: FastAPI, ChromaDB, Sentence Transformers
- **Package Manager**: uv (for development), pip/PyPI (for distribution)

## Project Structure Philosophy

### Root Directory - STRICT POLICY

**ONLY ESSENTIAL FILES ALLOWED IN PROJECT ROOT:**

✅ **Permitted Files:**
- `README.md` - Primary project documentation
- `CHANGELOG.md` - Version history
- `DEPLOYMENT.md` - Deployment instructions
- `CLAUDE.md` - This file (AI assistant instructions)
- `LICENSE` - Project license
- `pyproject.toml` - Python project configuration
- `.gitignore`, `.editorconfig`, `.pre-commit-config.yaml` - Configuration files
- `Makefile` - Build automation
- Standard Python files: `setup.py`, `setup.cfg`, `requirements*.txt`

❌ **PROHIBITED in Root:**
- Test files (`test_*.py`, `*_test.py`, `debug_*.py`)
- Summary/status documents (`*_SUMMARY.md`, `*_STATUS.md`, `*_RESULTS.md`)
- Temporary files (`*.tmp`, `*.temp`, `*.bak`)
- Data files (`*.json`, `*.csv`, `*.pkl` - except small configs)
- HTML files (`*.html`)
- Log files (`*.log`)
- Build artifacts (`dist/`, `build/`, `*.egg-info/`)

### Proper File Locations

```
mcp-vector-search/
├── src/mcp_vector_search/     # Source code
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── e2e/                   # End-to-end tests
│   └── manual/                # Manual test scripts and debug files
├── docs/
│   ├── summaries/             # Project summaries and status docs
│   ├── research/              # Research notes and test artifacts
│   ├── development/           # Development guides
│   ├── guides/                # User guides
│   └── reference/             # API reference
├── scripts/                   # Build and utility scripts
├── examples/                  # Usage examples
└── .mcp-vector-search/        # Runtime data (gitignored)
    └── visualization/         # Visualization artifacts
```

## File Organization Rules

### When Creating New Files

**Before creating any file in project root, ask:**
1. Is this file essential for all users? → Core docs only
2. Is this a temporary artifact? → Use appropriate subdirectory
3. Is this documentation? → Place in `docs/` subdirectories
4. Is this a test? → Place in `tests/` subdirectories
5. Is this a script? → Place in `scripts/` directory
6. Is this data/output? → Place in appropriate data directory (likely gitignored)

### Enforcement

- Pre-commit hooks will flag violations
- CI/CD should validate root directory cleanliness
- All pull requests must maintain root directory policy

## Development Guidelines

### Code Quality Standards

**All code must pass:**
- ✅ Black (formatting)
- ✅ Ruff (linting)
- ✅ Mypy (type checking)
- ✅ Bandit (security scanning)
- ✅ Pre-commit hooks

### Testing Requirements

- Unit tests for all new functionality
- Integration tests for MCP components
- Manual tests go in `tests/manual/` (not root!)
- Minimum 80% code coverage

### Documentation Requirements

- Docstrings for all public functions/classes
- Type hints for all function signatures
- Update relevant docs in `docs/` when changing features
- Keep CHANGELOG.md updated

## Key Technologies

### Core Stack
- **Python 3.11+**: Modern Python with type hints
- **FastAPI**: High-performance API framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Semantic embeddings
- **MCP**: Model Context Protocol integration

### Development Tools
- **uv**: Fast Python package manager
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **Mypy**: Static type checking
- **Pytest**: Testing framework

## Common Commands

```bash
# Development
uv sync                          # Install dependencies
uv run pytest                    # Run tests
uv run black .                   # Format code
uv run ruff check .              # Lint code
uv run mypy src/                 # Type check

# Usage
mcp-vector-search init           # Initialize project
mcp-vector-search index          # Index codebase
mcp-vector-search search "query" # Search code
mcp-vector-search visualize      # Visualize graph

# Build & Release
make pre-publish                 # Quality gate
make release-build               # Build package
make release-pypi                # Publish to PyPI
```

## Architecture Notes

### Core Components

1. **Indexer** (`src/mcp_vector_search/core/indexer.py`)
   - Parses source code into chunks
   - Generates embeddings
   - Stores in ChromaDB

2. **Search** (`src/mcp_vector_search/core/search.py`)
   - Semantic similarity search
   - Hybrid search (semantic + keyword)
   - Result ranking and filtering

3. **MCP Server** (`src/mcp_vector_search/mcp/server.py`)
   - Exposes tools via MCP protocol
   - Integration with Claude Desktop
   - Resource and prompt management

4. **CLI** (`src/mcp_vector_search/cli/`)
   - Command-line interface
   - Interactive search
   - Visualization tools

### Design Principles

- **Async-first**: Use async/await for I/O operations
- **Type-safe**: Full type hints, strict mypy checking
- **Modular**: Clear separation of concerns
- **Testable**: Dependency injection, mocks where needed
- **Performance**: Connection pooling, caching, lazy loading

## Memory Integration

This project uses KuzuMemory for intelligent context management.

### Available Commands:
- `kuzu-memory enhance <prompt>` - Enhance prompts with project context
- `kuzu-memory learn <content>` - Store learning from conversations
- `kuzu-memory recall <query>` - Query project memories
- `kuzu-memory stats` - View memory statistics

### Memory Guidelines
- Store project decisions and architectural choices
- Record technical specifications and API details
- Capture user preferences and usage patterns
- Document error solutions and workarounds
- Keep memories project-specific and relevant

## Important Notes

### Security
- Never commit API keys or credentials
- Use environment variables for sensitive data
- Run security scans before releases (Bandit)
- Document security considerations with `# nosec` when suppressing warnings

### Performance
- Use connection pooling for database access
- Implement caching for frequently accessed data
- Profile before optimizing
- Monitor memory usage for large codebases

### Compatibility
- Support Python 3.11+ only
- Follow semantic versioning (SemVer)
- Maintain backward compatibility in minor versions
- Document breaking changes in CHANGELOG

## Contributing

Before submitting changes:
1. ✅ Run full test suite: `make test`
2. ✅ Run quality gate: `make pre-publish`
3. ✅ Update documentation
4. ✅ Update CHANGELOG.md
5. ✅ Ensure root directory is clean (no test/temp files)
6. ✅ All pre-commit hooks pass

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Last Updated**: December 4, 2025

*This file provides instructions for AI assistants working on this project. All AI interactions should follow these guidelines to maintain project quality and organization.*

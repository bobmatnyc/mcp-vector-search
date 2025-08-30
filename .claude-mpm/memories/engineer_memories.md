# Agent Memory: engineer
<!-- Last Updated: 2025-08-30T15:33:00.000000Z -->

## MCP Vector Search - Technical Architecture

### Project Overview
- **Type**: CLI-first semantic code search tool with MCP (Model Context Protocol) integration
- **Language**: Python 3.11+ (14K LOC across 49 files)
- **Architecture**: Layered async-first design with extensive type safety
- **Key Features**: Vector embeddings, AST-aware parsing, Claude Desktop integration

### Core Architecture Components
1. **CLI Layer**: Typer-based with Rich output and "did you mean" suggestions
2. **Core Engine**: SemanticIndexer, SemanticSearchEngine, ChromaVectorDatabase
3. **Parser System**: Extensible registry supporting Python, JS, TS with AST parsing
4. **MCP Integration**: Server providing search tools for Claude Desktop
5. **Database Layer**: ChromaDB with connection pooling (13.6% performance boost)

### Key Technical Decisions
- **Connection Pooling**: Implemented for 13.6% performance improvement
- **Async-First**: All I/O operations use async/await patterns
- **AST-Aware Parsing**: Better code chunking than simple text splitting
- **Single-Path Workflows**: Comprehensive Makefile with priority-based commands
- **Rich Error Handling**: Custom exceptions with actionable error messages

### Development Workflow (CRITICAL)
```bash
make dev-setup     # ONE-COMMAND development environment setup
make quality       # ALL quality checks (lint, type, security, test)
make test          # Full test suite with coverage
make lint-fix      # Auto-fix formatting and linting issues
```

### Performance Characteristics
- **Search Latency**: <100ms for typical queries
- **Indexing Speed**: ~1000 files/minute
- **Memory Usage**: 50MB baseline + 1MB per 1000 chunks
- **Storage**: ~1KB per code chunk (compressed embeddings)

### MCP Integration Details
- **Server**: 725 LOC implementing 6 search tools for Claude Desktop
- **Tools**: search_code, search_similar, search_context, index_file, etc.
- **Configuration**: Requires claude_desktop_config.json setup
- **Testing**: `make test-mcp` for integration verification

### Critical Files
- **Version**: `src/mcp_vector_search/__init__.py` - single source of truth
- **Entry**: `src/mcp_vector_search/cli/main.py` - CLI application root
- **Config**: `pyproject.toml` - dependencies and tool configuration
- **Build**: `Makefile` - comprehensive build and release workflows
- **Docs**: `CLAUDE.md` (priority-based), `DEVELOPER.md` (technical)

### Quality & Testing
- **Pre-commit**: Automated quality checks (black, ruff, mypy, bandit)
- **Coverage**: >90% overall, >95% core modules
- **Testing**: Unit, integration, CLI, MCP, and performance tests
- **CI/CD**: Comprehensive Makefile-based workflows

### Known Technical Debt
- Tree-sitter integration needs improvement (using regex fallback)
- Language support limited to Python/JS/TS (Java/Go/Rust planned)
- Binary file support not yet implemented


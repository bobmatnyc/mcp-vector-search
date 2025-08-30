# CLAUDE.md - MCP Vector Search Project Guide

**Primary Instructions for Claude Code & Claude MPM Integration**

> 🎯 **Priority System**: 🔴 Critical | 🟡 High | 🟢 Medium | ⚪ Optional

This file provides comprehensive guidance for Claude Code (claude.ai/code) and Claude MPM when working with the MCP Vector Search codebase.

## 🔴 Project Overview (CRITICAL)

**MCP Vector Search** is a CLI-first semantic code search tool with MCP (Model Context Protocol) integration. It provides intelligent code search using vector embeddings and AST-aware parsing for Python, JavaScript, and TypeScript.

### 🔴 Core Architecture (MUST UNDERSTAND)
- **Vector Database**: ChromaDB with connection pooling (13.6% performance boost)
- **Embedding Model**: Configurable sentence transformers (default: all-MiniLM-L6-v2)
- **Parser System**: Extensible language parser registry with AST and regex fallback
- **CLI Framework**: Typer with Rich for beautiful terminal output
- **MCP Integration**: Server implementation for Claude Desktop integration
- **Async Processing**: Modern async Python with comprehensive type safety

### 🔴 Single-Path Commands (PRIMARY WORKFLOWS)
```bash
# Build & Development
make dev-setup     # One-command development environment setup
make test          # Run all tests with coverage
make lint-fix      # Format and lint code automatically
make build         # Build package for distribution

# Release & Deployment  
make release-patch # Bump patch version, commit, tag, build
make publish       # Publish to PyPI

# Quality Assurance
make quality       # Run all quality checks (lint, type, test, security)
```

## 🔴 Essential Development Commands (CRITICAL - LEARN THESE FIRST)

### 🔴 Development Setup (REQUIRED)
```bash
# ONE-COMMAND SETUP (Use this FIRST)
make dev-setup     # Sets up entire development environment

# Alternative manual setup
uv sync            # Install dependencies
uv pip install -e . # Editable install

# Verify installation
make verify-setup  # Check all tools and dependencies
```

### 🔴 Testing (CRITICAL)
```bash
# PRIMARY TESTING COMMANDS (use these)
make test          # Run ALL tests with coverage (MAIN COMMAND)
make test-quick    # Fast tests without coverage
make test-unit     # Unit tests only
make test-integration # Integration tests only

# Specific testing (when needed)
make test-file FILE=test_indexer.py    # Single test file
make test-pattern PATTERN="search"      # Pattern matching
make test-debug    # Tests with debugging output
```

### 🔴 Code Quality (REQUIRED BEFORE COMMITS)
```bash
# PRIMARY QUALITY COMMANDS (use these)
make quality       # Run ALL quality checks (lint, type, security)
make lint-fix      # Format and auto-fix all issues (MAIN COMMAND)
make typecheck     # Type checking with mypy
make security      # Security and safety checks

# Individual tools (for specific issues)
make format        # Code formatting only
make lint          # Linting without fixes
```

### 🔴 Building & Releasing (CRITICAL WORKFLOWS)
```bash
# PRIMARY RELEASE COMMANDS (single-path workflows)
make release-patch # Complete patch release (0.4.0 → 0.4.1)
make release-minor # Complete minor release (0.4.0 → 0.5.0) 
make release-major # Complete major release (0.4.0 → 1.0.0)
make publish       # Publish to PyPI (after release)

# Version management
make version-show  # Display current version
make version-check # Validate version consistency

# Build operations
make build         # Build package for distribution
make clean         # Clean all build artifacts
```

### 🟡 Local Testing (HIGH PRIORITY)
```bash
# PRIMARY TESTING WORKFLOWS
make test-deployment    # Test local deployment
make test-integration   # Test on sample projects
make test-mcp          # Test MCP server integration

# Manual testing (when needed)
./scripts/deploy-test.sh
cd ~/other-project && mcp-vector-search init && mcp-vector-search index
```

## 🟡 High-Level Architecture (HIGH PRIORITY - UNDERSTAND FOR DEVELOPMENT)

### 🔴 Module Organization (CRITICAL ARCHITECTURE)

The codebase follows a **layered architecture** with clear separation of concerns:

#### 🔴 1. CLI Layer (`src/mcp_vector_search/cli/`) - USER INTERFACE
- **Entry Point**: `main.py` - Typer app configuration and command routing
- **Commands**: `commands/` directory - each file handles specific CLI functionality
- **Output**: `output.py` - Rich-based beautiful terminal formatting
- **UX**: `didyoumean.py` - intelligent command suggestions for typos

#### 🔴 2. Core Engine (`src/mcp_vector_search/core/`) - BUSINESS LOGIC
- **Indexer**: `indexer.py` - semantic code chunking and vector indexing
- **Search**: `search.py` - vector similarity search with ranking algorithms
- **Database**: `database.py` - ChromaDB abstraction with connection pooling
- **Project**: `project.py` - project configuration and state management
- **Watcher**: `watcher.py` - file system monitoring for real-time updates
- **Auto-Indexer**: `auto_indexer.py` - intelligent reindexing strategies
- **Embeddings**: `embeddings.py` - text-to-vector transformation

#### 🟡 3. Parser System (`src/mcp_vector_search/parsers/`) - LANGUAGE SUPPORT
- **Base**: `base.py` - abstract `BaseParser` interface (extend for new languages)
- **Registry**: `registry.py` - dynamic parser discovery and selection
- **Language Parsers**: `python.py`, `javascript.py`, `typescript.py`, `text.py`
- **Output**: Each parser extracts functions, classes, methods, and semantic chunks

#### 🔴 4. MCP Integration (`src/mcp_vector_search/mcp/`) - CLAUDE INTEGRATION
- **Server**: `server.py` - Model Context Protocol server implementation
- **Tools**: Provides `search_code`, `search_similar`, `index_file` tools
- **Auto-Update**: File watching integration for real-time index updates

### 🟡 Key Design Patterns (IMPORTANT FOR CONTRIBUTORS)

#### 🟢 Parser Registry Pattern (EXTENSIBILITY)
```python
# Automatic parser registration system
from mcp_vector_search.parsers.base import BaseParser
from mcp_vector_search.parsers.registry import ParserRegistry

# Adding a new language parser (example)
class RustParser(BaseParser):
    def parse(self, content: str) -> List[CodeChunk]:
        """Extract functions, structs, traits from Rust code."""
        # Implementation here
        pass

# Auto-registration happens via __init__.py imports
```

#### 🟢 Connection Pooling (PERFORMANCE)
```python
# High-performance database connection management
from mcp_vector_search.core.database import PooledChromaVectorDatabase

database = PooledChromaVectorDatabase(
    persist_directory=config.index_path,
    use_connection_pool=True,    # 13.6% performance boost
    max_connections=10,          # Pool size
    min_connections=2,           # Always-ready connections
    max_idle_time=300.0         # 5-minute timeout
)
```

#### 🟡 Semi-Automatic Reindexing (SMART UPDATES)
Intelligent index maintenance without heavyweight daemon processes:

1. **🔴 Search-Triggered**: Checks for stale files during searches (primary)
2. **🟡 Git Hooks**: Triggers on commits/merges/checkouts (recommended)
3. **🟢 Scheduled Tasks**: System cron jobs or Windows tasks
4. **🟢 Manual Checks**: On-demand via CLI commands
5. **⚪ Periodic Checker**: In-process for long-running applications

```bash
# Setup all strategies (recommended)
mcp-vector-search auto-index setup --method all

# Check current status
mcp-vector-search auto-index status
```

### Configuration Management

Project configuration is stored in `.mcp-vector-search/config.json`:
- Auto-created during `mcp-vector-search init`
- Pydantic models for validation (`config/settings.py`)
- Default values in `config/defaults.py`
- Per-project customization supported

### Error Handling Strategy

The codebase uses custom exceptions in `core/exceptions.py`:
- `ProjectNotFoundError`: No initialized project
- `IndexNotFoundError`: No search index exists
- `ConfigurationError`: Invalid configuration
- All exceptions provide helpful user messages

### Testing Approach

Tests are organized by module in `tests/`:
- Unit tests for core functionality
- Integration tests for CLI commands
- Fixtures in `conftest.py` for reusable test data
- Async test support with `pytest-asyncio`
- Benchmarks with `pytest-benchmark`

## 🔴 MCP Server Integration (CRITICAL FOR CLAUDE DESKTOP)

The MCP server enables Claude Desktop to search your codebase semantically.

### 🔴 MCP Tools Available
- **`search_code`**: Semantic code search with context
- **`search_similar`**: Find similar code patterns
- **`search_context`**: Get surrounding context for code
- **`index_file`**: Index specific files on demand
- **`get_indexed_files`**: List all indexed files
- **`project_status`**: Get project indexing status

### 🔴 Claude Desktop Configuration
Add to your Claude Desktop settings (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mcp-vector-search": {
      "command": "uv",
      "args": ["run", "mcp-vector-search", "mcp"],
      "cwd": "/Users/masa/Projects/managed/mcp-vector-search"
    }
  }
}
```

### 🔴 Testing MCP Integration
```bash
# Test MCP server locally
make test-mcp

# Start MCP server manually
uv run mcp-vector-search mcp

# Verify MCP tools in Claude Desktop
# Use: "Search my code for authentication functions"
```

## 🟡 Performance Considerations (HIGH PRIORITY)

### 🔴 Automatic Optimizations (Already Enabled)
1. **Connection Pooling**: 13.6% speed improvement (auto-enabled)
2. **Incremental Indexing**: Only reindex changed files
3. **Embedding Cache**: Reuses embeddings for unchanged code
4. **Async Processing**: Non-blocking I/O operations
5. **Smart Chunking**: AST-aware code splitting for relevance

### 🟡 Performance Monitoring
```bash
# Benchmark search performance
make benchmark-search

# Profile indexing performance
make profile-indexing

# Check database statistics
mcp-vector-search status --performance
```

## 🟢 Adding New Features (MEDIUM PRIORITY)

### 🟢 Adding a New Language Parser
**Step-by-step process**:

1. **Create parser file**: `src/mcp_vector_search/parsers/new_language.py`
2. **Extend BaseParser**: Implement required methods
3. **Extract code chunks**: Functions, classes, methods, docstrings
4. **Auto-register**: Import in `parsers/__init__.py` 
5. **Add tests**: `tests/test_parsers/test_new_language.py`
6. **Update docs**: Add to supported languages list

```bash
# Test new parser
make test-parser PARSER=new_language

# Integration test
make test-integration LANG=new_language
```

### 🟢 Adding a New CLI Command
**Step-by-step process**:

1. **Create command**: `src/mcp_vector_search/cli/commands/new_command.py`
2. **Typer integration**: Use decorators and type hints
3. **Register command**: Import in `cli/main.py`
4. **Rich output**: Use consistent formatting
5. **Add tests**: `tests/test_cli/test_new_command.py`
6. **Update help**: Add to command documentation

```bash
# Test new command
make test-cli COMMAND=new_command

# Test CLI integration
make test-cli-integration
```

### 🟡 Modifying Search Algorithm (HIGH IMPACT)
**Key components to modify**:

1. **Search Logic**: `core/search.py` - ranking and filtering
2. **Embeddings**: `core/embeddings.py` - vector generation
3. **Database**: `core/database.py` - query optimization
4. **Indexing**: `core/indexer.py` - chunking strategy

```bash
# Test search modifications
make test-search

# Benchmark search performance
make benchmark-search

# Test search accuracy
make test-search-accuracy
```

## 🔴 Important Files to Know (CRITICAL REFERENCE)

### 🔴 Core Files (Must Know)
- **Version**: `src/mcp_vector_search/__init__.py` - single source of truth
- **Entry Point**: `src/mcp_vector_search/cli/main.py` - CLI application root
- **Dependencies**: `pyproject.toml` - all package dependencies and config
- **Build System**: `Makefile` - comprehensive build and release workflows

### 🟡 Runtime Files (Important)
- **Project Config**: `.mcp-vector-search/config.json` - per-project settings
- **Vector Database**: `.mcp-vector-search/chroma_db/` - indexed code storage
- **Lock File**: `uv.lock` - exact dependency versions

### 🟢 Development Files (Reference)
- **Pre-commit**: `.pre-commit-config.yaml` - automated quality checks
- **Coverage**: `.coverage` - test coverage reports
- **Environment**: `.env` - development environment variables

## 🟡 Debugging & Troubleshooting (HIGH PRIORITY)

### 🔴 Primary Debugging Commands
```bash
# Debug mode with full logging
make debug-search QUERY="your search term"

# Test MCP integration
make debug-mcp

# Check project health
make debug-status

# Verify installation
make debug-verify
```

### 🟢 Manual Debug Commands (When Needed)
```bash
# Enable detailed logging
export LOGURU_LEVEL=DEBUG
mcp-vector-search search "query" --verbose

# Test MCP server locally
uv run mcp-vector-search mcp --debug

# Detailed project status
mcp-vector-search status --verbose --debug

# Force complete reindex
mcp-vector-search index --force --verbose

# Test auto-indexing (dry run)
mcp-vector-search auto-index check --dry-run --verbose
```

### 🟢 Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| Search returns no results | `make debug-index-status` |
| MCP server not responding | `make debug-mcp` |
| Slow search performance | `make debug-performance` |
| Build failures | `make debug-build` |

## ⚪ Current Limitations (OPTIONAL AWARENESS)

### 🟡 Known Issues (Being Addressed)
- **Tree-sitter**: Integration needs improvement (using regex fallback)
- **Language Support**: Currently Python, JavaScript, TypeScript only
- **Binary Files**: No support for notebooks, images, compiled files

### 🟢 Future Improvements
- **Parser Enhancement**: Better AST parsing with tree-sitter
- **Language Expansion**: Java, Go, Rust, C++ support planned
- **Notebook Support**: Jupyter notebook parsing
- **Performance**: Further optimization of search algorithms

---

## 🔴 CRITICAL INSTRUCTIONS FOR CLAUDE CODE

### 🔴 File Management Rules (MUST FOLLOW)
1. **NEVER create new files** unless absolutely necessary
2. **ALWAYS edit existing files** instead of creating new ones
3. **NEVER create documentation files** (*.md) proactively
4. **ONLY create docs when explicitly requested** by the user

### 🔴 Workflow Priorities (CRITICAL)
1. **Use Makefile commands** - single-path workflows are established
2. **Follow priority system** - 🔴 Critical > 🟡 High > 🟢 Medium > ⚪ Optional
3. **Check project status first** - `make debug-status` before major changes
4. **Test before committing** - `make quality` must pass

### 🔴 MCP Integration Focus
- This project is **optimized for Claude Desktop integration**
- Test MCP functionality with `make test-mcp`
- Use semantic search commands in Claude Desktop after setup
- Maintain real-time indexing for best Claude experience

---

**🏆 This CLAUDE.md is optimized for Claude Code and Claude MPM integration.**  
**📚 Use the priority system (🔴🟡🟢⚪) to focus on what matters most.**  
**🔍 Quick reference: `make help` shows all available commands.**
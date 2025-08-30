# DEVELOPER.md - Technical Architecture Guide

**Comprehensive development guide for MCP Vector Search contributors and maintainers**

> 🎯 **Quick Start**: New developers should read [CLAUDE.md](CLAUDE.md) first for priority-based instructions.

## 📋 Table of Contents

- [🏗️ Architecture Overview](#-architecture-overview)
- [📦 Module Deep Dive](#-module-deep-dive)
- [🔧 Development Workflow](#-development-workflow)
- [🧪 Testing Strategy](#-testing-strategy)
- [📊 Performance & Monitoring](#-performance--monitoring)
- [🔌 MCP Integration](#-mcp-integration)
- [🚀 Deployment Pipeline](#-deployment-pipeline)
- [🔍 Debugging & Profiling](#-debugging--profiling)

---

## 🏗️ Architecture Overview

### System Design Philosophy

MCP Vector Search follows a **layered, async-first architecture** with these core principles:

1. **Separation of Concerns**: Clear boundaries between CLI, core logic, parsers, and MCP integration
2. **Extensibility**: Plugin-style architecture for parsers and embedding models
3. **Performance**: Connection pooling, async operations, and intelligent caching
4. **Developer Experience**: Rich CLI output, comprehensive error handling, and debugging support
5. **AI Integration**: Optimized for Claude Desktop and Model Context Protocol

### Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                 CLI Layer (Typer)                   │
├─────────────────────────────────────────────────────┤
│              MCP Server (Protocol)                  │
├─────────────────────────────────────────────────────┤
│               Core Engine (Business Logic)          │
├─────────────────────────────────────────────────────┤
│            Parser System (Language Support)         │
├─────────────────────────────────────────────────────┤
│           Database Layer (ChromaDB + Pooling)       │
├─────────────────────────────────────────────────────┤
│              Utilities (Config, Timing, etc.)       │
└─────────────────────────────────────────────────────┘
```

### Key Statistics

- **Total LOC**: ~14,000 lines of Python code across 49 files
- **Main Components**: 6 core modules, 8 CLI commands, 4 language parsers
- **Dependencies**: 15+ production dependencies, modern Python 3.11+ async stack
- **Test Coverage**: Comprehensive test suite with pytest + benchmarks

---

## 📦 Module Deep Dive

### 🎯 CLI Layer (`src/mcp_vector_search/cli/`)

**Purpose**: User interface and command routing  
**Key Files**: 268 LOC in `main.py`, plus command modules

#### Core Components:
- **`main.py`**: Typer app configuration with "did you mean" functionality
- **`commands/`**: Modular command implementations (init, search, index, etc.)
- **`output.py`**: Rich-based beautiful terminal formatting
- **`didyoumean.py`**: Intelligent command suggestions for typos

#### Command Structure:
```python
# Each command follows this pattern
@app.command()
def command_name(
    arg1: str = typer.Argument(..., help="Description"),
    option1: bool = typer.Option(False, help="Description")
) -> None:
    """Command description with rich help."""
    # Implementation with rich output and error handling
```

### 🔍 Core Engine (`src/mcp_vector_search/core/`)

**Purpose**: Business logic and semantic processing  
**Key Files**: 4,500+ LOC across indexer, search, database, and utilities

#### Major Classes:

##### `SemanticIndexer` (474 LOC)
- **Responsibility**: Code chunking and vector indexing
- **Key Methods**: `index_files()`, `_chunk_code()`, `_create_embeddings()`
- **Features**: AST-aware chunking, incremental indexing, metadata tracking

##### `SemanticSearchEngine` (834 LOC)
- **Responsibility**: Vector similarity search with ranking
- **Key Methods**: `search()`, `search_similar()`, `_rank_results()`
- **Features**: Adaptive thresholds, context-aware ranking, multi-query support

##### `ChromaVectorDatabase` (948 LOC)
- **Responsibility**: Vector storage and retrieval with connection pooling
- **Key Methods**: `upsert_chunks()`, `search_similar()`, `get_collections()`
- **Features**: Connection pooling (13.6% performance boost), auto-scaling

#### Core Workflows:

1. **Indexing Pipeline**:
   ```
   Files → Parser → Code Chunks → Embeddings → Vector DB → Metadata
   ```

2. **Search Pipeline**:
   ```
   Query → Embeddings → Similarity Search → Ranking → Context → Results
   ```

### 🔧 Parser System (`src/mcp_vector_search/parsers/`)

**Purpose**: Language-specific code analysis and extraction  
**Key Files**: Registry + language-specific parsers

#### Parser Registry Pattern:
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> List[CodeChunk]:
        """Extract semantic chunks from code."""
        pass

# Auto-registration via imports
from .python import PythonParser
from .javascript import JavaScriptParser
# New parsers automatically available
```

#### Supported Languages:
- **Python**: Functions, classes, methods, docstrings (AST-based)
- **JavaScript**: Functions, classes, JSDoc, ES6+ syntax
- **TypeScript**: Interfaces, types, generics, decorators  
- **Text**: Fallback for unsupported files

### 🔌 MCP Integration (`src/mcp_vector_search/mcp/`)

**Purpose**: Model Context Protocol server for Claude Desktop  
**Key Files**: 725 LOC in `server.py`

#### Available MCP Tools:
```python
{
    "search_code": "Semantic code search with context",
    "search_similar": "Find similar code patterns", 
    "search_context": "Get surrounding context",
    "index_file": "Index specific files on demand",
    "get_indexed_files": "List all indexed files",
    "project_status": "Get indexing status"
}
```

#### Claude Desktop Integration:
```json
{
  "mcpServers": {
    "mcp-vector-search": {
      "command": "uv",
      "args": ["run", "mcp-vector-search", "mcp"],
      "cwd": "/path/to/project"
    }
  }
}
```

---

## 🔧 Development Workflow

### 🚀 Quick Start (New Developers)

```bash
# 1. Clone and setup
git clone <repo>
cd mcp-vector-search
make dev-setup  # ONE-COMMAND setup

# 2. Verify installation
make verify-setup

# 3. Run tests
make test

# 4. Make changes and check quality
make quality  # Before committing
```

### 📁 Project Structure

```
src/mcp_vector_search/
├── cli/                    # Command-line interface
│   ├── main.py            # Entry point (268 LOC)
│   ├── commands/          # Command implementations
│   └── output.py          # Rich formatting
├── core/                  # Business logic
│   ├── indexer.py         # Semantic indexing (474 LOC)
│   ├── search.py          # Search engine (834 LOC)  
│   ├── database.py        # Vector database (948 LOC)
│   └── models.py          # Data models
├── mcp/                   # MCP server integration
│   └── server.py          # MCP server (725 LOC)
├── parsers/               # Language parsers
│   ├── registry.py        # Parser registry (202 LOC)
│   ├── python.py          # Python AST parser
│   └── javascript.py      # JS/TS parser
└── config/                # Configuration management
    └── settings.py        # Pydantic models
```

### 🔄 Development Commands (Priority Order)

```bash
# 🔴 Critical (daily use)
make dev-setup     # Development environment setup
make quality       # All quality checks (required before commit)
make test          # Full test suite
make lint-fix      # Auto-fix formatting and linting

# 🟡 High (regular use)
make test-unit     # Unit tests only (faster feedback)
make test-mcp      # MCP integration testing
make typecheck     # MyPy type checking
make build         # Build package

# 🟢 Medium (as needed)
make test-integration    # Integration tests
make debug-search QUERY="term"  # Debug search
make clean         # Clean build artifacts

# ⚪ Optional (special cases)
make performance   # Performance benchmarks
make security      # Security scanning
```

---

## 🧪 Testing Strategy

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_cli/            # CLI command tests
├── test_core/           # Core logic tests
├── test_parsers/        # Parser tests
├── test_mcp/           # MCP integration tests
└── benchmarks/         # Performance tests
```

### Testing Philosophy

1. **Unit Tests**: Fast, isolated, test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **CLI Tests**: Test command-line interface end-to-end
4. **Performance Tests**: Benchmark critical paths
5. **MCP Tests**: Test Claude Desktop integration

### Test Commands

```bash
# Primary testing workflows
make test          # All tests with coverage (main command)
make test-quick    # Fast tests without coverage  
make test-unit     # Unit tests only
make test-integration  # Integration tests only
make test-mcp      # MCP server integration

# Specific testing
make test-file FILE=test_indexer.py    # Single test file
make test-pattern PATTERN="search"     # Pattern matching
make test-debug    # Tests with debugging output
```

### Coverage Goals

- **Overall**: >90% code coverage
- **Core modules**: >95% coverage
- **CLI commands**: >85% coverage
- **Critical paths**: 100% coverage

---

## 📊 Performance & Monitoring

### Performance Optimizations

1. **Connection Pooling**: 13.6% performance improvement
2. **Async Operations**: Non-blocking I/O throughout
3. **Intelligent Caching**: Embedding and metadata caching
4. **Incremental Indexing**: Only reprocess changed files
5. **AST-Aware Chunking**: Better semantic boundaries

### Benchmarking

```bash
# Performance monitoring commands
make benchmark-search      # Search performance
make profile-indexing     # Indexing performance
make debug-performance    # Performance debugging
```

### Performance Targets

- **Search Latency**: <100ms for typical queries
- **Indexing Speed**: ~1000 files/minute
- **Memory Usage**: <50MB baseline + 1MB per 1000 chunks
- **Storage Efficiency**: ~1KB per code chunk

### Monitoring

```bash
# Check performance in development
mcp-vector-search status --performance
make debug-status  # Comprehensive health check
```

---

## 🔌 MCP Integration

### MCP Server Architecture

The MCP server enables Claude Desktop to search your codebase semantically:

```python
# Server implementation highlights
class MCPVectorSearchServer:
    async def search_code(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Semantic code search with context."""
        
    async def search_similar(self, reference: str) -> List[SearchResult]:
        """Find similar code patterns."""
        
    async def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index specific files on demand."""
```

### Claude Desktop Setup

1. **Configuration**: Add to Claude Desktop settings
2. **Testing**: Use `make test-mcp` to verify integration  
3. **Usage**: "Search my code for authentication functions"
4. **Debugging**: Use `make debug-mcp` for troubleshooting

### MCP Tool Capabilities

- **Real-time Search**: Instant semantic search in Claude Desktop
- **Context Awareness**: Provides surrounding code context
- **File Indexing**: On-demand indexing of specific files
- **Project Status**: Integration with project health monitoring

---

## 🚀 Deployment Pipeline

### Release Workflow (Automated)

```bash
# Single-command releases
make release-patch    # 0.4.13 → 0.4.14 (bug fixes)
make release-minor    # 0.4.13 → 0.5.0 (new features)  
make release-major    # 0.4.13 → 1.0.0 (breaking changes)

# After release
make publish          # Publish to PyPI
```

### Version Management

- **Source of Truth**: `src/mcp_vector_search/__init__.py`
- **Semantic Versioning**: Major.Minor.Patch format
- **Build Numbers**: Incremented automatically
- **Validation**: `make version-check` ensures consistency

### Deployment Verification

```bash
# Test deployment locally
make test-deployment

# Integration testing
make test-integration

# Verify published package
pip install mcp-vector-search --upgrade
```

---

## 🔍 Debugging & Profiling

### Debug Commands (From CLAUDE.md)

```bash
# 🔴 Primary debugging
make debug-search QUERY="term"    # Debug search with full logging
make debug-mcp                    # Debug MCP server
make debug-status                 # Project health check
make debug-verify                 # Installation verification

# 🟡 Specific debugging  
make debug-index-status          # Index health and status
make debug-performance          # Performance bottlenecks
make debug-build                # Build process issues
```

### Logging Configuration

```bash
# Enable debug logging
export LOGURU_LEVEL=DEBUG
mcp-vector-search search "query" --verbose

# MCP server debugging
LOGURU_LEVEL=DEBUG uv run mcp-vector-search mcp --debug
```

### Common Debug Scenarios

| Issue | Debug Command | Solution |
|-------|---------------|----------|
| Search returns no results | `make debug-index-status` | Check index health |
| MCP server not responding | `make debug-mcp` | Verify MCP integration |
| Slow search performance | `make debug-performance` | Profile bottlenecks |
| Build failures | `make debug-build` | Check build process |
| Installation issues | `make debug-verify` | Verify setup |

### Profiling Tools

```python
# Built-in timing decorators
from mcp_vector_search.utils.timing import timing_decorator

@timing_decorator
def slow_function():
    # Implementation
    pass
```

### Error Handling Strategy

1. **Custom Exceptions**: Defined in `core/exceptions.py`
2. **Rich Error Output**: Beautiful error messages with context
3. **Logging**: Structured logging with Loguru
4. **Debugging**: Comprehensive debug commands and verbose modes

---

## 🎯 Contributing Guidelines

### Code Quality Standards

- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all classes/functions
- **Error Handling**: Comprehensive exception handling
- **Testing**: Tests required for all new functionality
- **Performance**: Consider performance impact of changes

### Pull Request Process

1. **Quality Check**: `make quality` must pass
2. **Tests**: All tests must pass (`make test`)
3. **Documentation**: Update relevant documentation
4. **MCP Testing**: Test MCP integration if relevant
5. **Performance**: Consider performance implications

### Development Best Practices

- **Single-Path Commands**: Use established Makefile workflows
- **Priority System**: Follow 🔴🟡🟢⚪ priority system from CLAUDE.md
- **Error Messages**: Provide helpful, actionable error messages
- **Rich Output**: Use Rich library for beautiful terminal output
- **Async-First**: Use async/await for I/O operations

---

**📚 Related Documentation**:
- [CLAUDE.md](CLAUDE.md) - Priority-based development guide for Claude Code
- [README.md](README.md) - Project overview and quick start
- [docs/STRUCTURE.md](docs/STRUCTURE.md) - Detailed project structure
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development workflows

---

*This document is optimized for developers working on MCP Vector Search. For AI agent instructions, see [CLAUDE.md](CLAUDE.md).*
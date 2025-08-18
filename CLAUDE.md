# MCP Vector Search - Project Documentation Index

## 📋 Project Overview

**MCP Vector Search** is a CLI-first semantic code search tool with MCP (Model Context Protocol) integration. It provides intelligent code search capabilities using vector embeddings and AST-aware parsing.

### 🎯 Core Purpose
- **Semantic Code Search**: Find code by meaning, not just text matching
- **Multi-language Support**: Python, JavaScript, TypeScript with extensible architecture
- **Real-time Updates**: File watching with incremental indexing
- **Local-first Privacy**: Complete on-device processing
- **Developer Productivity**: Fast, intelligent code discovery

### 🏗️ Architecture Overview
- **CLI Interface**: Typer-based command-line tool
- **Vector Database**: ChromaDB for semantic embeddings
- **AST Parsing**: Tree-sitter with regex fallback
- **File Watching**: Real-time change detection
- **Async Processing**: Modern Python with type safety

---

## 📚 Documentation Structure

### 🚀 Quick Start
- **[README.md](README.md)** - Installation, usage, and basic examples
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Three-stage development workflow

### 🏗️ Architecture & Structure
- **[docs/STRUCTURE.md](docs/STRUCTURE.md)** - File organization and module architecture
- **[docs/DEPLOY.md](docs/DEPLOY.md)** - Deployment and installation instructions
- **[docs/architecture/REINDEXING_WORKFLOW.md](docs/architecture/REINDEXING_WORKFLOW.md)** - Reindexing implementation and workflow analysis

### 👨‍💻 Developer Resources
- **[docs/developer/](docs/developer/)** - Detailed developer documentation
  - **[CONTRIBUTING.md](docs/developer/CONTRIBUTING.md)** - Contribution guidelines
  - **[API.md](docs/developer/API.md)** - Internal API documentation
  - **[TESTING.md](docs/developer/TESTING.md)** - Testing strategies and practices
  - **[LINTING.md](docs/developer/LINTING.md)** - Code quality and linting setup

### 🚀 Performance & Features
- **[docs/FEATURES.md](docs/FEATURES.md)** - Comprehensive feature overview and usage guide
- **[docs/IMPROVEMENTS_SUMMARY.md](docs/IMPROVEMENTS_SUMMARY.md)** - Summary of major improvements and performance gains
- **[docs/performance/CONNECTION_POOLING.md](docs/performance/CONNECTION_POOLING.md)** - Connection pooling implementation and benchmarks
- **[examples/connection_pooling_example.py](examples/connection_pooling_example.py)** - Connection pooling usage examples
- **[examples/semi_automatic_reindexing_demo.py](examples/semi_automatic_reindexing_demo.py)** - Semi-automatic reindexing strategies demo

### 📦 Release Management
- **[docs/VERSIONING.md](docs/VERSIONING.md)** - Semantic versioning guidelines
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Version history and changes
- **[docs/RELEASES.md](docs/RELEASES.md)** - Release process and best practices

---

## 🔧 Major Functions & Components

### Core Modules

#### **CLI Interface** (`src/mcp_vector_search/cli/`)
- **`main.py`** - Entry point and command routing
- **`commands/`** - Individual command implementations
  - `init.py` - Project initialization
  - `index.py` - Codebase indexing
  - `search.py` - Semantic search
  - `watch.py` - File watching
  - `status.py` - Project statistics
  - `config.py` - Configuration management
  - `auto_index.py` - Automatic reindexing management

#### **Core Engine** (`src/mcp_vector_search/core/`)
- **`indexer.py`** - Code indexing and chunking
- **`search.py`** - Semantic search implementation
- **`database.py`** - Vector database abstraction with connection pooling
- **`connection_pool.py`** - Database connection pooling for performance
- **`auto_indexer.py`** - Semi-automatic reindexing strategies
- **`git_hooks.py`** - Git hooks integration for auto-reindexing
- **`scheduler.py`** - Scheduled task management for auto-reindexing
- **`embeddings.py`** - Text embedding generation
- **`project.py`** - Project management
- **`watcher.py`** - File system monitoring

#### **Language Parsers** (`src/mcp_vector_search/parsers/`)
- **`base.py`** - Abstract parser interface
- **`python.py`** - Python AST parsing
- **`javascript.py`** - JavaScript/TypeScript parsing
- **`registry.py`** - Parser registration system

#### **Configuration** (`src/mcp_vector_search/config/`)
- **`settings.py`** - Application settings
- **`defaults.py`** - Default configurations

### Key Algorithms

#### **Semantic Chunking**
```python
# Intelligent code chunking for optimal search
def chunk_code(content: str, language: str) -> List[CodeChunk]:
    # AST-aware chunking with function/class boundaries
    # Fallback to regex-based chunking
    # Preserve context and relationships
```

#### **Vector Search**
```python
# Similarity-based code search
def search_similar(query: str, limit: int = 10) -> List[SearchResult]:
    # Generate query embedding
    # Perform vector similarity search
    # Rank and filter results
```

#### **Incremental Indexing**
```python
# Real-time file change processing
def update_index(file_path: Path, change_type: str):
    # Detect file changes
    # Update vector database
    # Maintain index consistency
```

---

## 🎯 Quick Navigation

### For Users
1. **Getting Started**: [README.md](README.md) → Installation & Usage
2. **Deployment**: [docs/DEPLOY.md](docs/DEPLOY.md) → Production setup

### For Developers
1. **Development Setup**: [DEVELOPMENT.md](DEVELOPMENT.md) → Three-stage workflow
2. **Code Structure**: [docs/STRUCTURE.md](docs/STRUCTURE.md) → Architecture overview
3. **Contributing**: [docs/developer/CONTRIBUTING.md](docs/developer/CONTRIBUTING.md) → Guidelines
4. **API Reference**: [docs/developer/API.md](docs/developer/API.md) → Internal APIs

### For Maintainers
1. **Versioning**: [docs/VERSIONING.md](docs/VERSIONING.md) → Version management
2. **Releases**: [docs/RELEASES.md](docs/RELEASES.md) → Release process
3. **Changelog**: [docs/CHANGELOG.md](docs/CHANGELOG.md) → Version history

---

## 🚀 Common Tasks

### Development
```bash
./scripts/workflow.sh        # Show development workflow
./scripts/dev-test.sh        # Run development tests
uv run mcp-vector-search     # Test CLI locally
```

### Deployment
```bash
./scripts/deploy-test.sh     # Test local deployment
./scripts/publish.sh         # Publish to PyPI
```

### Versioning & Releasing
```bash
make version-show        # Display current version
make version-patch       # Bump patch version
make release-minor       # Full release with minor bump
make publish            # Publish to PyPI
```

### Usage
```bash
mcp-vector-search init       # Initialize project
mcp-vector-search index      # Index codebase
mcp-vector-search search "query"  # Search code
mcp-vector-search watch      # Start file watching
```

---

## 📈 Project Status

- **Version**: 0.0.3 (Alpha)
- **Status**: Active development
- **License**: MIT
- **Python**: 3.11+
- **Platform**: Cross-platform (macOS, Linux, Windows)

### Current Capabilities
- ✅ Multi-language code parsing (Python, JavaScript, TypeScript)
- ✅ Semantic vector search with similarity scoring
- ✅ Real-time file watching and incremental updates
- ✅ CLI interface with rich output and syntax highlighting
- ✅ Project-aware configuration management
- ✅ **Connection pooling** for 13.6% performance improvement
- ✅ **Semi-automatic reindexing** with 5 different strategies
- ✅ **Production-ready features** (error handling, monitoring, graceful degradation)
- ✅ **Git hooks integration** for development workflows
- ✅ **Scheduled tasks** for production environments

### Roadmap
- 🔄 Enhanced Tree-sitter integration
- 🔄 Additional language support
- 🔮 MCP server implementation
- 🔮 IDE extensions
- 🔮 Team collaboration features

---

**For detailed information on any topic, follow the links to specific documentation files.**

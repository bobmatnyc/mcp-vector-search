# MCP Vector Search

🔍 **CLI-first semantic code search with MCP integration**

[![PyPI version](https://badge.fury.io/py/mcp-vector-search.svg)](https://badge.fury.io/py/mcp-vector-search)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **Alpha Release (v0.0.3)**: This is an early-stage project under active development. Expect breaking changes and rough edges. Feedback and contributions are welcome!

A modern, fast, and intelligent code search tool that understands your codebase through semantic analysis and AST parsing. Built with Python, powered by ChromaDB, and designed for developer productivity.

## ✨ Features

### 🚀 **Core Capabilities**
- **Semantic Search**: Find code by meaning, not just keywords
- **AST-Aware Parsing**: Understands code structure (functions, classes, methods)
- **Multi-Language Support**: Python, JavaScript, TypeScript (with extensible architecture)
- **Real-time Indexing**: File watching with automatic index updates
- **Local-First**: Complete privacy with on-device processing
- **Zero Configuration**: Auto-detects project structure and languages

### 🛠️ **Developer Experience**
- **CLI-First Design**: Simple commands for immediate productivity
- **Rich Output**: Syntax highlighting, similarity scores, context
- **Fast Performance**: Sub-second search responses, efficient indexing
- **Modern Architecture**: Async-first, type-safe, modular design

### 🔧 **Technical Features**
- **Vector Database**: ChromaDB for efficient similarity search
- **Embedding Models**: Configurable sentence transformers
- **Incremental Updates**: Smart file watching and re-indexing
- **Extensible Parsers**: Plugin architecture for new languages
- **Configuration Management**: Project-specific settings

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install mcp-vector-search

# Or with UV (recommended)
uv add mcp-vector-search

# Or install from source
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search
uv sync && uv pip install -e .
```

### Basic Usage

```bash
# Initialize your project
mcp-vector-search init

# Index your codebase
mcp-vector-search index

# Search your code
mcp-vector-search search "authentication logic"
mcp-vector-search search "database connection setup"
mcp-vector-search search "error handling patterns"

# Check project status
mcp-vector-search status

# Start file watching (auto-update index)
mcp-vector-search watch
```

## 📖 Documentation

### Commands

#### `init` - Initialize Project
```bash
# Basic initialization
mcp-vector-search init

# Custom configuration
mcp-vector-search init --extensions .py,.js,.ts --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Force re-initialization
mcp-vector-search init --force
```

#### `index` - Index Codebase
```bash
# Index all files
mcp-vector-search index

# Index specific directory
mcp-vector-search index /path/to/code

# Force re-indexing
mcp-vector-search index --force
```

#### `search` - Semantic Search
```bash
# Basic search
mcp-vector-search search "function that handles user authentication"

# Adjust similarity threshold
mcp-vector-search search "database queries" --threshold 0.7

# Limit results
mcp-vector-search search "error handling" --limit 10

# Search in specific context
mcp-vector-search search similar "path/to/function.py:25"
```

#### `watch` - File Watching
```bash
# Start watching for changes
mcp-vector-search watch

# Check watch status
mcp-vector-search watch status

# Enable/disable watching
mcp-vector-search watch enable
mcp-vector-search watch disable
```

#### `status` - Project Information
```bash
# Basic status
mcp-vector-search status

# Detailed information
mcp-vector-search status --verbose
```

#### `config` - Configuration Management
```bash
# View configuration
mcp-vector-search config show

# Update settings
mcp-vector-search config set similarity_threshold 0.8
mcp-vector-search config set embedding_model microsoft/codebert-base

# List available models
mcp-vector-search config models
```

### Configuration

Projects are configured via `.mcp-vector-search/config.json`:

```json
{
  "project_root": "/path/to/project",
  "file_extensions": [".py", ".js", ".ts"],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "similarity_threshold": 0.75,
  "languages": ["python", "javascript", "typescript"],
  "watch_files": true,
  "cache_embeddings": true
}
```

## 🏗️ Architecture

### Core Components

- **Parser Registry**: Extensible system for language-specific parsing
- **Semantic Indexer**: Efficient code chunking and embedding generation
- **Vector Database**: ChromaDB integration for similarity search
- **File Watcher**: Real-time monitoring and incremental updates
- **CLI Interface**: Rich, user-friendly command-line experience

### Supported Languages

| Language   | Status | Features |
|------------|--------|----------|
| Python     | ✅ Full | Functions, classes, methods, docstrings |
| JavaScript | ✅ Full | Functions, classes, JSDoc, ES6+ syntax |
| TypeScript | ✅ Full | Interfaces, types, generics, decorators |
| Java       | 🔄 Planned | Classes, methods, annotations |
| Go         | 🔄 Planned | Functions, structs, interfaces |
| Rust       | 🔄 Planned | Functions, structs, traits |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search

# Install dependencies with UV
uv sync

# Install in development mode
uv pip install -e .

# Run tests
uv run pytest

# Run linting
uv run ruff check
uv run mypy src/
```

### Adding Language Support

1. Create a new parser in `src/mcp_vector_search/parsers/`
2. Extend the `BaseParser` class
3. Register the parser in `parsers/registry.py`
4. Add tests and documentation

## 📊 Performance

- **Indexing Speed**: ~1000 files/minute (typical Python project)
- **Search Latency**: <100ms for most queries
- **Memory Usage**: ~50MB baseline + ~1MB per 1000 code chunks
- **Storage**: ~1KB per code chunk (compressed embeddings)

## ⚠️ Known Limitations (Alpha)

- **Tree-sitter Integration**: Currently using regex fallback parsing (Tree-sitter setup needs improvement)
- **Search Relevance**: Embedding model may need tuning for code-specific queries
- **Error Handling**: Some edge cases may not be gracefully handled
- **Documentation**: API documentation is minimal
- **Testing**: Limited test coverage, needs real-world validation

## 🙏 Feedback Needed

We're actively seeking feedback on:

- **Search Quality**: How relevant are the search results for your codebase?
- **Performance**: How does indexing and search speed feel in practice?
- **Usability**: Is the CLI interface intuitive and helpful?
- **Language Support**: Which languages would you like to see added next?
- **Features**: What functionality is missing for your workflow?

Please [open an issue](https://github.com/bobmatnyc/mcp-vector-search/issues) or start a [discussion](https://github.com/bobmatnyc/mcp-vector-search/discussions) to share your experience!

## 🔮 Roadmap

### v0.0.x: Alpha (Current) 🔄
- [x] Core CLI interface
- [x] Python/JS/TS parsing
- [x] ChromaDB integration
- [x] File watching
- [x] Basic search functionality
- [ ] Real-world testing and feedback
- [ ] Bug fixes and stability improvements
- [ ] Performance optimizations

### v0.1.x: Beta 🔮
- [ ] Advanced search modes (contextual, similar code)
- [ ] Additional language support (Java, Go, Rust)
- [ ] Configuration improvements
- [ ] Comprehensive testing suite
- [ ] Documentation improvements

### v1.0.x: Stable 🔮
- [ ] MCP server implementation
- [ ] IDE extensions (VS Code, JetBrains)
- [ ] Git integration
- [ ] Team collaboration features
- [ ] Production-ready performance

## 🛠️ Development

### Three-Stage Development Workflow

**Stage A: Local Development & Testing**
```bash
# Setup development environment
uv sync && uv pip install -e .

# Run development tests
./scripts/dev-test.sh

# Test CLI locally
uv run mcp-vector-search version
```

**Stage B: Local Deployment Testing**
```bash
# Build and test clean deployment
./scripts/deploy-test.sh

# Test on other projects
cd ~/other-project
mcp-vector-search init && mcp-vector-search index
```

**Stage C: PyPI Publication**
```bash
# Publish to PyPI
./scripts/publish.sh

# Verify published version
pip install mcp-vector-search --upgrade
```

### Quick Reference
```bash
./scripts/workflow.sh  # Show workflow overview
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

## 📚 Documentation

For comprehensive documentation, see **[CLAUDE.md](CLAUDE.md)** - the main documentation index.

### Quick Links
- **[Installation & Deployment](docs/DEPLOY.md)** - Setup and deployment guide
- **[Project Structure](docs/STRUCTURE.md)** - Architecture and file organization
- **[Contributing Guidelines](docs/developer/CONTRIBUTING.md)** - How to contribute
- **[API Reference](docs/developer/API.md)** - Internal API documentation
- **[Testing Guide](docs/developer/TESTING.md)** - Testing strategies
- **[Code Quality](docs/developer/LINTING.md)** - Linting and formatting
- **[Versioning](docs/VERSIONING.md)** - Version management
- **[Releases](docs/RELEASES.md)** - Release process

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ChromaDB](https://github.com/chroma-core/chroma) for vector database
- [Tree-sitter](https://tree-sitter.github.io/) for parsing infrastructure
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Typer](https://typer.tiangolo.com/) for CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output

---

**Built with ❤️ for developers who love efficient code search**

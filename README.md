# MCP Vector Search

🔍 **CLI-first semantic code search with MCP integration**

[![PyPI version](https://badge.fury.io/py/mcp-vector-search.svg)](https://badge.fury.io/py/mcp-vector-search)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **Alpha Release (v0.12.7)**: This is an early-stage project under active development. Expect breaking changes and rough edges. Feedback and contributions are welcome!

A modern, fast, and intelligent code search tool that understands your codebase through semantic analysis and AST parsing. Built with Python, powered by ChromaDB, and designed for developer productivity.

## ✨ Features

### 🚀 **Core Capabilities**
- **Semantic Search**: Find code by meaning, not just keywords
- **AST-Aware Parsing**: Understands code structure (functions, classes, methods)
- **Multi-Language Support**: 8 languages - Python, JavaScript, TypeScript, Dart/Flutter, PHP, Ruby, HTML, and Markdown/Text (with extensible architecture)
- **Real-time Indexing**: File watching with automatic index updates
- **Automatic Version Tracking**: Smart reindexing on tool upgrades
- **Local-First**: Complete privacy with on-device processing
- **Zero Configuration**: Auto-detects project structure and languages

### 🛠️ **Developer Experience**
- **CLI-First Design**: Simple commands for immediate productivity
- **Rich Output**: Syntax highlighting, similarity scores, context
- **Fast Performance**: Sub-second search responses, efficient indexing
- **Modern Architecture**: Async-first, type-safe, modular design
- **Semi-Automatic Reindexing**: Multiple strategies without daemon processes

### 🔧 **Technical Features**
- **Vector Database**: ChromaDB with connection pooling for 13.6% performance boost
- **Embedding Models**: Configurable sentence transformers
- **Smart Reindexing**: Search-triggered, Git hooks, scheduled tasks, and manual options
- **Extensible Parsers**: Plugin architecture for new languages
- **Configuration Management**: Project-specific settings
- **Production Ready**: Connection pooling, auto-indexing, comprehensive error handling

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install mcp-vector-search

# Or with UV (faster)
uv pip install mcp-vector-search

# Or install from source
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search
uv sync && uv pip install -e .
```

**Verify Installation:**
```bash
# Check that all dependencies are installed correctly
mcp-vector-search doctor

# Should show all ✓ marks
# If you see missing dependencies, try:
pip install --upgrade mcp-vector-search
```

### Zero-Config Setup (Recommended)

The fastest way to get started - **completely hands-off, just one command**:

```bash
# Smart zero-config setup (recommended)
mcp-vector-search setup
```

**What `setup` does automatically:**
- ✅ Detects your project's languages and file types
- ✅ Initializes semantic search with optimal settings
- ✅ Indexes your entire codebase
- ✅ Configures ALL installed MCP platforms (Claude Code, Cursor, etc.)
- ✅ **Uses native Claude CLI integration** (`claude mcp add`) when available
- ✅ **Falls back to `.mcp.json`** if Claude CLI not available
- ✅ Sets up file watching for auto-reindex
- ✅ **Zero user input required!**

**Behind the scenes:**
- **Server name**: `mcp` (for consistency with other MCP projects)
- **Command**: `uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}`
- **File watching**: Enabled via `MCP_ENABLE_FILE_WATCHING=true`
- **Integration method**: Native `claude mcp add` (or `.mcp.json` fallback)

**Example output:**
```
🚀 Smart Setup for mcp-vector-search
🔍 Detecting project...
   ✅ Found 3 language(s): Python, JavaScript, TypeScript
   ✅ Detected 8 file type(s)
   ✅ Found 2 platform(s): claude-code, cursor
⚙️  Configuring...
   ✅ Embedding model: sentence-transformers/all-MiniLM-L6-v2
🚀 Initializing...
   ✅ Vector database created
   ✅ Configuration saved
🔍 Indexing codebase...
   ✅ Indexing completed in 12.3s
🔗 Configuring MCP integrations...
   ✅ Using Claude CLI for automatic setup
   ✅ Registered with Claude CLI
   ✅ Configured 2 platform(s)
🎉 Setup Complete!
```

**Options:**
```bash
# Force re-setup
mcp-vector-search setup --force

# Verbose output for debugging (shows Claude CLI commands)
mcp-vector-search setup --verbose
```

### Advanced Setup Options

For more control over the installation process:

```bash
# Manual setup with MCP integration
mcp-vector-search install --with-mcp

# Custom file extensions
mcp-vector-search install --extensions .py,.js,.ts,.dart

# Skip automatic indexing
mcp-vector-search install --no-auto-index

# Just initialize (no indexing or MCP)
mcp-vector-search init
```

### Add MCP Integration for AI Tools

**Automatic (Recommended):**
```bash
# One command sets up all detected platforms
mcp-vector-search setup
```

**Manual Platform Installation:**
```bash
# Add Claude Code integration (project-scoped)
mcp-vector-search install claude-code

# Add Cursor IDE integration (global)
mcp-vector-search install cursor

# See all available platforms
mcp-vector-search install list
```

**Note**: The `setup` command uses native `claude mcp add` when Claude CLI is available, providing better integration than manual `.mcp.json` creation.

### Remove MCP Integrations

```bash
# Remove specific platform
mcp-vector-search uninstall claude-code

# Remove all integrations
mcp-vector-search uninstall --all

# List configured integrations
mcp-vector-search uninstall list
```

### Basic Usage

```bash
# Search your code
mcp-vector-search search "authentication logic"
mcp-vector-search search "database connection setup"
mcp-vector-search search "error handling patterns"

# Index your codebase (if not done during setup)
mcp-vector-search index

# Check project status
mcp-vector-search status

# Start file watching (auto-update index)
mcp-vector-search watch
```

### Smart CLI with "Did You Mean" Suggestions

The CLI includes intelligent command suggestions for typos:

```bash
# Typos are automatically detected and corrected
$ mcp-vector-search serach "auth"
No such command 'serach'. Did you mean 'search'?

$ mcp-vector-search indx
No such command 'indx'. Did you mean 'index'?
```

See [docs/guides/cli-usage.md](docs/guides/cli-usage.md) for more details.

## Versioning & Releasing

This project uses semantic versioning with an automated release workflow.

### Quick Commands
- `make version-show` - Display current version
- `make release-patch` - Create patch release
- `make publish` - Publish to PyPI

See [docs/development/versioning.md](docs/development/versioning.md) for complete documentation.

## 📖 Documentation

### Commands

#### `setup` - Zero-Config Smart Setup (Recommended)
```bash
# One command to do everything (recommended)
mcp-vector-search setup

# What it does automatically:
# - Detects project languages and file types
# - Initializes semantic search
# - Indexes entire codebase
# - Configures all detected MCP platforms
# - Sets up file watching
# - Zero configuration needed!

# Force re-setup
mcp-vector-search setup --force

# Verbose output for debugging
mcp-vector-search setup --verbose
```

**Key Features:**
- **Zero Configuration**: No user input required
- **Smart Detection**: Automatically discovers languages and platforms
- **Comprehensive**: Handles init + index + MCP setup in one command
- **Idempotent**: Safe to run multiple times
- **Fast**: Timeout-protected scanning (won't hang on large projects)
- **Team-Friendly**: Commit `.mcp.json` to share configuration

**When to use:**
- ✅ First-time project setup
- ✅ Team onboarding
- ✅ Quick testing in new codebases
- ✅ Setting up multiple MCP platforms at once

#### `install` - Install Project and MCP Integrations (Advanced)
```bash
# Manual setup with more control
mcp-vector-search install

# Install with all MCP integrations
mcp-vector-search install --with-mcp

# Custom file extensions
mcp-vector-search install --extensions .py,.js,.ts

# Skip automatic indexing
mcp-vector-search install --no-auto-index

# Platform-specific MCP integration
mcp-vector-search install claude-code      # Project-scoped
mcp-vector-search install cursor           # Global
mcp-vector-search install windsurf         # Global
mcp-vector-search install vscode           # Global

# List available platforms
mcp-vector-search install list
```

**When to use:**
- Use `install` when you need fine-grained control over extensions, models, or MCP platforms
- Use `setup` for quick, zero-config onboarding (recommended)

#### `uninstall` - Remove MCP Integrations
```bash
# Remove specific platform
mcp-vector-search uninstall claude-code

# Remove all integrations
mcp-vector-search uninstall --all

# List configured integrations
mcp-vector-search uninstall list

# Skip backup creation
mcp-vector-search uninstall claude-code --no-backup

# Alias (same as uninstall)
mcp-vector-search remove claude-code
```

#### `init` - Initialize Project (Simple)
```bash
# Basic initialization (no indexing or MCP)
mcp-vector-search init

# Custom configuration
mcp-vector-search init --extensions .py,.js,.ts --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Force re-initialization
mcp-vector-search init --force
```

**Note**: For most users, use `setup` instead of `init`. The `init` command is for advanced users who want manual control.

#### `index` - Index Codebase
```bash
# Index all files
mcp-vector-search index

# Index specific directory
mcp-vector-search index /path/to/code

# Force re-indexing
mcp-vector-search index --force

# Reindex entire project
mcp-vector-search index reindex

# Reindex entire project (explicit)
mcp-vector-search index reindex --all

# Reindex entire project without confirmation
mcp-vector-search index reindex --force

# Reindex specific file
mcp-vector-search index reindex path/to/file.py
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

#### `auto-index` - Automatic Reindexing
```bash
# Setup all auto-indexing strategies
mcp-vector-search auto-index setup --method all

# Setup specific strategies
mcp-vector-search auto-index setup --method git-hooks
mcp-vector-search auto-index setup --method scheduled --interval 60

# Check for stale files and auto-reindex
mcp-vector-search auto-index check --auto-reindex --max-files 10

# View auto-indexing status
mcp-vector-search auto-index status

# Remove auto-indexing setup
mcp-vector-search auto-index teardown --method all
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

# Configure indexing behavior
mcp-vector-search config set skip_dotfiles true    # Skip dotfiles (default)
mcp-vector-search config set respect_gitignore true # Respect .gitignore (default)

# Get specific setting
mcp-vector-search config get skip_dotfiles
mcp-vector-search config get respect_gitignore

# List available models
mcp-vector-search config models

# List all configuration keys
mcp-vector-search config list-keys
```

## 🚀 Performance Features

### Connection Pooling
Automatic connection pooling provides **13.6% performance improvement** with zero configuration:

```python
# Automatically enabled for high-throughput scenarios
from mcp_vector_search.core.database import PooledChromaVectorDatabase

database = PooledChromaVectorDatabase(
    max_connections=10,    # Pool size
    min_connections=2,     # Warm connections
    max_idle_time=300.0,   # 5 minutes
)
```

### Semi-Automatic Reindexing
Multiple strategies to keep your index up-to-date without daemon processes:

1. **Search-Triggered**: Automatically checks for stale files during searches
2. **Git Hooks**: Triggers reindexing after commits, merges, checkouts
3. **Scheduled Tasks**: System-level cron jobs or Windows tasks
4. **Manual Checks**: On-demand via CLI commands
5. **Periodic Checker**: In-process periodic checks for long-running apps

```bash
# Setup all strategies
mcp-vector-search auto-index setup --method all

# Check status
mcp-vector-search auto-index status
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
  "cache_embeddings": true,
  "skip_dotfiles": true,
  "respect_gitignore": true
}
```

#### Indexing Configuration Options

**`skip_dotfiles`** (default: `true`)
- Controls whether files and directories starting with "." are skipped during indexing
- **Whitelisted directories** are always indexed regardless of this setting:
  - `.github/` - GitHub workflows and actions
  - `.gitlab-ci/` - GitLab CI configuration
  - `.circleci/` - CircleCI configuration
- When `false`: All dotfiles are indexed (subject to gitignore rules if `respect_gitignore` is `true`)

**`respect_gitignore`** (default: `true`)
- Controls whether `.gitignore` patterns are respected during indexing
- When `false`: Files in `.gitignore` are indexed (subject to `skip_dotfiles` if enabled)

#### Configuration Use Cases

**Default Behavior** (Recommended for most projects):
```bash
# Skip dotfiles AND respect .gitignore
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true
```

**Index Everything** (Useful for deep code analysis):
```bash
# Index all files including dotfiles and gitignored files
mcp-vector-search config set skip_dotfiles false
mcp-vector-search config set respect_gitignore false
```

**Index Dotfiles but Respect .gitignore**:
```bash
# Index configuration files but skip build artifacts
mcp-vector-search config set skip_dotfiles false
mcp-vector-search config set respect_gitignore true
```

**Skip Dotfiles but Ignore .gitignore**:
```bash
# Useful when you want to index files in .gitignore but skip hidden config files
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore false
```

## 🏗️ Architecture

### Core Components

- **Parser Registry**: Extensible system for language-specific parsing
- **Semantic Indexer**: Efficient code chunking and embedding generation
- **Vector Database**: ChromaDB integration for similarity search
- **File Watcher**: Real-time monitoring and incremental updates
- **CLI Interface**: Rich, user-friendly command-line experience

### Supported Languages

MCP Vector Search supports **8 programming languages** with full semantic search capabilities:

| Language   | Extensions | Status | Features |
|------------|------------|--------|----------|
| Python     | `.py`, `.pyw` | ✅ Full | Functions, classes, methods, docstrings |
| JavaScript | `.js`, `.jsx`, `.mjs` | ✅ Full | Functions, classes, JSDoc, ES6+ syntax |
| TypeScript | `.ts`, `.tsx` | ✅ Full | Interfaces, types, generics, decorators |
| Dart       | `.dart` | ✅ Full | Functions, classes, widgets, async, dartdoc |
| PHP        | `.php`, `.phtml` | ✅ Full | Classes, methods, traits, PHPDoc, Laravel patterns |
| Ruby       | `.rb`, `.rake`, `.gemspec` | ✅ Full | Modules, classes, methods, RDoc, Rails patterns |
| HTML       | `.html`, `.htm` | ✅ Full | Semantic content extraction, heading hierarchy, text chunking |
| Text/Markdown | `.txt`, `.md`, `.markdown` | ✅ Basic | Semantic chunking for documentation |

**Planned Languages:**
| Language   | Status | Features |
|------------|--------|----------|
| Java       | 🔄 Planned | Classes, methods, annotations |
| Go         | 🔄 Planned | Functions, structs, interfaces |
| Rust       | 🔄 Planned | Functions, structs, traits |

#### New Language Support

**HTML Support** (Unreleased):
- **Semantic Extraction**: Content from h1-h6, p, section, article, main, aside, nav, header, footer
- **Intelligent Chunking**: Based on heading hierarchy (h1-h6)
- **Context Preservation**: Maintains class and id attributes for searchability
- **Script/Style Filtering**: Ignores non-content elements
- **Use Cases**: Static sites, documentation, web templates, HTML fragments

**Dart/Flutter Support** (v0.4.15):
- **Widget Detection**: StatelessWidget, StatefulWidget recognition
- **State Classes**: Automatic parsing of `_WidgetNameState` patterns
- **Async Support**: Future<T> and async function handling
- **Dartdoc**: Triple-slash comment extraction
- **Tree-sitter AST**: Fast, accurate parsing with regex fallback

**PHP Support** (v0.5.0):
- **Class Detection**: Classes, interfaces, traits
- **Method Extraction**: Public, private, protected, static methods
- **Magic Methods**: __construct, __get, __set, __call, etc.
- **PHPDoc**: Full comment extraction
- **Laravel Patterns**: Controllers, Models, Eloquent support
- **Tree-sitter AST**: Fast parsing with regex fallback

**Ruby Support** (v0.5.0):
- **Module/Class Detection**: Full namespace support (::)
- **Method Extraction**: Instance and class methods
- **Special Syntax**: Method names with ?, ! support
- **Attribute Macros**: attr_accessor, attr_reader, attr_writer
- **RDoc**: Comment extraction (# and =begin...=end)
- **Rails Patterns**: ActiveRecord, Controllers support
- **Tree-sitter AST**: Fast parsing with regex fallback

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search

# Install development environment (includes dependencies + editable install)
make dev

# Test CLI from source (recommended during development)
./dev-mcp version        # Shows [DEV] indicator
./dev-mcp search "test"  # No reinstall needed after code changes

# Run tests and quality checks
make test-unit           # Run unit tests
make quality            # Run linting and type checking
make fix                # Auto-fix formatting issues

# View all available targets
make help
```

For detailed development workflow and `dev-mcp` usage, see the [Development](#-development) section below.

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
make dev

# Run development tests
make test-unit

# Run CLI from source (recommended during development)
./dev-mcp version        # Visual [DEV] indicator
./dev-mcp status         # Any command works
./dev-mcp search "auth"  # Immediate feedback on changes

# Run quality checks
make quality

# Alternative: use uv run directly
uv run mcp-vector-search version
```

#### Using the `dev-mcp` Development Helper

The `./dev-mcp` script provides a streamlined way to run the CLI from source code during development, eliminating the need for repeated installations.

**Key Features:**
- **Visual [DEV] Indicator**: Shows `[DEV]` prefix to distinguish from installed version
- **No Reinstall Required**: Reflects code changes immediately
- **Complete Argument Forwarding**: Works with all CLI commands and options
- **Verbose Mode**: Debug output with `--verbose` flag
- **Built-in Help**: Script usage with `--help`

**Usage Examples:**
```bash
# Basic commands (note the [DEV] prefix in output)
./dev-mcp version
./dev-mcp status
./dev-mcp index
./dev-mcp search "authentication logic"

# With CLI options
./dev-mcp search "error handling" --limit 10
./dev-mcp index --force

# Script verbose mode (shows Python interpreter, paths)
./dev-mcp --verbose search "database"

# Script help (shows dev-mcp usage, not CLI help)
./dev-mcp --help

# CLI command help (forwards --help to the CLI)
./dev-mcp search --help
./dev-mcp index --help
```

**When to Use:**
- **`./dev-mcp`** → Development workflow (runs from source code)
- **`mcp-vector-search`** → Production usage (runs installed version via pipx/pip)

**Benefits:**
- **Instant Feedback**: Changes to source code are reflected immediately
- **No Build Step**: Skip the reinstall cycle during active development
- **Clear Context**: Visual `[DEV]` indicator prevents confusion about which version is running
- **Error Handling**: Built-in checks for uv installation and project structure

**Requirements:**
- Must have `uv` installed (`pip install uv`)
- Must run from project root directory
- Requires `pyproject.toml` in current directory

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

For comprehensive documentation, see **[docs/index.md](docs/index.md)** - the complete documentation hub.

### Getting Started
- **[Installation Guide](docs/getting-started/installation.md)** - Complete installation instructions
- **[First Steps](docs/getting-started/first-steps.md)** - Quick start tutorial
- **[Configuration](docs/getting-started/configuration.md)** - Basic configuration

### User Guides
- **[Searching Guide](docs/guides/searching.md)** - Master semantic code search
- **[Indexing Guide](docs/guides/indexing.md)** - Indexing strategies and optimization
- **[CLI Usage](docs/guides/cli-usage.md)** - Advanced CLI features
- **[MCP Integration](docs/guides/mcp-integration.md)** - AI tool integration
- **[File Watching](docs/guides/file-watching.md)** - Real-time index updates

### Reference
- **[CLI Commands](docs/reference/cli-commands.md)** - Complete command reference
- **[Configuration Options](docs/reference/configuration-options.md)** - All configuration settings
- **[Features](docs/reference/features.md)** - Feature overview
- **[Architecture](docs/reference/architecture.md)** - System architecture

### Development
- **[Contributing](docs/development/contributing.md)** - How to contribute
- **[Testing](docs/development/testing.md)** - Testing guide
- **[Code Quality](docs/development/code-quality.md)** - Linting and formatting
- **[API Reference](docs/development/api.md)** - Internal API docs
- **[Deployment](docs/deployment/README.md)** - Release and deployment guide

### Advanced
- **[Troubleshooting](docs/advanced/troubleshooting.md)** - Common issues and solutions
- **[Performance](docs/architecture/performance.md)** - Performance optimization
- **[Extending](docs/advanced/extending.md)** - Adding new features

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

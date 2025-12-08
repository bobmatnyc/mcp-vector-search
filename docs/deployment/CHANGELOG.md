# Changelog

All notable changes to MCP Vector Search will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Zero-Config Setup Command**: New `mcp-vector-search setup` command for instant onboarding
  - **Smart Auto-Detection**: Automatically detects project languages and file types
  - **Comprehensive Setup**: Combines init + index + MCP configuration in one command
  - **Multi-Platform MCP**: Configures all installed MCP platforms (Claude Code, Cursor, Windsurf, VS Code, Claude Desktop)
  - **Native Claude CLI Integration**: Uses `claude mcp add` command when Claude CLI is available
  - **Intelligent Fallback**: Creates `.mcp.json` automatically if Claude CLI unavailable
  - **Consistent Server Naming**: MCP server registered as `mcp` (not `mcp-vector-search`) for consistency
  - **File Watching**: Sets up automatic reindexing via file watching
  - **Timeout Protection**: Prevents hanging on large projects (2s scan timeout)
  - **Idempotent**: Safe to run multiple times, won't break existing configuration
  - **Zero User Input**: No questions asked, sensible defaults for everything
  - **Team-Friendly**: Creates `.mcp.json` for committing to repos
  - **Progress Reporting**: Clear, real-time feedback on each phase
  - **Flags**: `--force` to re-setup, `--verbose` for detailed output
  - **48 Comprehensive Tests**: Full test coverage including integration tests
  - **Performance**: Completes setup in seconds even for large codebases

  **Example Usage:**
  ```bash
  # One command to set everything up (recommended)
  mcp-vector-search setup

  # Force re-setup
  mcp-vector-search setup --force

  # Verbose debugging output
  mcp-vector-search setup --verbose
  ```

  **What it does:**
  1. Detects project languages (Python, JavaScript, TypeScript, etc.)
  2. Scans for file types in use (with 2s timeout)
  3. Selects optimal embedding model based on detected languages
  4. Initializes vector database and configuration
  5. Indexes entire codebase with progress reporting
  6. Detects all installed MCP platforms
  7. Configures each platform automatically (using `claude mcp add` when available)
  8. Enables file watching for automatic reindexing

  **Behind the scenes (MCP Integration):**
  - **Server name**: `mcp` (for consistency with other MCP projects)
  - **Command**: `uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}`
  - **Primary method**: `claude mcp add --transport stdio mcp --env MCP_ENABLE_FILE_WATCHING=true -- uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}`
  - **Fallback method**: Manual `.mcp.json` creation if Claude CLI unavailable

  **This is now the recommended way to get started** - replacing manual `init` + `index` + `install` workflows.

- **Enhanced Install Command with Native Claude CLI Support**: Updated `mcp-vector-search install` command with Claude CLI integration
  - **Native Integration**: Uses `claude mcp add` when Claude CLI is available
  - **Automatic Detection**: Checks for both `claude` and `uv` availability
  - **Graceful Fallback**: Falls back to `.mcp.json` creation if CLI unavailable
  - **Consistent Naming**: Registers server as `mcp` (not `mcp-vector-search`)
  - **Environment Variables**: Properly sets `MCP_ENABLE_FILE_WATCHING=true`
  - **Error Handling**: Comprehensive error handling with clear user feedback
  - **Logging**: Detailed logging for debugging with `--verbose` flag

- **Automatic .gitignore Entry Management**: Project initialization now automatically adds `.mcp-vector-search/` to `.gitignore`
  - Prevents accidental commits of local search index data to version control
  - Activates during `mcp-vector-search init` and `install` commands
  - Non-blocking operation - initialization continues even if update fails
  - Intelligent pattern detection avoids duplicate entries
  - Respects existing negation patterns (user intent to track directory)
  - Only modifies `.gitignore` in git repositories (checks for `.git` directory)
  - Handles edge cases: missing files, empty files, permission errors, encoding issues
  - Comprehensive test coverage: 20 unit tests + 8 integration tests
  - New utility module: `src/mcp_vector_search/utils/gitignore_updater.py`

- **Indexing Configuration Options**: New configuration settings for fine-grained control over indexing behavior
  - `skip_dotfiles` (default: `true`) - Controls whether dotfiles are skipped during indexing
    - Whitelisted directories (`.github/`, `.gitlab-ci/`, `.circleci/`) are always indexed
    - Improves indexing performance by 20-30% when enabled
  - `respect_gitignore` (default: `true`) - Controls whether `.gitignore` patterns are respected
    - Can improve indexing speed by 40-60% for large projects with dependencies
    - Reduces index size and improves search relevance
  - Both settings can be configured via CLI: `mcp-vector-search config set <key> <value>`
  - Four common use cases documented: default, index everything, dotfiles only, gitignore only
  - Settings work together for flexible indexing control

### Enhanced
- **Configuration Documentation**: Comprehensive configuration guide added
  - New `docs/CONFIGURATION.md` with detailed documentation
  - README.md updated with indexing configuration examples
  - Use cases and examples for different project scenarios
  - Troubleshooting guide for common indexing issues
  - Performance considerations documented

### Changed
- **Default Indexing Behavior**: Now skips dotfiles (except whitelisted) and respects `.gitignore` by default
  - Existing projects continue with their current settings
  - New projects automatically use optimized defaults
  - CI/CD configurations always indexed for searchability

## [0.13.0] - 2025-10-27

### Added
- **Hierarchical Install/Uninstall Commands**: Complete CLI command restructuring for better user experience
  - New `install` command with platform-specific subcommands
  - New `uninstall` command for removing MCP integrations
  - Support for 5 MCP platforms: claude-code, claude-desktop, cursor, windsurf, vscode
  - Project-scoped configuration (`.mcp.json`) for claude-code
  - Global-scoped configuration for other platforms (home directory)
  - `install list` and `uninstall list` for platform discovery
  - `remove` command as alias for `uninstall`
  - `--with-mcp` flag to install all MCP integrations at once
  - `--no-auto-index` flag to skip automatic indexing
  - `--no-backup` flag for uninstall operations

### Changed
- **MCP Command Reserved**: The `mcp` command is now exclusively for MCP server operations (not installation)
  - `mcp-vector-search mcp` - Start MCP server (stdio mode)
  - `mcp-vector-search mcp list` - List configured servers
  - `mcp-vector-search mcp test` - Test server startup
- **Install Command Reintroduced**: Previously deprecated `install` command is back with new hierarchical structure
  - `install` is now the primary command for project setup and MCP integration
  - `init` remains available for simple project initialization without MCP

### Deprecated
- `init-mcp` command deprecated in favor of `install claude-code` (or other platform subcommands)

### Files Modified
- `src/mcp_vector_search/cli/commands/install.py` - Rewritten with hierarchical structure
- `src/mcp_vector_search/cli/commands/uninstall.py` - New command for MCP removal
- `src/mcp_vector_search/cli/main.py` - Updated command registration and help text
- `CLAUDE.md` - Updated with new command examples and structure
- `README.md` - Updated quick start and command documentation

### Migration Guide
**Old Pattern** (v0.12.0 and earlier):
```bash
mcp-vector-search init --mcp
```

**New Pattern** (v0.13.0+):
```bash
# Quick setup
mcp-vector-search install

# Add specific platform
mcp-vector-search install claude-code
```

The new structure provides:
- Better command discoverability
- Fine-grained platform control
- Clear separation between installation and MCP server operations
- Ability to remove specific integrations

### Added
- **HTML Language Support**: Full HTML parser implementation
  - Semantic content extraction from HTML tags
  - Intelligent chunking based on heading hierarchy
  - Ignores script and style tag content
  - Extracts text from h1-h6, p, section, article, main, aside, nav, header, footer
  - Preserves class and id attributes for context
  - Supported extensions: `.html`, `.htm`

- **Automatic Version-Based Reindexing**: Smart index updates on tool upgrades
  - Tracks index version in metadata
  - Auto-reindexes on major/minor version changes (skips patch updates)
  - Search-triggered version checks with user-friendly messages
  - Configurable via `auto_reindex_on_upgrade` setting (default: true)
  - Zero user intervention required - seamless upgrade experience
  - New dependency: `packaging>=23.0` for semantic version comparison

### Enhanced
- **Search Performance Optimizations**: 60-85% faster search operations
  - Async file I/O eliminates blocking reads in hot path (+35-40% speed)
  - LRU file content caching reduces repeated disk access (+15-20% speed)
  - Health check throttling (60-second intervals) reduces overhead (+5-7% speed)
  - Connection pooling enabled by default (+13.6% speed)
  - Optimized reranking algorithm with reduced string operations (+8-12% speed)

- **Query Expansion**: Improved semantic search relevance
  - Automatic expansion of common abbreviations (auth → authentication, db → database, etc.)
  - Better matching for programming concepts (class, method, function, etc.)
  - 15-20% improvement in search result relevance
  - Pre-computed expansion dictionaries eliminate runtime overhead (+20-25% speed)

- **Text Parser**: Now supports markdown files (`.md`, `.markdown`) in addition to `.txt`

- **Language Count**: Now supports **8 languages** (up from 7):
  - Python, JavaScript, TypeScript (existing)
  - Dart, PHP, Ruby (v0.5.0)
  - **HTML** (new)
  - Text/Markdown (enhanced)

### Fixed
- Bare exception handlers now use specific exception types for better debugging
- Added comprehensive logging to exception handlers
- Expanded ignore patterns: `.mypy_cache`, `.ruff_cache`, `.claude-mpm`, `.mcp-vector-search`

### Internal
- Extracted magic numbers to project-wide constants for maintainability
- Created parser utilities module to reduce code duplication (potential -800 to -1000 LOC)
- Proper LRU cache implementation with statistics and configurable size

## [0.12.0] - 2025-10-25

### Added
- **Hierarchical Graph Visualization**: Complete directory, file, and chunk-based visualization system
  - Interactive D3.js force-directed graph with expand/collapse functionality
  - Directory nodes expand to show all files and subdirectories
  - File nodes expand to show individual code chunks with metadata
  - Visual hierarchy with optimized 35px folder/file icons
  - Cache-busting URL parameters and meta tags for fresh data loading

### Enhanced
- **Graph Link Visibility**: Improved visual clarity of relationships
  - Enhanced link colors with better opacity (0.4 for directories, 0.3 for files)
  - Dynamic link distances based on node types (150px directories, 80px files/chunks)
  - Optimized collision detection with dynamic radius calculations
  - Better auto-spacing with enhanced force simulation parameters

- **Expand/Collapse Functionality**: Robust node interaction system
  - Working directory expansion showing all child files and subdirectories
  - File expansion revealing all contained code chunks
  - Preserves original link structure before D3 modifications
  - Proper state management for expanded/collapsed nodes
  - Visual feedback with node color changes

### Fixed
- **Parent Directory Linking**: Corrected absolute to relative path conversion
  - Fixed lookup of parent directory IDs during graph generation
  - Proper handling of nested directory structures
  - Consistent path normalization across all node types

- **Stale Data Prevention**: Enhanced data freshness mechanisms
  - Added `Cache-Control: no-cache, no-store, must-revalidate` meta tags
  - Implemented timestamp-based cache-busting for JSON files
  - Ensured visualization directory contains latest graph data
  - Prevents browser caching of outdated graph structures

### Files Modified
- `src/mcp_vector_search/visualization/index.html` - UI improvements, expand/collapse fixes, cache prevention
- `src/mcp_vector_search/cli/commands/visualize.py` - Parent directory path normalization and linking fixes

### Technical Details
- Zero new dependencies (uses existing D3.js v7)
- Enhanced force simulation with configurable parameters
- Improved node interaction state management
- Better visual design with optimized spacing and colors

## [0.5.0] - 2025-10-02

### Added
- **PHP Language Support**: Full PHP parser implementation
  - Class, interface, and trait detection
  - Method extraction (public, private, protected, static)
  - Magic methods (__construct, __get, __set, etc.)
  - PHPDoc comment extraction
  - Namespace and use statement handling
  - Laravel framework patterns (Controllers, Models, Eloquent)
  - Supported extensions: `.php`, `.phtml`

- **Ruby Language Support**: Full Ruby parser implementation
  - Module and class detection with namespace support (::)
  - Instance and class method extraction
  - Special method names (?, !)
  - Attribute macros (attr_accessor, attr_reader, attr_writer)
  - RDoc comment extraction (# and =begin...=end)
  - Rails framework patterns (ActiveRecord, Controllers)
  - Supported extensions: `.rb`, `.rake`, `.gemspec`

### Fixed
- **MCP Configuration Bug**: Install command now correctly creates `.mcp.json` in project root instead of trying to create `claude-code` directory
- **Configuration Format**: Added required `"type": "stdio"` field for Claude Code compatibility

### Enhanced
- **Language Support**: Now supports **7 languages** total (8 as of next release)
  - Python, JavaScript, TypeScript (existing)
  - Dart/Flutter (v0.4.15)
  - **PHP** (new)
  - **Ruby** (new)
  - Markdown/Text (fallback)

- **Cross-Language Search**: Semantic search now works across all 7 languages
- **Framework Support**: Added specialized support for Laravel (PHP) and Rails (Ruby)

### Technical Details
- Zero new dependencies (uses existing tree-sitter-language-pack)
- Tree-sitter AST parsing with regex fallback for both PHP and Ruby
- Performance: PHP ~2.5ms, Ruby ~4ms per file (sub-5ms target)
- 100% test coverage for new parsers
- Type safety maintained (mypy compliant)

## [0.4.15] - 2025-10-02

### Added
- **Dart Language Support**: Full Dart/Flutter parser implementation
  - Widget detection (StatelessWidget, StatefulWidget)
  - State class parsing (_WidgetNameState pattern)
  - Async function support (Future<T> async)
  - Dartdoc comment extraction (///)
  - Tree-sitter AST parsing with regex fallback
  - Supported extensions: `.dart`
  - 20+ code chunks extracted from comprehensive test files
  - Cross-language semantic search across all 5 languages

- **Enhanced Install Command**: Complete project setup workflow
  - Multi-tool MCP detection (Claude Code, Cursor, Windsurf, VS Code)
  - Interactive MCP configuration with tool selection
  - Rich progress indicators and status updates
  - Automatic indexing after setup (optional)
  - New options:
    - `--no-mcp`: Skip MCP configuration
    - `--no-index`: Skip automatic indexing
    - `--extensions`: Customize file extensions
    - `--mcp-tool`: Specify MCP tool directly

### Enhanced
- **Rich Help System**: Industry-standard CLI help patterns
  - Help panels organized by purpose (Core Operations, Customization, Advanced)
  - Comprehensive examples in all command help text
  - Next-step hints after operations complete
  - Error messages with clear recovery instructions
  - Progressive disclosure pattern (basic → advanced)
  - Emoji indicators for visual hierarchy
  - Follows patterns from git, npm, docker CLIs

- **Language Support**: Now supports 7 languages
  - Python, JavaScript, TypeScript (existing)
  - Dart/Flutter (v0.4.15)
  - PHP, Ruby (v0.5.0)
  - Text/Markdown (fallback)

### Technical Details
- Zero new dependencies (uses existing tree-sitter-language-pack)
- Type safety: 100% mypy compliance maintained
- Test coverage maintained across all new features
- Backward compatible: No breaking changes

## [0.4.14] - 2025-09-23

### Added
- Initial release structure

### Changed
- Version bump preparation

### Fixed
- Build system improvements

## [0.4.1] - 2025-08-18

### 🐛 Critical Bug Fixes
- **BREAKING FIX**: Fixed search functionality returning zero results for all queries
  - Corrected ChromaDB cosine distance to similarity conversion that was producing negative scores
  - Fixed adaptive threshold logic ignoring user-specified threshold values (especially 0.0)
  - Search now properly returns relevant results with accurate similarity percentages

### ✨ Improvements
- Enhanced debug logging for search operations and threshold calculations
- Improved similarity score clamping to ensure values stay within [0, 1] range
- Better CLI output formatting with proper similarity percentage display

### 🧪 Testing & Validation
- Validated search functionality with real-world codebase (claude-mpm project)
- Tested multi-language search across Python, JavaScript, and TypeScript files
- Confirmed performance with 7,723 indexed code chunks from 120 files
- Added comprehensive debugging documentation and analysis

### 📚 Documentation
- Added detailed debugging analysis in `docs/debugging/SEARCH_BUG_ANALYSIS.md`
- Documented ChromaDB distance behavior and similarity calculation methods
- Enhanced troubleshooting guides for search-related issues

### 🎯 Impact
This release fixes the core search functionality that was completely broken in v0.4.0, making MCP Vector Search fully functional and production-ready for real-world use cases.

## [4.0.3] - 2025-01-18

### Added
- Consolidated versioning and build system via comprehensive Makefile
- Unified version management through scripts/version_manager.py
- Automated release workflows with git integration
- Dry-run mode for safe testing of version changes
- **Connection Pooling**: 13.6% performance improvement with automatic connection reuse
- **Semi-Automatic Reindexing**: 5 strategies without daemon processes
  - Search-triggered auto-indexing (built-in)
  - Git hooks integration for development workflows
  - Scheduled tasks (cron/Windows tasks) for production
  - Manual checks via CLI commands
  - Periodic checker for long-running applications
- **Auto-Index CLI Commands**: Complete management of automatic reindexing
- **Performance Testing**: Comprehensive benchmarking and optimization
- **Production Features**: Error handling, monitoring, graceful degradation

### Fixed
- Import error in factory.py (EmbeddingFunction → CodeBERTEmbeddingFunction)
- CLI typer.Choice() AttributeError in auto_index.py
- Missing ConnectionPoolError exception for tests
- Default embedding model updated to valid 'sentence-transformers/all-MiniLM-L6-v2'
- **Critical Bug**: Incremental indexing was creating duplicate chunks
- **Metadata Consistency**: Improved tracking of indexed files

### Changed
- Deprecated old build scripts in favor of unified Makefile workflow
- Version management centralized through single interface
- Build process streamlined with color-coded output
- **Incremental Indexing**: Now properly removes old chunks before adding new ones
- **Search Engine**: Integrated with auto-indexing for seamless updates
- **Database Layer**: Added pooled database option for high-throughput scenarios
- **CLI Interface**: Added auto-index subcommand with comprehensive options

### Deprecated
- scripts/build.sh - Use `make` commands instead
- scripts/dev-build.py - Use `make version-*` commands
- scripts/publish.sh - Use `make publish`

## [Unreleased]

### Added
-

### Changed
-

### Fixed
-

## [0.0.3] - 2024-01-10

### Added
- 🎉 **Initial public alpha release**
- **CLI Interface**: Complete Typer-based command-line tool
  - `init` - Initialize projects for semantic search
  - `index` - Index codebase with smart chunking
  - `search` - Semantic search with similarity scoring
  - `watch` - Real-time file monitoring
  - `status` - Project statistics and health
  - `config` - Configuration management
- **Multi-language Support**: Python, JavaScript, TypeScript parsing
  - AST-aware parsing with Tree-sitter integration
  - Regex fallback for robust parsing
  - Extensible parser registry system
- **Semantic Search**: ChromaDB-powered vector search
  - Sentence transformer embeddings
  - Similarity scoring and ranking
  - Rich terminal output with syntax highlighting
- **Real-time Updates**: File watching with incremental indexing
  - Debounced file change detection
  - Efficient incremental updates
  - Project-aware configuration
- **Developer Experience**:
  - Zero-config project initialization
  - Rich terminal output with progress bars
  - Comprehensive error handling
  - Local-first privacy with on-device processing

### Technical Details
- **Architecture**: Modern async Python with type safety
- **Dependencies**: ChromaDB, Sentence Transformers, Typer, Rich
- **Parsing**: AST-aware with Tree-sitter + regex fallback
- **Database**: Vector database abstraction layer
- **Configuration**: Project-aware settings management
- **Performance**: Sub-second search, ~1000 files/minute indexing

### Documentation
- Comprehensive README with examples
- MIT License for open-source distribution
- Professional project structure
- Development workflow documentation

### Infrastructure
- PyPI package distribution
- GitHub repository with releases
- Pre-commit hooks for code quality
- UV-based dependency management

## [0.0.2] - Internal Testing

### Added
- Core indexing functionality
- Basic search capabilities
- Python parser implementation

### Fixed
- File handling edge cases
- Memory usage optimizations

*Note: This version was used for internal testing and was not publicly released.*

## [0.0.1] - Initial Prototype

### Added
- Basic project structure
- Proof of concept implementation
- Initial CLI framework

*Note: This version was a prototype and was not publicly released.*

---

## Release Notes

### v0.0.3 - "Alpha Launch" 🚀

This is the first public release of MCP Vector Search! We're excited to share this semantic code search tool with the developer community.

**What's Working:**
- ✅ Multi-language code parsing (Python, JS, TS)
- ✅ Semantic search with vector embeddings
- ✅ Real-time file watching and indexing
- ✅ Rich CLI interface with progress indicators
- ✅ Project-aware configuration

**Known Limitations:**
- Tree-sitter integration needs improvement (using regex fallback)
- Search relevance may need tuning for specific codebases
- Limited error handling for edge cases
- Minimal test coverage

**Getting Started:**
```bash
pip install mcp-vector-search
mcp-vector-search init
mcp-vector-search index
mcp-vector-search search "your query here"
```

**Feedback Welcome:**
This is an alpha release - we're actively seeking feedback on search quality, performance, and usability. Please [open an issue](https://github.com/bobmatnyc/mcp-vector-search/issues) or start a [discussion](https://github.com/bobmatnyc/mcp-vector-search/discussions)!

---

## Migration Guides

### Upgrading to Future Versions

*Migration guides will be added here as new versions are released with breaking changes.*

#### From v0.0.x to v0.1.0 (Planned)
- Configuration file format may change
- CLI command options may be refined
- Database schema may be updated (automatic migration planned)

#### From v0.x to v1.0.0 (Planned)
- Stable API guarantees will begin
- MCP server integration will be added
- Plugin system will be introduced

---

## Development History

### Project Milestones

**2024-01-10**: 🎉 First public alpha release (v0.0.3)
- Published to PyPI
- GitHub repository made public
- Documentation and development workflow established

**2024-01-08**: Internal testing phase
- Core functionality implemented
- Multi-language parsing working
- CLI interface polished

**2024-01-05**: Project inception
- Initial concept and architecture design
- Technology stack selection
- Development environment setup

### Key Decisions

**Technology Choices:**
- **Python 3.11+**: Modern async/await, type hints
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: High-quality embeddings
- **Typer**: Modern CLI framework
- **Rich**: Beautiful terminal output
- **UV**: Fast Python package management

**Architecture Decisions:**
- **Local-first**: Complete on-device processing for privacy
- **Async-first**: Non-blocking operations for performance
- **Extensible**: Plugin-ready parser registry system
- **Type-safe**: Comprehensive type hints throughout

**Development Practices:**
- **Three-stage workflow**: Development → Local deployment → PyPI
- **Quality gates**: Linting, formatting, type checking
- **Documentation-first**: Comprehensive docs from day one
- **Community-focused**: Open source with clear contribution guidelines

---

## Statistics

### Release Metrics

**v0.0.3 (Alpha)**:
- **Files**: 39 source files
- **Lines of Code**: 11,718+ lines
- **Languages Supported**: 3 (Python, JavaScript, TypeScript)
- **CLI Commands**: 6 main commands
- **Dependencies**: 100+ (including transitive)
- **Documentation**: 15+ documentation files

### Performance Benchmarks

**Indexing Performance** (typical Python project):
- **Speed**: ~1000 files/minute
- **Memory**: ~50MB baseline + ~1MB per 1000 chunks
- **Storage**: ~1KB per code chunk

**Search Performance**:
- **Latency**: <100ms for most queries
- **Accuracy**: Semantic similarity-based ranking
- **Throughput**: Multiple concurrent searches supported

---

## Future Roadmap

### Short-term (v0.0.x - v0.1.0)
- [ ] Improve Tree-sitter integration
- [ ] Enhanced search relevance tuning
- [ ] Additional language support (Go, Rust, Java)
- [ ] Comprehensive test suite
- [ ] Performance optimizations

### Medium-term (v0.1.0 - v1.0.0)
- [ ] MCP (Model Context Protocol) server implementation
- [ ] Advanced search modes (contextual, similar code)
- [ ] Plugin system for extensibility
- [ ] IDE integrations (VS Code, JetBrains)
- [ ] Team collaboration features

### Long-term (v1.0.0+)
- [ ] Distributed indexing for large codebases
- [ ] Machine learning-powered code understanding
- [ ] Integration with code review tools
- [ ] Enterprise features and support
- [ ] Cloud-hosted option

---

## Contributing to Changelog

When contributing changes, please update this changelog following these guidelines:

1. **Add entries to [Unreleased]** section
2. **Use appropriate categories**: Added, Changed, Deprecated, Removed, Fixed, Security
3. **Write clear, user-focused descriptions**
4. **Include breaking change warnings**
5. **Reference issues/PRs when relevant**

Example entry:
```markdown
### Added
- New `--parallel` flag for faster indexing (#123)

### Fixed
- Handle Unicode characters in file names (#124)

### Changed
- **BREAKING**: Configuration file format updated (see migration guide)
```

The changelog will be updated with each release to move items from [Unreleased] to the appropriate version section.

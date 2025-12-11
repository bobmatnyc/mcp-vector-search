# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.3] - 2025-12-11

### Fixed
- **ChromaDB Rust panic recovery**
  - Added resilient corruption recovery for ChromaDB Rust panic errors
  - Implemented SQLite integrity check and Rust panic recovery
  - Use BaseException to properly catch pyo3_runtime.PanicException
  - Improved database health checking with sync wrapper

### Added
- **Reset command improvements**
  - Registered reset command and updated error messages
  - Corrected database path for reset operations
  - Better guidance for users experiencing index corruption

### Changed
- **Chat mode improvements**
  - Increased max iterations from 10 to 25 for better complex query handling

## [0.16.1] - 2025-12-09

### Added
- **Structural Code Analysis project roadmap**
  - Created GitHub Project with 38 issues across 5 phases
  - Added milestones: v0.17.0 through v0.21.0
  - Full dependency tracking between issues
  - Roadmap view with start/target dates

- **Project documentation improvements**
  - Added `docs/projects/` directory for active project tracking
  - Created comprehensive project tracking doc for Structural Analysis
  - Added PR workflow guide with branch naming conventions
  - HyperDev December 2025 feature write-up

- **Optimized CLAUDE.md**
  - Reduced from 235 to 120 lines (49% reduction)
  - Added Active Projects section
  - Added quick reference tables
  - Streamlined for AI assistant consumption

### Documentation
- New: `docs/projects/structural-code-analysis.md` - Full project tracking
- New: `docs/projects/README.md` - Projects index
- New: `docs/development/pr-workflow-guide.md` - PR workflow
- New: `docs/internal/hyperdev-2025-12.md` - Feature write-up
- Updated: `CLAUDE.md` - Optimized AI instructions

## [0.16.0] - 2025-12-09

### Added
- **Agentic chat mode with search tools**
  - Dual-intent mode: automatically detects question vs find requests
  - `--think` flag for complex reasoning with advanced models
  - `--files` filter support for scoped chat

## [0.15.17] - 2025-12-08

### Fixed
- **Fixed TOML config writing for Codex platform**
  - Now requires py-mcp-installer>=0.1.4 which adds missing `tomli-w` dependency
  - Fixes "Failed to serialize config: TOML write support requires tomli-w" error
  - Added Python 3.9+ compatibility with `from __future__ import annotations`

## [0.15.16] - 2025-12-08

### Fixed
- **Cleaned up verbose traceback output during setup**
  - Suppressed noisy "already exists" tracebacks when reinstalling MCP servers
  - Errors now show clean, single-line messages instead of full stack traces
  - "Already exists" is treated as success (server is already configured)
  - Debug output available via `--verbose` flag for troubleshooting

## [0.15.15] - 2025-12-08

### Fixed
- **Fixed platform forcing bug in MCP installer**
  - Now requires py-mcp-installer>=0.1.3 which fixes platform detection when forcing specific platforms
  - Fixes "Platform not supported: claude_code" errors during `mcp-vector-search setup`
  - Added `detect_for_platform()` method to detect specific platforms instead of highest-confidence one
  - Enables setup to work correctly in multi-platform environments (Claude Code + Claude Desktop + Cursor)

## [0.15.14] - 2025-12-08

### Fixed
- **Fixed Claude Code CLI installation syntax error**
  - Now requires py-mcp-installer>=0.1.2 which fixes the CLI command building
  - Fixes "error: unknown option '--command'" during `mcp-vector-search setup`
  - Claude Code CLI uses positional arguments, not `--command`/`--arg` flags
  - Correct syntax: `claude mcp add <name> <command> [args...] -e KEY=val --scope project`

## [0.15.13] - 2025-12-08

### Fixed
- **Updated py-mcp-installer dependency to 0.1.1**
  - Now requires py-mcp-installer>=0.1.1 which includes the platform forcing fix
  - Fixes "Platform not supported: claude_code" error during `mcp-vector-search setup`
  - Users must upgrade to get the fix: `pipx upgrade mcp-vector-search`

## [0.15.12] - 2025-12-08

### Fixed
- **`--version` flag now works correctly**
  - Fixed "Error: Missing command" when running `mcp-vector-search --version`
  - Added `is_eager=True` callback for version flag to process before command parsing
  - The `-v` short form also works now

## [0.15.11] - 2025-12-08

### Fixed
- **MCP installer platform forcing bug**
  - Fixed error "Platform not supported: claude_code" when forcing a platform
  - Now correctly detects info for the specific forced platform
  - Previously failed when another platform had higher confidence
  - Added `detect_for_platform()` method to PlatformDetector

## [0.15.10] - 2025-12-08

### Added
- **`--think` flag for chat command**
  - Uses advanced models for complex queries (gpt-4o / claude-sonnet-4)
  - Better reasoning capabilities for architectural and design questions
  - Higher cost but more thorough analysis
  - Example: `mcp-vector-search chat "explain the authentication flow" --think`

## [0.15.9] - 2025-12-08

### Added
- **`--files` filter support for chat command**
  - Filter chat results by file glob patterns (e.g., `--files "*.py"`)
  - Works the same as the search command's `--files` option
  - Examples: `chat "how does validation work?" --files "src/*.py"`

## [0.15.8] - 2025-12-08

### Fixed
- **Graceful handling of missing files during search**
  - Changed noisy WARNING logs to silent DEBUG level for missing files
  - Files deleted since indexing no longer spam warnings
  - Added `file_missing` flag to SearchResult for optional filtering
  - Hint: Use `mcp-vector-search index --force` to refresh stale index

## [0.15.7] - 2025-12-08

### Fixed
- **Index command crash: "name 'project_root' is not defined"**
  - Fixed undefined variable reference in "Ready to Search" panel code
  - Changed `project_root` to `indexer.project_root`

## [0.15.6] - 2025-12-08

### Added
- **Chat command shown in "Ready to Search" panel** after indexing completes
  - Displays LLM configuration status (✓ OpenAI or ✓ OpenRouter when configured)
  - Shows "(requires API key)" hint when no LLM is configured
  - Helps users discover the chat feature immediately after setup

## [0.15.5] - 2025-12-08

### Fixed
- **Chat command fails with "Extra inputs are not permitted" error**
  - Added `openrouter_api_key` field to `ProjectConfig` Pydantic model
  - Config file can now properly store the API key without validation errors

## [0.15.4] - 2025-12-08

### Fixed
- **Platform detection now works when CLI is available but config doesn't exist yet**
  - Claude Code, Claude Desktop, and Cursor can now be detected and configured via CLI
  - Previously required existing config file, now works with just CLI installation
  - Enables first-time setup without manual config file creation

## [0.15.3] - 2025-12-08

### Fixed
- **py-mcp-installer dependency now available from PyPI** - Users can install mcp-vector-search directly via pip
  - Published py-mcp-installer v0.1.0 to PyPI
  - Fixed dependency resolution that previously required local vendor directory
  - Added version constraint `>=0.1.0` for compatibility

## [0.15.2] - 2025-12-08

### Changed
- **Setup command now always prompts for API key** with existing value shown as obfuscated default
  - Shows keys like `sk-or-...abc1234` (first 6 + last 4 chars)
  - Press Enter to keep existing value (no change)
  - Type `clear` or `delete` to remove key from config
  - Warns when environment variable takes precedence over config file
- Deprecated `--save-api-key` flag (now always interactive during setup)

### Added
- New `_obfuscate_api_key()` helper for consistent key display
- 19 new unit tests for API key prompt behavior

## [0.15.1] - 2025-12-08

### Added
- **Secure local API key storage** - Store OpenRouter API key in `.mcp-vector-search/config.json`
  - File permissions set to 0600 (owner read/write only)
  - Priority: Environment variable > Config file
  - Config directory already gitignored for security
- New `--save-api-key` flag for `setup` command to interactively save API key
- New `config_utils` module for secure configuration management
- API key storage user guide in `docs/guides/api-key-storage.md`

### Changed
- Chat command now checks both environment variable and config file for API key
- Setup command shows API key source (env var or config file) when found

## [0.15.0] - 2025-12-08

### Added
- **LLM-powered `chat` command** for intelligent code Q&A using OpenRouter API
  - Natural language questions about your codebase
  - Automatic multi-query search and result ranking
  - Configurable LLM model selection
  - Default model: claude-3-haiku (fast and cost-effective)
- OpenRouter API key setup guidance in `setup` command
- Enhanced main help text with chat command examples and API setup instructions
- Automatic detection and display of OpenRouter API key status during setup
- Clear instructions for obtaining and configuring OpenRouter API keys
- Chat command aliases for "did you mean" support (ask, qa, llm, gpt, etc.)
- LLM benchmark script for testing model performance/cost trade-offs
- Two-phase visualization layout with progressive disclosure
- Visualization startup performance instrumentation

### Changed
- Improved main CLI help text to highlight chat command and its requirements
- Setup command now checks for OpenRouter API key and provides setup guidance
- Enhanced user experience with clearer distinction between search and chat commands
- Default LLM changed to claude-3-haiku for 4x faster responses at lower cost
- Visualization cache-busting with no-cache headers for better development experience

### Fixed
- **Glob pattern matching** for `--files` filter now works correctly with patterns like `*.ts`
- LLM result identifier parsing handles filenames in parentheses gracefully

## [0.14.6] - 2025-12-04

### Added
- Interactive D3.js force-directed graph visualization for code relationships
- `--code-only` filter option for improved performance with large datasets
- Variable force layout algorithm that spreads connected nodes and clusters unconnected ones
- Increased click target sizes for better usability in graph interface
- Clickable node outlines with thicker strokes for easier interaction

### Fixed
- Path resolution for visualizer to use project-local storage correctly
- JavaScript template syntax errors caused by unescaped newlines (2 fixes)
- Caching bug where serve command didn't respect `--code-only` flag
- Force layout tuning to fit nodes better on screen without excessive spread

### Changed
- Enhanced project description to highlight visualization capabilities
- Added visualization-related keywords and classifiers to package metadata
- Tightened initial force layout for more compact and readable graphs

## [0.14.5] - 2025-11-XX

### Changed
- Version bump for MCP installation improvements

### Fixed
- MCP installation bug analysis and documentation
- MCP server installation configuration

## [0.14.4] - 2025-11-XX

### Fixed
- Corrected MCP server installation configuration
- Automatically force-update .mcp.json when Claude CLI registration fails

## [0.14.3] - 2025-11-XX

### Changed
- Previous version baseline

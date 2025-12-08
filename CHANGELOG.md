# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

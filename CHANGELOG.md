# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

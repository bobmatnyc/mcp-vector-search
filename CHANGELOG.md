# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.9] - 2026-01-28

### Fixed
- **Bus Error Prevention** - Multi-layered defense against ChromaDB HNSW index corruption
  - Added binary validation to detect corrupted index files before loading
  - Subprocess isolation layer to prevent parent process crashes
  - Improved initialization order to reduce corruption risk
  - 13 new tests for bus error protection and recovery scenarios

## [1.2.2] - 2026-01-23

### Changed
- Patch release - maintenance update

## [1.2.1] - 2026-01-21

### Changed
- Patch release - maintenance update

## [1.2.0] - 2026-01-21

### Added
- **Dead Code Analysis** (`mcp-vector-search analyze dead-code`)
  - Entry point detection (main blocks, CLI commands, routes, tests, exports)
  - AST-based reachability analysis from entry points
  - Confidence levels (HIGH/MEDIUM/LOW) to reduce false positives
  - Output formats: Console, JSON, SARIF, Markdown
  - CI/CD integration with `--fail-on-dead` flag
  - Custom entry points via `--entry-point` flag
  - File exclusions via `--exclude` patterns

### Changed
- Refactored `analyze` command to use subcommands
  - `mcp-vector-search analyze complexity` (previously `analyze`)
  - `mcp-vector-search analyze dead-code` (new)

## [1.1.19] - 2026-01-08

### Fixed
- **CI Pipeline Improvements** - Better reliability and error handling
  - Install package before running performance tests
  - Skip existing PyPI uploads (don't fail if version already published)

## [1.1.18] - 2026-01-08

### Fixed
- **CI Pipeline Reliability** - Multiple CI/CD fixes for better reliability
  - Don't fail build on Codecov rate limits (non-critical service)
  - Create venv for integration tests (PEP 668 compliance)
  - Make init command non-interactive in integration tests
  - Correct search command argument order in integration tests

## [1.1.17] - 2026-01-07

### Fixed
- **Visualization Node Convergence Bug** - Fixed missing link types in D3 force simulation
  - Added `subproject_containment` and `dependency` link types to D3 visualization
  - Previously missing link types caused nodes to not converge to stable positions
  - Force simulation now properly recognizes all relationship types
  - Fixes erratic node movement and improves graph stability

### CI/CD
- **Streamlined CI Pipeline** - Reduced CI complexity and removed ineffective checks
  - Simplified test matrix to ubuntu-latest + Python 3.11 only
  - Removed broken documentation check job
  - Removed ineffective security job (used `|| true` making it always pass)
  - Made performance benchmarks release-only to reduce CI load
  - Removed duplicate pytest.ini (consolidated to pyproject.toml config)

## [1.1.16] - 2026-01-07

### Performance
- **Async Relationship Computation During Startup** - Non-blocking relationship processing
  - Changed default indexing behavior to mark relationships for background computation instead of blocking
  - Indexing now completes immediately without waiting for relationship computation
  - Relationships computed asynchronously during background processing or on-demand
  - Expected 2-5x faster initial indexing completion for large codebases

### Added
- **On-Demand Relationship Computation** - New command for manual relationship processing
  - New `mcp-vector-search index relationships` command for on-demand computation
  - `--background` flag for non-blocking relationship computation
  - `--relationships-only` mode in background indexer for targeted computation
  - Separate progress tracking for indexing vs relationship computation
- **CLI Alias** - Better UX for common commands
  - Added "ask" as alias for "chat" command

### Fixed
- Removed unused variable assignments in relationship computation code

## [1.1.15] - 2025-12-30

### Fixed
- **MCP Setup Command** - Removed hardcoded project path from MCP registration
  - Setup command was storing absolute project paths in ~/.claude.json causing stale configurations
  - MCP server now correctly resolves project path dynamically at runtime
  - Prevents "project not found" errors when moving or sharing configurations

## [1.1.14] - 2025-12-20

### Added
- **Background Indexing Mode** (#69) - Non-blocking indexing for large codebases
  - `--background/-bg` flag to run indexing as detached process
  - `status` subcommand to display real-time progress with Rich table
  - `cancel` subcommand to terminate running background process
  - Atomic progress file writes (crash-safe)
  - Cross-platform support (Unix: start_new_session, Windows: DETACHED_PROCESS)
  - SIGTERM/SIGINT handling for graceful shutdown
  - Concurrent indexing prevention with stale file detection

### Performance
- **Async Parallel Relationship Computation** (#68) - Concurrent semantic link processing
  - Parallel processing of semantic relationships using asyncio
  - Configurable concurrency limit via `max_concurrent_queries` parameter (default: 50)
  - Uses semaphore to prevent database/system resource exhaustion
  - Expected 3-5x faster relationship computation on typical 8-core machines
  - Graceful exception handling - individual chunk failures don't break computation
  - Backward compatible - default parameter works for all existing code
- **Multiprocess File Parsing** (#61) - Parallel parsing across CPU cores
  - Uses Python's ProcessPoolExecutor for CPU-bound tree-sitter parsing
  - Automatically uses 75% of CPU cores (capped at 8 workers)
  - Expected 4-8x faster parsing on multi-core systems
  - Graceful fallback to single-process for debugging (`use_multiprocessing=False`)
  - No change to public API - enabled by default
- **Batch Embedding Generation** (#59) - Significant indexing performance improvement
  - Accumulates chunks from multiple files before database insertion
  - Single database transaction per batch (default: 10 files) instead of per-file
  - Enables efficient batch embedding generation (32-64 embeddings at once)
  - Expected 2-4x faster indexing for typical projects
  - Better GPU/CPU utilization with larger embedding batches

## [0.21.3] - 2025-12-13

### Added
- **Historical Trend Tracking** - Track codebase metrics over time
  - Daily metric snapshots stored in `.mcp-vector-search/trends.json`
  - One entry per day (updates existing entry on reindex)
  - Tracks files, chunks, lines, complexity, health score, and code smells
  - D3.js line charts in Trends visualization page
- **Smart Dual Chunking** - Improved code parsing strategy
  - Class skeleton chunks preserve class structure with method signatures
  - Separate method chunks contain full implementation
  - Better semantic search for both class overview and method details
- **Visualizer Node Selection Highlighting**
  - Selected nodes glow with persistent orange highlight
  - Automatic path expansion when clicking collapsed nodes
  - Clear visual indication of currently viewed content
- **Markdown Report Format** - New `--output-format markdown` option for `analyze` command
- **Development Launcher Script** - `mcp-vector-search-dev` for development environments

### Fixed
- **Class/Function Parent Linking** - Classes and functions now correctly link to file nodes
  - Previously linked to imports chunk, causing wrong content display
  - Clicking files now shows actual file children, not imports
- **Visualizer Duplicate Links** - Removed duplicate directory hierarchy links
- **Dotfile Scanning** - Skip `.venv` and other dotfiles by default during indexing
- **JavaScript Template Syntax** - Fixed escaped backticks in visualizer templates

## [0.21.2] - 2025-12-13

### Fixed
- **JavaScript template literal escaping** - Removed unnecessary backslash escaping in visualizer JavaScript templates
  - Cleaned up template literal syntax in dependency analysis display functions
  - Fixed escaped dollar signs and backticks that were causing unnecessary verbosity

## [0.21.1] - 2025-12-13

### Added
- **Static Code Analysis Pages** - Four new analysis reports in visualization tool
  - **Complexity Report** - Summary stats, grade distribution chart, sortable hotspots table
  - **Code Smells** - Detects Long Method, High Complexity, Deep Nesting, God Class with severity badges
  - **Dependencies** - File dependency graph with circular dependency detection and warnings
  - **Trends** - Metrics snapshot with Code Health Score, complexity and size distributions

### Fixed
- **Node sizing algorithm** - Now uses actual line count instead of content-based estimates
  - 330-line functions correctly display larger than 7-line functions
  - Collapsed file+chunk nodes use chunk's actual line count
  - Logarithmic scaling for better visual distribution
- **Complexity stroke colors** - High complexity nodes (≥10) now have red outlines, moderate (≥5) orange

### Changed
- Visualization node sizing uses absolute thresholds instead of percentile-based scaling
- File nodes sized by total lines of code rather than chunk count

## [0.21.0] - 2025-12-12

### Added
- **Visualization Tool** - Interactive D3.js tree visualization for codebase exploration
  - Hierarchical view of directories, files, and code chunks
  - Lazy-loaded caller relationships for performance
  - Complexity-based coloring and sizing
  - Click-to-view source code with syntax highlighting

## [0.20.0] - 2025-12-11

### Added
- **Visualization Export (Phase 4)** - Complete HTML report generation and metrics export
  - **JSON Export Schema** (#28) - 13 Pydantic models for structured analysis data serialization
  - **JSON Exporter** (#29) - Export analysis results to JSON with full metrics and code smell data
  - **HTML Standalone Report Generator** (#30) - Self-contained HTML reports with embedded visualization
  - **Halstead Metrics Collector** (#31) - Software science metrics (volume, difficulty, effort, bugs)
  - **Technical Debt Estimation** (#32) - SQALE-based debt calculation with time-to-fix estimates
  - **CLI Metrics Display** (#33) - `status --metrics` command for comprehensive project metrics

### Features
- Self-contained HTML reports with no external dependencies
- Embedded D3.js force-directed graph visualization
- Interactive code navigation with syntax highlighting
- Export to JSON for custom tooling integration
- Halstead complexity metrics with scientifically-derived bug estimation
- Technical debt quantification in person-hours
- Comprehensive metrics dashboard in CLI

### Technical Details
- Complete JSON schema with backward compatibility
- Standalone HTML with inline CSS, JavaScript, and data
- Pydantic validation for all exported data
- Full test coverage for all Phase 4 components
- SQALE methodology for technical debt calculation

## [0.19.0] - 2025-12-11

### Added
- **Cross-File Analysis (Phase 3)** - Complete dependency and coupling analysis suite
  - **Efferent Coupling Collector** (#20) - Measures outgoing dependencies (Ce) from each module
  - **Afferent Coupling Collector** (#21) - Measures incoming dependencies (Ca) to each module
  - **Instability Index Calculator** (#22) - Computes I = Ce/(Ce+Ca) with A-F grading system
  - **Circular Dependency Detection** (#23) - DFS-based cycle detection with path visualization
  - **SQLite Metrics Store** (#24) - Persistent storage for metrics with git commit metadata
  - **Trend Tracking** (#25) - Regression detection with configurable thresholds (default 5%)
  - **LCOM4 Cohesion Metric** (#26) - Lack of Cohesion of Methods calculator for class quality

### Technical Details
- Import graph builder for dependency analysis
- Configurable instability thresholds (stable: I<0.3, unstable: I>0.7)
- Git integration for historical trend analysis
- Grade-based quality gates for coupling metrics
- Full test coverage for all Phase 3 collectors

## [0.18.0] - 2025-12-11

### Added
- **Code Smell Detection** (`mcp-vector-search analyze`)
  - Long Method detection (lines > 50 or cognitive complexity > 15)
  - Deep Nesting detection (max nesting depth > 4)
  - Long Parameter List detection (parameters > 5)
  - God Class detection (methods > 20 and lines > 500)
  - Complex Method detection (cyclomatic complexity > 10)
  - Configurable thresholds via ThresholdConfig
  - `--fail-on-smell` flag for CI/CD quality gates

- **SARIF Output Format**
  - Full SARIF 2.1.0 support for CI/CD integration
  - `--sarif` flag to output analysis results in SARIF format
  - Compatible with GitHub Code Scanning, Azure DevOps, and other tools
  - Includes rule definitions, locations, and severity levels

- **Exit Code Support for CI/CD**
  - Exit code 1 when quality gate fails (smells detected with --fail-on-smell)
  - Exit code 0 on success
  - Proper propagation through CLI wrapper chain

- **Diff-Aware Analysis** (`--changed-only`, `--baseline`)
  - Git integration for analyzing only changed files
  - `--changed-only` to analyze uncommitted changes
  - `--baseline <branch>` to compare against a specific branch
  - Fallback strategy: main → master → develop → HEAD~1
  - Reduces analysis time in large codebases

- **Baseline Comparison**
  - Save metric snapshots: `--save-baseline <name>`
  - Compare against baselines: `--compare-baseline <name>`
  - List saved baselines: `--list-baselines`
  - Delete baselines: `--delete-baseline <name>`
  - Regression/improvement tracking with configurable threshold (default 5%)
  - Rich console output showing changes per file

### Technical Details
- 43 new unit tests for baseline functionality
- 40 new unit tests for git integration
- Exit code propagation fix in didyoumean.py CLI wrapper
- GitManager class for robust git operations
- BaselineManager and BaselineComparator classes

## [0.17.0] - 2024-12-11

### Added
- **Structural Code Analysis Module** (`src/mcp_vector_search/analysis/`)
  - New analysis module with metric dataclasses and collector interfaces
  - Multi-language support for Python, JavaScript, and TypeScript via TreeSitter
  - Designed for <10ms overhead per 1000 LOC

- **Five Complexity Metric Collectors**
  - **Cognitive Complexity**: Measures code understandability and mental burden
  - **Cyclomatic Complexity**: Counts independent execution paths through code
  - **Nesting Depth**: Tracks maximum depth of nested control structures
  - **Parameter Count**: Analyzes function parameter complexity
  - **Method Count**: Enumerates class methods for complexity assessment

- **New `analyze` CLI Command** (`mcp-vector-search analyze`)
  - `--quick` mode for fast analysis (cognitive + cyclomatic complexity only)
  - `--language` filter to analyze specific languages (python, javascript, typescript)
  - `--path` filter to analyze specific directories or files
  - `--top N` option to show top complexity hotspots
  - `--json` output format for programmatic integration

- **Rich Console Reporter**
  - Project-wide summary statistics with file and function counts
  - Complexity grade distribution (A-F) with visual breakdown
  - Top complexity hotspots ranked by severity
  - Actionable recommendations for code improvements
  - Color-coded output with severity indicators

- **ChromaDB Metadata Extension**
  - Extended chunk metadata schema with structural metrics
  - Configurable complexity thresholds for grading
  - Automatic metrics storage during indexing
  - Threshold-based quality gates

### Technical Details
- 14 new unit tests for analyze command with 100% coverage
- Multi-language TreeSitter integration for accurate parsing
- Efficient collector pipeline with minimal performance impact
- Seamless integration with existing indexer workflow

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

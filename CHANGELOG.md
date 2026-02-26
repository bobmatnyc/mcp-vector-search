# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.28] - 2026-02-26

### Fixed

- **MCP Installer List Servers Traceback** — Resolved three bugs in post-install cleanup and reporting during MCP setup
  - Fixed traceback that occurred during list_servers operation when running `mcp-vector-search setup`
  - Improved post-install cleanup handling to prevent cascading failures
  - Enhanced error reporting during installer initialization

## [3.0.27] - 2026-02-26

### Performance

- **Skip Blame Default Fixed** — Resolved accidental enablement of blame during init/setup that added 50-100ms per file overhead
  - Root cause: blame feature was incorrectly enabled by default instead of respecting the skip_blame flag
  - Removed unintended blame initialization during project setup
  - Projects will now skip blame analysis unless explicitly enabled, improving initial indexing performance

## [3.0.26] - 2026-02-26

### Fixed

- **Stale LanceDB Table Entries on Fresh Projects** — Resolved "Table 'chunks' was not found" error when running on a project that has never been indexed
  - Root cause: stale entry detection attempted to query a table that does not yet exist
  - Added existence check before querying LanceDB tables during init
  - Added graceful abort on backend init failure to prevent cascading errors

### Added

- **E2E Tests for CLI Entry Points** — 25 new end-to-end tests covering `init`, `setup`, and `index` CLI commands
  - Tests cover happy path, error handling, and edge cases for each entry point
  - Validates CLI output, exit codes, and side effects

## [3.0.25] - 2026-02-26

### Fixed

- **LanceDB SEGV Crash on Large Datasets (Critical, Linux)** — Eliminated signal 139 (SEGV) crashes when indexing repositories with 31k+ files
  - Root cause: `delete_files_batch()` constructed a single SQL `OR` expression with up to 31,985 clauses, overflowing the stack inside DataFusion's recursive-descent parser running in a Rust tokio worker thread
  - Fix 1 (critical): Rewrote delete expressions as batched `IN` clauses of 500 files each — replaces O(n)-deep OR chains with flat membership tests; each batch wrapped in `try/except` for resilience; SQL-safe escaping for single quotes in paths
  - Fix 2: Added `gc.collect()` every 1,000 files during Phase 1 chunking to prevent PyArrow buffer accumulation (LanceDB issue #2512)
  - Fix 3: Added row-count guard for `compact_files()` on Linux — skips compaction when table exceeds 100k rows to prevent arrow offset overflow in background thread (lance issue #3330)
  - Fix 4: Applied same batched `IN`-clause delete pattern to `vectors_backend.py` for bulk vector cleanup consistency
  - Workaround for systemd environments: set `RUST_MIN_STACK=8388608` (8 MB) in the service unit file if deep OR expressions cannot be avoided

## [3.0.24] - 2026-02-26

### Fixed

- **LanceDB SIGBUS Crash on macOS (Critical)** — Extended comprehensive platform guards to all remaining LanceDB compaction operations that could trigger SIGBUS crashes due to memory-mapped file conflicts between PyTorch MPS and LanceDB Rust native code
  - Added platform guards to `_compact_table()` in `chunks_backend.py` and `vectors_backend.py`
  - Added platform guard to `optimize()` in `lancedb_backend.py`
  - Added platform guard to `remove_file()` in `indexer.py`
  - Added SIGBUS signal handler in `cli/main.py` for better crash diagnostics
  - Root cause: tokio worker thread crashes in `_lancedb.abi3.so` during fragment compaction while PyTorch holds model weights via Metal Performance Shaders memory mapping

## [3.0.23] - 2026-02-25

### Fixed

- **Incremental Index Change Detection** — Silent failures in `get_all_indexed_file_hashes()` that caused all files to appear changed, triggering a full reindex disguised as incremental with no warning
  - Added warnings when hash table is unavailable or fails to load
  - Added progress reporting every 1000 files during change detection
  - Reports actual counts ("N changed, M unchanged out of T total")

## [3.0.22] - 2026-02-25

### Added

- **Treemap/Sunburst Drill-Down with Code Complexity** — Visualization views now load code chunks with actual complexity/quality/smell data
  - New `/api/graph-code-chunks` endpoint returning function/class/method nodes with quality metrics
  - Treemap and sunburst auto-fetch code chunks before rendering
  - Filtered out text/doc chunks (86% of data) from hierarchy builders for these views
  - Treemap cells and sunburst arcs now show complexity grade coloring (A=green through F=red)

## [3.0.21] - 2026-02-25

### Added

- **`mvs index rebuild` Command** — Rebuild ANN vector index over existing embeddings without re-chunking or re-embedding; completes in under 1 second for 36K rows
- **IVF_SQ Vector Index** — Switched from IVF_PQ to scalar quantization for 384-dimensional embeddings
  - 100% recall@10 vs 97.5% with IVF_PQ
  - 4x index memory reduction (27 MB vs 108 MB for 74K chunks)
  - Faster index build (no codebook training required)

## [3.0.20] - 2026-02-24

### Added

#### Search Performance Optimizations

- **IVF-PQ Index Activation** — Automatic approximate nearest neighbor index creation for datasets >256 rows
  - Adaptive parameters: `num_partitions = clamp(sqrt(N), 16, 512)`, `num_sub_vectors = dim // 4`
  - Fixed dead code bug: `rebuild_index()` was never called, had hardcoded dimensions for wrong model
  - Applied to both VectorsBackend and LanceVectorDatabase search paths

- **Two-Stage Retrieval** — `nprobes=20` + `refine_factor=5` for optimal speed/quality tradeoff
  - Scan 20 IVF partitions, fetch 5× candidates, rerank with exact cosine similarity
  - Result: **4.9× speedup** (3.4ms vs 16.7ms median query time)

- **Contextual Chunking** — Metadata-enriched embeddings for better retrieval
  - Prepends file path, language, class/function name, imports to chunk text before embedding
  - Format: `File: core/search.py | Lang: python | Class: Engine | Fn: search | Uses: lancedb`
  - Stored content unchanged; only embedding input is enriched
  - Based on Anthropic research: 35-49% fewer retrieval failures
  - 23 new unit tests for context builder

- **CodeRankEmbed Model Option** — Optional code-specific embedding model
  - Registered `nomic-ai/CodeRankEmbed` (768d, 8K context, Apache-2.0)
  - Asymmetric instruction prefix support for instruction-tuned models
  - Default model unchanged (all-MiniLM-L6-v2)
  - Enable with: `mvs init --embedding-model nomic-ai/CodeRankEmbed`

#### Document Ontology (Knowledge Graph)

- **Document Nodes** — File-level document classification in the knowledge graph
  - New `Document` node type with `doc_category`, word count, section count
  - New `Topic` node type for future hierarchical taxonomy
  - Relationships: `CONTAINS_SECTION`, `RELATED_TO`, `DESCRIBES`, `HAS_TOPIC`

- **Automated Document Classification** — 4-pass rule-based classifier with 23 categories
  - Pass 1: File extension and well-known filenames
  - Pass 2: Exact filename stem matches
  - Pass 3: Path/directory-based patterns
  - Pass 4: Filename keyword patterns
  - Categories: api_doc, bugfix, changelog, configuration, contributing, deployment, design, example, faq, feature, guide, internal, migration, performance, project, readme, release_notes, report, research, script, setup, spec, test_doc, troubleshooting, tutorial, upgrade_guide
  - 0% "other" classification (down from 53%)

- **`kg ontology` CLI Command** — Browse document ontology as Rich tree
  - Grouped by category with emoji icons
  - `--category` filter, `--verbose` shows section headings
  - Shows tags, cross-references, word counts per document

- **`kg_ontology` MCP Tool** — Document ontology for agent consumption
  - Optional category filter parameter
  - Returns structured JSON

### Fixed

- **KG Build Stats Display** — "Documents" count now shown in build output table
- **LanceVectorDatabase Search Path** — Added missing `nprobes`/`refine_factor` ANN parameters (was running brute-force at 16.7ms instead of 3.4ms)

## [3.0.19] - 2026-02-24

### Added
- **IVF-PQ Vector Index** — Automatic approximate nearest neighbor index for datasets >256 rows with two-stage retrieval (`nprobes=20`, `refine_factor=5`), delivering 4.9× query speedup
- **Contextual Chunking** — Metadata-enriched embeddings prepend file path, language, class/function name, imports to chunk text (35-49% fewer retrieval failures per Anthropic research)
- **CodeRankEmbed Model Option** — Optional `nomic-ai/CodeRankEmbed` (768d, 8K context) with asymmetric instruction prefix support; default model unchanged

### Fixed
- LanceVectorDatabase search path missing ANN parameters (`nprobes`/`refine_factor`), was running brute-force at 16.7ms instead of 3.4ms

## [3.0.18] - 2026-02-24

### Added
- **GitHub Release + Homebrew** in all `make publish` targets (PyPI, GitHub Release, Homebrew tap update)
- **JS/TS Knowledge Graph** — Extract imports, calls, and inheritance from JavaScript/TypeScript AST for KG builder
- **Local Code Review MCP Tool** (`code_review`) — Pre-push validation analyzing staged changes for security, performance, quality, and style issues

## [3.0.17] - 2026-02-23

### Fixed
- Always show skill installation progress in `setup` command

## [3.0.16] - 2026-02-23

### Added
- Enhanced `mcp-vector-search-pr-mr-skill` v1.1.0 with research-backed improvements and branch modification focus

## [3.0.15] - 2026-02-23

### Added
- `mcp-vector-search-pr-mr-skill` with automatic installation during setup

### Fixed
- PR engine JSON parsing fix propagation

## [3.0.14] - 2026-02-23

### Fixed
- JSON parsing bug in `review_repository` tool

## [3.0.13] - 2026-02-23

### Added
- **Claude Sonnet 4.6 via Bedrock** as default LLM for chat and code review commands

### Fixed
- Updated `chat --model` help text to reflect Sonnet 4.6 Bedrock default

## [3.0.11] - 2026-02-23

### Fixed
- Corrected method name `reset_stale_processing_chunks` → `cleanup_stale_processing`

## [3.0.10] - 2026-02-23

### Fixed
- ChunksBackend path resolution in KG build — was using wrong directory, causing "Chunks backend not initialized" error

## [3.0.8] - 2026-02-23

### Fixed
- Migration forward-compatibility improvements
- Removed stale Homebrew note from docs

### Changed
- Removed agent-generated diagnostic scripts

## [3.0.7] - 2026-02-23

### Fixed
- Suppress debug output for file descriptor limit raise

## [3.0.4] - 2026-02-23

### Fixed
- Skip LanceDB deletes on macOS to prevent SIGBUS crash

## [3.0.3] - 2026-02-23

### Fixed
- Use `spawn` multiprocessing context on macOS to prevent SIGILL crash

## [3.0.2] - 2026-02-23

### Fixed
- Reindex defaults to incremental mode with clean Ctrl+C termination

## [3.0.0] - 2026-02-22

### Changed
- Major version bump — stable release of AI code review system, multi-language support, and review caching

## [2.10.4] - 2026-02-22

### Changed
- Version alignment release

## [2.10.3] - 2026-02-22

### Changed
- Thorough project cleanup — removed 131 agent artifacts and temp files
- Comprehensive AI code review feature documentation

## [2.10.2] - 2026-02-22

### Added
- **Multi-language PR review** — 12 languages with language-specific idioms, anti-patterns, and security patterns
- **Review caching system** — 5× speedup on repeat reviews with SQLite cache
- **6 review types** — security, architecture, performance, quality, testing, documentation
- **CI/CD integration** — GitHub Actions and GitLab CI examples

## [2.10.1] - 2026-02-22

### Fixed
- Ruff format auto-discovery code style fixes

## [2.10.0] - 2026-02-22

### Added

#### AI-Powered Code Review System

- **`analyze review [type]` Command** — Targeted code reviews with full codebase context
  - Six review types: `security`, `architecture`, `performance`, `quality`, `testing`, `documentation`
  - Semantic search finds relevant code patterns for each review type
  - Knowledge graph provides callers, dependencies, and impact analysis
  - LLM analysis with specialized prompts per review type
  - Output formats: `console`, `json`, `sarif`, `markdown`
  - Options: `--path` (scope), `--max-chunks` (context size), `--changed-only`, `--baseline`, `--types` (batch mode)
  - Cache control: `--no-cache`, `--clear-cache`

- **`analyze review-pr` Command** — Context-aware pull request review
  - Reviews PR/MR diffs with full codebase context
  - For each changed file, gathers: similar patterns (consistency), callers (impact), tests (coverage)
  - Auto-detects languages and applies language-specific standards
  - Custom instructions via `.mcp-vector-search/review-instructions.yaml`
  - Options: `--baseline`, `--head`, `--format github-json|console|json|markdown|sarif`
  - Structured output: inline comments with severity, blocking policy

- **Multi-Language Support (12 Languages)** — Language-specific idioms, anti-patterns, security patterns
  - Python, TypeScript, JavaScript, Java, C#, Ruby, Go, Rust, PHP, Swift, Kotlin, Scala
  - Each language has 5-8 idioms, 5-7 anti-patterns, 4-6 security patterns
  - Auto-detects language from file extensions
  - Language context automatically injected into review prompts

- **Auto-Discovery of Standards** — Reads existing config files to extract project standards
  - Python: `pyproject.toml`, `.flake8`, `mypy.ini`, `ruff.toml`
  - TypeScript/JavaScript: `tsconfig.json`, `.eslintrc.js`, `.eslintrc.json`
  - Ruby: `.rubocop.yml`
  - Java: `checkstyle.xml`, `pom.xml`
  - Go: `.golangci.yml`
  - Rust: `Cargo.toml`, `clippy.toml`
  - PHP: `.php-cs-fixer.php`, `phpstan.neon`
  - C#: `.editorconfig`, `stylecop.json`
  - Swift: `.swiftlint.yml`
  - Kotlin: `detekt.yml`
  - Scala: `scalafmt.conf`, `.scalafmt.conf`
  - Discovered standards merged with language profiles and custom instructions

- **Review Caching System** — 5x speedup on repeat reviews
  - SQLite cache at `.mcp-vector-search/reviews.db`
  - Cache key: (file_path, SHA256 content hash, review_type)
  - Typical 80-90% cache hit rate on stable codebases
  - Automatic invalidation when file content changes
  - Cache stats in console output: "cache: 24/30 hits (80%)"

- **Custom Review Instructions** — Team-specific standards and focus areas
  - YAML config at `.mcp-vector-search/review-instructions.yaml`
  - Sections: `language_standards`, `scope_standards`, `style_preferences`, `custom_review_focus`
  - Additive to auto-discovered standards (doesn't replace)
  - Per-language and per-path customization

- **MCP Tool Integration** — `review_pull_request` tool for IDE integration
  - Available in Claude Desktop, Cursor, and other MCP-compatible tools
  - Natural language: "Review the current PR"
  - Parameters: base_ref, head_ref, format

- **Chat Integration** — Natural language code review requests
  - "do a security review of the auth module"
  - Automatically uses `review_code` tool with vector search and KG context

- **CI/CD Integration** — GitHub Actions and GitLab CI examples
  - GitHub Actions workflow: PR reviews with inline comments and SARIF upload
  - GitLab CI pipeline: Merge request reviews with artifacts
  - Pre-commit hook example for local reviews
  - Blocking policies: exit code 1 on critical/high issues
  - SARIF output compatible with GitHub Security tab

#### Review Intelligence Features

- **Targeted Search Queries** — Specialized queries per review type
  - Security: "authentication login password", "sql query database", "user input validation"
  - Architecture: "class definition inheritance", "import module dependency"
  - Performance: "database query loop", "async await concurrent", "sort search algorithm"
  - Quality: "duplicate code", "long method function", "magic number hardcoded"
  - Testing: "test function coverage", "edge case boundary", "mock fixture"
  - Documentation: "docstring documentation", "TODO FIXME", "parameter return type"

- **Structured Findings** — Actionable feedback with metadata
  - Fields: title, description, severity (critical/high/medium/low/info), file_path, line numbers
  - Category tags: security, architecture, performance, quality, testing, documentation
  - CWE IDs for security vulnerabilities (e.g., CWE-89 for SQL injection)
  - Confidence scores (0.0-1.0)
  - Recommendations with code suggestions
  - Related files for cross-cutting concerns

- **GitHub JSON Format** — PR comment format for GitHub Actions
  - Summary comment with verdict, score, blocking issues count
  - Inline comments with file path, line number, severity
  - Blocking policy indicator (is_blocking: true/false)
  - Formatted for GitHub PR comment API

### Changed

- **Documentation Reorganization** — Major docs overhaul for code review feature
  - New: `docs/features/code-review.md` — Comprehensive feature documentation
  - Updated: `README.md` — Added "AI Code Review" section with quick examples
  - Enhanced: CI/CD integration guide with GitHub Actions and GitLab CI examples
  - Added: HyperDev article draft for external publication

### Performance

- **Review Caching** — 5x speedup on repeat reviews
  - First review: 45s (cold cache)
  - Second review: 8s (warm cache)
  - Partial changes: 3x speedup with 80% cache hit rate

- **Incremental Reviews** — Only review changed files with `--changed-only`
  - Git integration to detect uncommitted changes
  - Baseline comparison against branches (--baseline main)
  - Combined with cache for maximum speed

### Documentation

- **Code Review Documentation** (`docs/features/code-review.md`)
  - Architecture overview with ASCII diagrams
  - Quick start guide with 3-command workflow
  - All 6 review types explained with examples
  - PR review pipeline and context strategy
  - Custom instructions format and examples
  - Auto-discovery of standards (11 languages, 30+ config files)
  - Multi-language support table
  - Review caching explanation and performance data
  - CI/CD integration overview with examples
  - MCP tool reference
  - Chat integration examples
  - Output formats (console, JSON, SARIF, Markdown, GitHub JSON)
  - Performance characteristics and timing breakdown
  - Configuration reference
  - Troubleshooting guide
  - Best practices
  - Real-world examples

- **HyperDev Article** (`~/Duetto/repos/duetto-code-intelligence/docs/hyperdev-article-draft.md`)
  - Technical article for HyperDev Substack publication
  - Hook: Traditional tools see diffs, context-aware tools see relationships
  - Explains vector search + KG + LLM pipeline
  - Language-aware review examples (Python, TypeScript)
  - Custom standards and auto-discovery
  - Review caching for speed
  - Practical examples (security, PR, CI/CD)
  - Performance and cost analysis
  - Getting started guide

### Technical Details

- **Review Engine Architecture**
  - `ReviewEngine` class: Orchestrates vector search → KG → LLM pipeline
  - `PRReviewEngine` class: PR-specific review with context gathering
  - `ReviewCache` class: SQLite-based caching with content hashing
  - `LanguageProfile` dataclass: Language-specific standards storage
  - `InstructionsLoader` class: Custom YAML instructions + auto-discovery

- **Models and Schemas**
  - `ReviewType` enum: security, architecture, performance, quality, testing, documentation
  - `ReviewFinding`: Structured finding with severity, category, CWE ID
  - `ReviewResult`: Complete review output with findings and metadata
  - `PRReviewComment`: PR comment with file, line, severity, blocking flag
  - `PRReviewResult`: PR review output with verdict and score

- **LLM Integration**
  - Specialized prompts per review type
  - PR review prompt with language context injection
  - Structured JSON output parsing
  - Confidence scoring for findings
  - Model selection: GPT-4, Claude Sonnet, OpenRouter

- **CLI Integration**
  - `analyze review` subcommand with rich help text
  - `analyze review-pr` subcommand with PR-specific options
  - Batch review mode with `--types security,quality` or `--types all`
  - Multiple output formats with auto-detection
  - Verbose mode for debugging with `--verbose`

## [2.8.0] - 2026-02-22

### Added

#### Major Features
- **BM25 Hybrid Search with RRF** - Combined vector similarity with BM25 keyword matching using Reciprocal Rank Fusion for more accurate results
  - BM25 index builds during `index` step (reads from chunks, not vectors)
  - Cross-encoder reranking with ms-marco-MiniLM for precision improvement
  - Query expansion with automatic synonym expansion (137 code synonyms across 25 groups)
  - MMR diversity filtering using Maximal Marginal Relevance to reduce near-duplicate results
  - Sigmoid normalization for cross-encoder scores

- **Split Index/Embed Commands** - Decoupled indexing into independent `index` (CPU chunking) and `embed` (GPU embedding) commands for better resource utilization
  - `index` command handles file discovery, parsing, and BM25 index creation (CPU-only)
  - `embed` command generates vector embeddings (GPU-optimized)
  - Better resource utilization for distributed workflows

#### Enhancements
- **OpenRouter Default Provider** - OpenRouter is now the default/recommended LLM provider
- **Progress Bar Improvements** - Dynamic ETA on progress bars showing accurate time remaining based on actual processing rate
- **Database Auto-Initialization** - Database auto-initializes on health check (no more "not initialized" warnings)

### Fixed
- Fixed reindex backend re-initialization after atomic rebuild
- `setup` command no longer auto-indexes (use `--force` to reindex)
- Fixed duplicate PyTorch/initialization messages during reindex
- Fixed unawaited async parser calls in chat
- Fixed datetime JSON serialization in analyze_code
- Fixed schema column name mismatch in reset_all_to_pending

### Added

#### Major Features
- **`story` Command** — Generate development narratives from git history
  - Three-act narrative structure with markdown, JSON, and interactive HTML output
  - Contributor analysis and timeline visualization
  - Supports `--no-llm` mode for extraction-only output
  - Output formats: `--format markdown|json|html`, `--serve` for HTTP server
  - Model selection via `--model`, support for `--no-llm` extraction-only mode

- **Knowledge Graph (KuzuDB)** — Temporal knowledge graph with entity extraction and relationship mapping
  - Entities: CodeFile, Function, Class, Person, ProgrammingLanguage, ProgrammingFramework
  - Commands: `kg build`, `kg status`, `kg query`
  - 3 MCP tools for knowledge graph operations
  - Ontology navigation and semantic entity extraction
  - Person nodes populated from git contributors with Authored/Modified relationships
  - Frontmatter tag extraction for KG enrichment

- **Interactive D3.js Visualization** — `visualize` command with multiple views
  - 5+ visualization types: Treemap, Sunburst, Force Graph, Knowledge Graph, Heatmap
  - Three-axis visual encoding:
    - Fill color: complexity (green→yellow→red gradient)
    - Border dash: code smells (solid=clean, dashed=smells detected)
    - Blue intensity: quality score (darker=higher quality)
  - Progressive loading with drill-down/zoom capabilities
  - Dark/light mode support
  - Loading indicators for KG visualization

- **Chat Mode Enhancements** — Iterative refinement and advanced tools
  - Iterative refinement up to 30 queries (increased from 10/25)
  - Deep search tool for comprehensive code analysis
  - Knowledge graph query tool integration
  - Search history tracking
  - Synthesized responses at max iterations
  - `/wiki` command for ontology navigation in chat mode

- **CodeT5+ Code Embeddings** — Code-specific embeddings via `index-code` command
  - Uses Salesforce/codet5p-110m-embedding model
  - Separate code_vectors.lance index for code-only embeddings
  - Feature-flagged via `MCP_CODE_ENRICHMENT=true` environment variable
  - Unified visualization across all index types

- **Pipeline Parallelism** — Async pipeline for 37% faster indexing
  - Async pipeline with `asyncio.to_thread()` for GPU/CPU overlap
  - Parallel parsing and embedding generation
  - Significantly reduced indexing time on multi-core systems

- **Progress Tracking** — Rich progress bars with detailed metrics
  - Progress bars for indexing, KG build, and embedding phases
  - ETA calculation and chunks/sec throughput metrics
  - Real-time feedback during long operations

- **Language Support Expansions**
  - **C# Language Support** — Full Tree-sitter AST parsing for C# (.cs files)
    - Classes, interfaces, structs, enums, methods, constructors, properties
    - XML documentation comment extraction (`///`) as docstrings
    - C# attribute extraction (`[HttpGet]`, `[Authorize]`, etc.)
    - Cyclomatic complexity calculation for methods
    - Regex fallback when Tree-sitter unavailable
  - **Dart/Flutter Support** — Full Tree-sitter parser with Flutter patterns
    - Flutter widget detection (StatelessWidget, StatefulWidget)
    - Async/stream pattern recognition
    - Annotation extraction
    - Dartdoc comment parsing
  - **Now supports 13 languages**: C#, Dart, Go, HTML, Java, JavaScript, PHP, Python, Ruby, Rust, TypeScript, plus Markdown/Text

- **Git Blame Integration** — Author attribution per code chunk
  - Per-chunk git blame data for author tracking
  - Contributor analysis in knowledge graph

- **Monorepo Support** — Multi-root visualization and KG support
  - Handle multiple project roots in single repository
  - Unified visualization across monorepo structure

- **AWS Bedrock Support** — LLM integration via AWS Bedrock
  - AWS Bedrock integration for chat and story synthesis
  - Additional LLM backend option beyond OpenRouter

- **17 MCP Tools** — Expanded MCP server capabilities
  - Search tools (3): semantic search, code search, similar code
  - Project management (2): index, status
  - Analysis (6): complexity, dead code, dependencies, trends, code smells, metrics
  - Documentation (2): generate docs, update docs
  - Knowledge graph (3): kg build, kg query, kg status
  - Story generation (1): generate development narrative

#### Enhancements
- **CLI Short Flags** — 11 high-priority short flags added
  - Search: `-j` (json), `-L` (limit), `-s` (similarity), `-c` (context), `-e` (extension)
  - Story: `-n` (no-llm), `-s` (serve), `-m` (message), `-M` (model)
  - Status: `-j` (json)
  - Main: `-q` (quiet)
- **NLP Entity Extraction** — Semantic entity extraction from docstrings and comments
  - Enriches knowledge graph with semantic entities from documentation
- **Write Buffer Optimization** — LanceDB write buffer flush and transaction compaction
  - Improved indexing performance with better buffer management

### Changed
- **Default Backend** — LanceDB is now the default vector database
  - Serverless, file-based architecture
  - Better performance for large codebases
  - ChromaDB still supported via `MCP_VECTOR_SEARCH_BACKEND=chromadb`
- **Default Embedding Model** — Switched to MiniLM for local, GraphCodeBERT for GPU
  - Local: all-MiniLM-L6-v2 (fast, efficient)
  - GPU: GraphCodeBERT (code-optimized)
- **Apple Silicon Optimization** — MPS re-enabled on Apple Silicon with PyTorch version guard
  - 2-4x speedup on M4 Max and other Apple Silicon devices
  - Automatic MPS backend detection and configuration

### Fixed
- **Visualization Data Pipeline** — Fixed complexity=0 bug
  - LanceDB column name mismatch resolved: `complexity` vs `complexity_score`
  - Grade distribution now realistic: A:62K, B:6.3K, C:3.6K, D:1.5K, F:415
  - Proper complexity score calculation across all visualization views

- **Quality Metrics in Visualizer** — Added `compute_quality_metrics()` function
  - Proper quality_score calculation
  - Accurate smell_count tracking
  - Code smell detection across all views
  - Propagated quality_score, smell_count, smells to treemap/sunburst views

- **KG Loading Progress** — Added loading indicator to KG visualization
  - Improved UX during knowledge graph load operations

- **`--json` Output** — Fixed Rich formatting wrapping raw JSON output
  - Clean, parseable JSON output without Rich formatting interference

- **Kuzu Segfaults** — Isolated Kuzu operations in dedicated threads
  - Prevents segmentation faults during KG operations
  - Improved stability for knowledge graph features

- **Schema Mismatch** — Auto-reset on schema version mismatch with config preservation
  - Automatic schema migration on version changes
  - Preserves user configuration during schema updates

- **Duplicate Vectors** — Prevention of duplicate vectors in vectors.lance during indexing
  - Deduplication logic during indexing phase

- **Progressive Loading** — Fixed expansion state preservation and drill-down
  - Treemap/sunburst views correctly maintain expansion state
  - Drill-down navigation works reliably

- **Dead Code Analyzer** — Fixed false positives for class methods
  - Improved accuracy in dead code detection
  - Reduced false positive rate for class method analysis

- **Visualizer Cache Staleness** — Fixed cache not invalidating on data changes
  - Proper cache invalidation ensures fresh visualization data

## [2.2.4] - 2026-02-04

### Fixed
- Increase setup command auto-detect timeout from 2s to 5s to prevent missing languages in large projects (e.g., Rust with large target/ directories)
- Add verbose diagnostic output showing detected extensions count or timeout message during setup

## [2.2.3] - 2026-02-04

### Fixed
- **Language Tagging Accuracy** - Added parsers for Java, Go, and Rust
  - Added Java parser to fix `.java` files being tagged as "text" instead of "java"
  - Added Go parser to fix `.go` files being tagged as "text" instead of "go"
  - Added Rust parser to fix `.rs` files being tagged as "text" instead of "rust"
  - Now supports 11 languages: dart, go, html, java, javascript, php, python, ruby, rust, text, typescript

## [2.1.9] - 2026-02-03

### Fixed
- **force_include_paths Pattern Matching** - Improved default ignore pattern handling
  - `DEFAULT_IGNORE_PATTERNS` now properly blocks `force_include_paths` in `FileDiscovery`
  - Ensures force-included paths respect sensible defaults (node_modules, .git, etc.)
  - Prevents accidental indexing of large dependency directories

## [2.1.8] - 2026-02-02

### Changed
- Version bump for package maintenance

## [2.1.6] - 2026-02-02

### Added
- **LanceDB Default Backend** - Switched default vector database from ChromaDB to LanceDB
  - LanceDB now the recommended backend (set via `MCP_VECTOR_SEARCH_BACKEND=lancedb` or use default)
  - Better performance for large codebases (>100k chunks)
  - Simpler serverless architecture with file-based storage
  - ChromaDB still supported via `MCP_VECTOR_SEARCH_BACKEND=chromadb` environment variable
  - Migration command: `mcp-vector-search migrate db chromadb-to-lancedb`

### Performance
- **Apple Silicon M4 Max Optimizations** - 2-4x speedup on Apple Silicon
  - MPS (Metal Performance Shaders) backend support for PyTorch embeddings
  - GPU-aware batch size auto-detection (384-512 for Apple Silicon with high RAM)
  - Optimized for M4's 12 performance cores and 128GB unified memory
  - Environment variable: `MCP_VECTOR_SEARCH_MPS_BATCH_SIZE` for MPS-specific tuning
- **Write Buffering for LanceDB** - Faster bulk indexing
  - Accumulates records across batches before database writes
  - Reduces disk I/O and index update overhead
  - Configurable buffer size (default: 1000 chunks)
  - Expected 2-4x indexing speedup for large codebases

### Fixed
- **Reset Command Error Handling** - Better LanceDB support
  - Improved error handling for database resets
  - Properly handles both ChromaDB and LanceDB backends

## [2.1.2] - 2026-02-02

### Added
- **Force-Include Patterns** - Selective override of gitignore rules
  - New `force_include_patterns` configuration option
  - Glob pattern support with `**` for recursive matching (e.g., `repos/**/*.java`)
  - Force-include patterns override `.gitignore` rules, enabling selective indexing of gitignored directories
  - Use case: Index specific file types (e.g., `.java`, `.kt`) in gitignored directories (e.g., `repos/`)
  - CLI command: `mcp-vector-search config set force_include_patterns '["repos/**/*.java"]'`
  - Documentation: Added configuration examples and use cases to README

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

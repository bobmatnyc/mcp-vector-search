# Feature Overview

## 🚀 Core Features

### 🔍 Semantic Code Search
**Find code by meaning, not just keywords**

- **Natural Language Queries**: Search using plain English descriptions
- **Context-Aware**: Understands code structure and relationships
- **Multi-Language Support**: 11 languages (Python, JavaScript, TypeScript, C#, Java, Go, Rust, Dart, PHP, Ruby, HTML) with extensible architecture
- **Similarity Scoring**: Ranked results with confidence scores
- **Rich Output**: Syntax highlighting, file paths, line numbers

```bash
mcp-vector-search search "function that validates user input"
mcp-vector-search search "database connection with retry logic"
mcp-vector-search search "error handling for API calls"
```

### 📚 Intelligent Indexing
**AST-aware parsing with incremental updates**

- **AST Parsing**: Understands functions, classes, methods, and docstrings
- **Smart Chunking**: Preserves code context and relationships
- **Incremental Updates**: Only reprocesses changed files
- **Metadata Tracking**: Maintains file modification times and statistics
- **Configurable**: Customizable file extensions and parsing options

```bash
mcp-vector-search index                    # Index entire project
mcp-vector-search index --force            # Force full reindex
mcp-vector-search index /path/to/specific  # Index specific directory
```

### 👁️ Real-Time File Watching
**Automatic index updates as you code**

- **File System Monitoring**: Detects changes in real-time
- **Debounced Updates**: Prevents excessive reindexing during rapid changes
- **Selective Processing**: Only updates affected files
- **Background Operation**: Non-blocking, runs in background
- **Status Monitoring**: Track watching status and statistics

```bash
mcp-vector-search watch        # Start file watching
mcp-vector-search watch status # Check watching status
```

---

## ⚡ Performance Features

### IVF-PQ Index and Two-Stage Retrieval
**4.9x faster queries with adaptive index parameters**

After indexing more than 256 rows, the engine automatically builds an IVF-PQ (Inverted File + Product Quantization) index. Index parameters adapt to the data:

- `num_partitions = clamp(sqrt(N), 16, 512)` — scales with dataset size
- `num_sub_vectors = dim // 4` — adapts to the embedding dimension

Query execution uses two stages for accuracy without sacrificing speed:

1. **IVF scan** (`nprobes=20`): scans 20 nearest partitions instead of the full index
2. **Exact reranking** (`refine_factor=5`): fetches 5x the requested candidates, reranks with exact cosine similarity

Applied to both the LanceDB backend and the legacy VectorsBackend.

**Benchmark results**: 3.4ms vs 16.7ms median query time (4.9x speedup).

See [docs/performance/search-optimizations.md](../performance/search-optimizations.md) for full details.

### Contextual Chunking
**Metadata-enriched embeddings for improved retrieval accuracy**

Before embedding, each chunk receives a compact metadata header that captures its structural context:

```
File: core/search.py | Lang: python | Class: Engine | Fn: search | Uses: lancedb
```

The header is prepended to the text sent to the embedding model. The stored chunk content remains unchanged, so search result display is unaffected. Based on Anthropic research showing **35-49% fewer retrieval failures** versus embedding raw code text alone.

Applied at both embedding paths: `indexer.py` (two-phase pipeline) and `lancedb_backend.py` (direct path). The implementation is in `src/mcp_vector_search/core/context_builder.py`.

### 🔗 Connection Pooling
**13.6% performance improvement with zero configuration**

- **Automatic Connection Reuse**: Eliminates connection setup overhead
- **Configurable Pool Size**: Tune for your workload (default: 10 connections)
- **Health Monitoring**: Validates connections before use
- **Statistics Tracking**: Monitor pool efficiency and usage
- **Graceful Degradation**: Falls back gracefully if pool is exhausted

**Performance Results:**
- Sequential searches: 7.03ms → 6.07ms (13.6% faster)
- Connection reuse: 30 reused vs 8 created
- 100% pool hit rate after warmup

```python
# Automatically enabled for high-throughput scenarios
from mcp_vector_search.core.database import PooledChromaVectorDatabase

database = PooledChromaVectorDatabase(
    max_connections=10,    # Maximum connections in pool
    min_connections=2,     # Minimum warm connections
    max_idle_time=300.0,   # 5 minutes idle timeout
)
```

### 🔄 Semi-Automatic Reindexing
**Keep your index up-to-date without daemon processes**

Five different strategies to ensure your search index stays current:

#### 1. **Search-Triggered Auto-Indexing** ⚡
- **Built-in**: Zero setup required
- **Smart Timing**: Checks every 10 searches to avoid overhead
- **Threshold-Based**: Only auto-reindexes small numbers of files
- **Non-Blocking**: Never slows down search operations

#### 2. **Git Hooks Integration** 🔗
- **Development Workflow**: Triggers after commits, merges, checkouts
- **One-Time Setup**: Install hooks once, works automatically
- **Non-Blocking**: Never blocks Git operations
- **Cross-Platform**: Works on macOS, Linux, Windows

```bash
mcp-vector-search auto-index setup --method git-hooks
```

#### 3. **Scheduled Tasks** ⏰
- **Production Ready**: System-level cron jobs or Windows tasks
- **Configurable Interval**: Run every N minutes (default: 60)
- **Persistent**: Survives system reboots
- **Background**: No impact on interactive operations

```bash
mcp-vector-search auto-index setup --method scheduled --interval 60
```

#### 4. **Manual Checks** 🔧
- **On-Demand**: Run when needed via CLI
- **Flexible**: Check status or force reindexing
- **Configurable**: Set thresholds and limits
- **Informative**: Detailed status and staleness information

```bash
mcp-vector-search auto-index check --auto-reindex --max-files 10
mcp-vector-search auto-index status
```

#### 5. **Periodic Checker** 🔄
- **Long-Running Apps**: For applications that run continuously
- **Configurable Interval**: Default 1 hour, adjustable
- **In-Process**: No external dependencies
- **Efficient**: Only checks when interval expires

```python
from mcp_vector_search.core.auto_indexer import PeriodicIndexChecker

periodic_checker = PeriodicIndexChecker(
    auto_indexer=auto_indexer,
    check_interval=3600.0  # 1 hour
)

# In your application loop
await periodic_checker.maybe_check_and_reindex()
```

---

## 🛠️ Developer Experience

### 🎨 Rich CLI Interface
**Beautiful, informative command-line experience**

- **Syntax Highlighting**: Color-coded search results
- **Progress Indicators**: Visual feedback during operations
- **Structured Output**: Clear, organized information display
- **Error Handling**: Helpful error messages and suggestions
- **Completion Support**: Tab completion for commands and options

### ⚙️ Flexible Configuration
**Project-specific settings with sensible defaults**

- **Auto-Detection**: Automatically detects project structure
- **Customizable**: Override defaults for specific needs
- **Persistent**: Settings saved per project
- **Validation**: Ensures configuration correctness
- **Migration**: Automatic config updates between versions

```json
{
  "file_extensions": [".py", ".js", ".ts"],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "similarity_threshold": 0.7,
  "max_chunk_size": 1000,
  "auto_indexing": {
    "enabled": true,
    "threshold": 5,
    "staleness_minutes": 5
  }
}
```

### 📊 Comprehensive Monitoring
**Track performance and usage statistics**

- **Index Statistics**: File counts, chunk counts, index size
- **Performance Metrics**: Search times, indexing speed
- **Connection Pool Stats**: Hit rates, connection usage
- **Auto-Indexing Status**: Staleness info, reindex history
- **Health Checks**: System status and diagnostics

```bash
mcp-vector-search status --verbose
mcp-vector-search auto-index status
```

---

## 🏗️ Architecture Features

### CodeRankEmbed Embedding Model
**Optional code-specific embedding model for improved code search**

`nomic-ai/CodeRankEmbed` is available as an optional embedding model alongside the default `all-MiniLM-L6-v2`:

- **Dimensions**: 768d (vs 384d for the default model)
- **Context window**: 8K tokens
- **Optimized for**: Code retrieval tasks

The model uses asymmetric instruction prefixes: queries receive `query_prefix` and documents receive `document_prefix` at embedding time. This is transparent — existing models with empty prefixes behave identically to before.

Enable with:

```bash
mvs init --embedding-model nomic-ai/CodeRankEmbed
```

**Note**: CodeRankEmbed requires re-indexing your codebase after switching models, since the embedding dimension changes from 384d to 768d.

### Document Ontology
**File-level document classification for the knowledge graph**

The knowledge graph includes `Document` nodes for indexed files, each classified into one of 23 categories (API, guide, configuration, architecture, bugfix, etc.) using a 4-pass rule-based classifier:

1. File extension (e.g., `.md`, `.yaml`)
2. Filename pattern (e.g., `README`, `CHANGELOG`)
3. Directory path (e.g., `docs/guides/`, `tests/`)
4. Content keywords

Cross-references between documents are captured as `RELATED_TO` relationships (extracted from Markdown links).

```bash
# Browse the document ontology
mvs kg ontology

# Filter by category
mvs kg ontology --category guide

# Show file paths
mvs kg ontology --verbose
```

Output is a Rich tree grouped by category.

### 🔌 Extensible Design
**Plugin architecture for future growth**

- **Parser Registry**: Easy addition of new language parsers
- **Database Abstraction**: Support for different vector databases
- **Embedding Models**: Configurable embedding providers
- **Command System**: Modular CLI command structure
- **Event System**: Hooks for custom integrations

### 🛡️ Production Ready
**Robust error handling and reliability**

- **Graceful Degradation**: Continues working when components fail
- **Comprehensive Logging**: Detailed logs for troubleshooting
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Resource Management**: Proper cleanup and resource handling
- **Type Safety**: Full type annotations for reliability

### 🔒 Privacy First
**Complete local processing**

- **No Cloud Dependencies**: All processing happens locally
- **Private Data**: Your code never leaves your machine
- **Offline Capable**: Works without internet connection
- **Secure**: No external API calls or data transmission
- **Compliant**: Meets enterprise security requirements

---

## 🎯 Use Cases

### 👨‍💻 Individual Developers
- **Code Discovery**: Find existing implementations quickly
- **Refactoring**: Locate all usages of patterns or functions
- **Learning**: Explore unfamiliar codebases efficiently
- **Documentation**: Find examples and usage patterns

### 👥 Development Teams
- **Code Review**: Find similar code for consistency checks
- **Knowledge Sharing**: Discover team coding patterns
- **Onboarding**: Help new team members navigate codebase
- **Standards**: Enforce coding patterns and best practices

### 🏢 Enterprise
- **Code Auditing**: Find security patterns and vulnerabilities
- **Compliance**: Locate specific implementations for audits
- **Migration**: Find code that needs updating during migrations
- **Architecture**: Understand system structure and dependencies

---

## 🚀 Getting Started

### Quick Setup
```bash
# Install
pip install mcp-vector-search

# Initialize project
mcp-vector-search init

# Index codebase
mcp-vector-search index

# Setup auto-indexing (recommended)
mcp-vector-search auto-index setup --method all

# Start searching!
mcp-vector-search search "your query here"
```

### Performance Optimization
```bash
# For high-throughput scenarios, use connection pooling
# (automatically enabled when using PooledChromaVectorDatabase)

# Setup all auto-indexing strategies
mcp-vector-search auto-index setup --method all

# Monitor performance
mcp-vector-search status --verbose
mcp-vector-search auto-index status
```

---

## 📈 Roadmap

### Near Term (v0.1.0)
- Enhanced Tree-sitter integration
- Additional language support (Java, Go, Rust)
- Advanced search modes (contextual, similar code)
- Performance optimizations

### Medium Term (v1.0.0)
- MCP server implementation
- IDE extensions (VS Code, JetBrains)
- Team collaboration features
- Plugin system

### Long Term (v2.0+)
- AI-powered code suggestions
- Code quality analysis
- Integration with CI/CD pipelines
- Advanced analytics and insights

---

For detailed documentation, see **[CLAUDE.md](../CLAUDE.md)** - the main documentation index.

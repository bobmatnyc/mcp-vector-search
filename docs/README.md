# MCP Vector Search Documentation

📖 **Complete documentation for mcp-vector-search v2.x**

## Quick Links

- **[Full Documentation Index](index.md)** - Complete documentation navigation
- **[Getting Started](getting-started/README.md)** - Installation and first steps
- **[User Guides](guides/README.md)** - How-to guides
- **[Reference](reference/README.md)** - CLI commands and features
- **[Development](development/README.md)** - Contributing and architecture

## What's New in v2.x

### LanceDB Default Backend (v2.1+)

**LanceDB is now the default vector database** for better performance and stability:

- **Serverless**: No separate server process needed
- **Better Scaling**: Superior performance for codebases >100k chunks
- **More Stable**: Fewer corruption issues than ChromaDB
- **Faster Indexing**: 2-4x speedup with write buffering

ChromaDB is still supported. See **[LanceDB Backend Documentation](LANCEDB_BACKEND.md)** for details.

### Apple Silicon Optimizations (v2.1.6+)

**2-4x speedup on Apple Silicon** Macs:

- Metal Performance Shaders (MPS) GPU acceleration
- Intelligent batch sizing for M4 Max with high RAM
- Multi-core optimization for performance cores
- Automatic hardware detection

### Recent Improvements

- **Force-Include Patterns** (v2.1.9): Override gitignore for specific patterns
- **Write Buffering**: Faster bulk indexing with accumulated writes
- **GPU Auto-Detection**: Automatically uses best available hardware
- **Improved Error Handling**: Better recovery from database issues

## Documentation Structure

```
docs/
├── README.md                           # This file
├── index.md                            # Full documentation index
├── LANCEDB_BACKEND.md                  # LanceDB documentation
├── getting-started/                    # Installation & setup
├── guides/                             # How-to guides
├── reference/                          # CLI & API reference
├── development/                        # Contributing & architecture
├── architecture/                       # Design documentation
├── advanced/                           # Advanced topics
├── research/                           # Research notes & analysis
└── [other directories]                 # Additional resources
```

## Key Documentation

### For New Users

1. **[Installation Guide](getting-started/installation.md)** - Install mcp-vector-search
2. **[First Steps](getting-started/first-steps.md)** - Your first semantic search
3. **[CLI Usage Guide](guides/cli-usage.md)** - Command-line features

### For Developers

1. **[Contributing Guide](development/contributing.md)** - How to contribute
2. **[Architecture Guide](development/architecture.md)** - Technical deep dive
3. **[API Reference](development/api.md)** - Internal APIs

### Performance & Optimization

1. **[LanceDB Backend](LANCEDB_BACKEND.md)** - Default vector database
2. **[Performance Tuning](advanced/performance-tuning.md)** - Optimization tips
3. **[Performance Improvements](performance-improvements.md)** - Recent optimizations

### Research & Analysis

Research documentation is organized in [`research/`](research/):

- Performance analysis and benchmarks
- Feature investigations and prototypes
- Optimization research
- Bug investigations and fixes

Recent research:
- [M4 Max Performance Optimizations](research/m4-max-performance-optimizations-2026-02-02.md)
- [Indexing Performance Bottleneck Analysis](research/indexing-performance-bottleneck-analysis-2026-02-02.md)

## Migration from ChromaDB

If you have an existing ChromaDB installation:

```bash
# Migrate to LanceDB (recommended)
mcp-vector-search migrate db chromadb-to-lancedb

# Or continue using ChromaDB
export MCP_VECTOR_SEARCH_BACKEND=chromadb
```

See [LanceDB Backend Documentation](LANCEDB_BACKEND.md) for complete migration guide.

## Finding What You Need

### By Task

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](getting-started/README.md) |
| Search my code | [Searching Guide](guides/searching.md) |
| Configure my project | [Configuration Guide](getting-started/configuration.md) |
| Integrate with my IDE | [MCP Integration](guides/mcp-integration.md) |
| Understand how it works | [Architecture Overview](architecture/overview.md) |
| Optimize performance | [Performance Tuning](advanced/performance-tuning.md) |
| Troubleshoot issues | [Troubleshooting](advanced/troubleshooting.md) |
| Contribute code | [Contributing Guide](development/contributing.md) |

### By Experience Level

**Beginner**: [Getting Started](getting-started/README.md) → [Guides](guides/README.md)
**Intermediate**: [Reference](reference/README.md) → [Advanced Topics](advanced/README.md)
**Advanced**: [Development](development/README.md) → [Architecture](architecture/README.md)

## Getting Help

- **Questions**: [Troubleshooting Guide](advanced/troubleshooting.md)
- **Issues**: [GitHub Issues](https://github.com/bobmatnyc/mcp-vector-search/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bobmatnyc/mcp-vector-search/discussions)

---

**Last Updated**: 2026-02-03
**Documentation Version**: 2.1 (LanceDB Default)

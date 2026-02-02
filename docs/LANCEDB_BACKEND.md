# LanceDB Backend for MCP Vector Search

This document explains how to use the LanceDB backend as an alternative to ChromaDB.

## Overview

LanceDB provides a simpler, more stable vector database backend with these advantages:

- **Serverless architecture**: No separate server process needed
- **File-based storage**: Simple directory-based persistence
- **Better stability**: Fewer corruption issues compared to ChromaDB's HNSW indices
- **Apache Arrow**: Fast columnar operations for better performance at scale
- **Native vector search**: Built-in ANN (Approximate Nearest Neighbor) indices

## Installation

LanceDB is included as a dependency in `mcp-vector-search`. Install or update:

```bash
pip install -U mcp-vector-search
```

This will install `lancedb>=0.6.0` along with other dependencies.

## Using LanceDB Backend

### Option 1: Environment Variable (Recommended)

Set the backend via environment variable before running any commands:

```bash
export MCP_VECTOR_SEARCH_BACKEND=lancedb

# Now all commands use LanceDB
mcp-vector-search index
mcp-vector-search search "your query"
mcp-vector-search status
```

Add to your shell profile (`.bashrc`, `.zshrc`) for persistence:

```bash
echo 'export MCP_VECTOR_SEARCH_BACKEND=lancedb' >> ~/.zshrc
```

### Option 2: Programmatic (Python API)

```python
import os
os.environ["MCP_VECTOR_SEARCH_BACKEND"] = "lancedb"

from mcp_vector_search.core.factory import ComponentFactory

# This will create a LanceDB instance
components = await ComponentFactory.create_standard_components(
    project_root=project_path
)
```

## Migrating Existing Data

### From ChromaDB to LanceDB

If you have an existing ChromaDB database, migrate it to LanceDB:

```bash
mcp-vector-search migrate db chromadb-to-lancedb /path/to/project
```

This command:
1. Reads all chunks from your ChromaDB database
2. Creates a new LanceDB database with the same data
3. Preserves all metadata, embeddings, and relationships
4. Verifies the migration succeeded

After migration, set the environment variable:

```bash
export MCP_VECTOR_SEARCH_BACKEND=lancedb
```

### From LanceDB to ChromaDB

To migrate back to ChromaDB:

```bash
mcp-vector-search migrate db lancedb-to-chromadb /path/to/project
```

Then unset or change the environment variable:

```bash
unset MCP_VECTOR_SEARCH_BACKEND
# or
export MCP_VECTOR_SEARCH_BACKEND=chromadb
```

## Directory Structure

LanceDB stores data in a subdirectory of your index path:

```
.mcp-vector-search/
├── index/
│   ├── lancedb/              # LanceDB storage
│   │   └── code_search.lance # Table data
│   └── chroma.sqlite3        # ChromaDB storage (if using ChromaDB)
└── config.json
```

## Performance Comparison

| Operation | ChromaDB | LanceDB | Notes |
|-----------|----------|---------|-------|
| Index creation | Fast | Fast | Similar performance |
| Search (< 10k chunks) | Fast | Fast | Comparable |
| Search (> 100k chunks) | Moderate | Fast | LanceDB scales better |
| Memory usage | High | Lower | Arrow's columnar format |
| Corruption recovery | Manual | Automatic | LanceDB more resilient |

## API Compatibility

LanceDB backend implements the same `VectorDatabase` interface as ChromaDB:

```python
# All these methods work identically
await db.initialize()
await db.add_chunks(chunks)
results = await db.search(query, limit=10, filters={...})
await db.delete_by_file(file_path)
stats = await db.get_stats()
await db.reset()
await db.health_check()
await db.close()
```

## Limitations

Current limitations of the LanceDB backend:

1. **No connection pooling**: Unlike ChromaDB's `PooledChromaVectorDatabase`, LanceDB doesn't support connection pooling yet (not needed due to serverless architecture)
2. **Different distance metric**: LanceDB uses L2 distance by default, converted to similarity scores for compatibility
3. **Filter syntax**: Uses different SQL-like WHERE clauses (handled internally)

## Troubleshooting

### "lancedb not found" error

```bash
pip install lancedb>=0.6.0
```

### Migration verification failed

If migration verification fails:

```bash
# Reset and re-index from scratch
export MCP_VECTOR_SEARCH_BACKEND=lancedb
mcp-vector-search reset index --force
mcp-vector-search index
```

### Switching back to ChromaDB

```bash
unset MCP_VECTOR_SEARCH_BACKEND
# or
export MCP_VECTOR_SEARCH_BACKEND=chromadb
```

## Example Workflow

Complete example of setting up a new project with LanceDB:

```bash
# 1. Set backend
export MCP_VECTOR_SEARCH_BACKEND=lancedb

# 2. Initialize project
cd /path/to/your/project
mcp-vector-search init

# 3. Index codebase
mcp-vector-search index

# 4. Search
mcp-vector-search search "authentication logic"

# 5. Check status
mcp-vector-search status
```

## When to Use LanceDB

Use LanceDB if you:
- Experience ChromaDB corruption issues
- Have large codebases (> 100k chunks)
- Want simpler storage architecture
- Need better long-term stability
- Prefer serverless databases

Stick with ChromaDB if you:
- Already have a working ChromaDB setup
- Use connection pooling features
- Have small codebases (< 10k chunks)
- Want exact compatibility with older versions

## Technical Details

### Schema

LanceDB stores each chunk as a record with these fields:

- `id` (string): Unique chunk ID
- `vector` (float array): Embedding vector
- `content` (string): Code content
- `file_path` (string): Source file path
- `start_line` (int): Starting line number
- `end_line` (int): Ending line number
- `language` (string): Programming language
- `chunk_type` (string): Type (function, class, etc.)
- `function_name` (string): Function name (if applicable)
- `class_name` (string): Class name (if applicable)
- Plus all other CodeChunk metadata fields

### Search Implementation

LanceDB search process:

1. Generate query embedding using same model
2. Perform ANN search with L2 distance
3. Convert distance to similarity: `similarity ≈ 1 - (distance² / 2)`
4. Apply metadata filters using SQL WHERE clauses
5. Filter results by similarity threshold
6. Return SearchResult objects

### Caching

Both backends use the same LRU search cache:

- Configurable size via `MCP_VECTOR_SEARCH_CACHE_SIZE` (default: 100)
- Cache key includes query, limit, filters, and threshold
- Invalidated on any database modifications

## Future Enhancements

Planned improvements for LanceDB backend:

- [ ] Support for hybrid search (vector + keyword)
- [ ] Incremental indexing without full reindex
- [ ] Advanced ANN index tuning options
- [ ] Multi-vector search for hierarchical chunks
- [ ] Batch operations optimization

## References

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Apache Arrow](https://arrow.apache.org/)
- [MCP Vector Search GitHub](https://github.com/bobmatnyc/mcp-vector-search)

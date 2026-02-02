# Testing the LanceDB Backend

Quick reference for testing the new LanceDB backend implementation.

## Installation

First, install the lancedb dependency:

```bash
pip install lancedb>=0.6.0
```

Or reinstall the project to get all dependencies:

```bash
pip install -e .
```

## Manual Testing

### Test 1: Basic Operations

Run the manual test script:

```bash
python tests/manual/test_lancedb_backend.py
```

Expected output:
```
🧪 Testing LanceDB Backend

📁 Test directory: tests/manual/test_lancedb
📦 Loading embedding model...
✅ LanceDB initialized

📝 Test 1: Adding chunks...
✅ Added 3 chunks

📊 Test 2: Getting statistics...
  Total chunks: 3
  Indexed files: 2
  Backend: lancedb
✅ Stats correct

🔍 Test 3: Searching...
  Found X results
  Top result: hello_world (score: 0.XXX)
✅ Search works

🔍 Test 4: Searching with file filter...
  Found X results in async_test.py
  Result: fetch_data
✅ Filtered search works

📥 Test 5: Getting all chunks...
  Retrieved 3 chunks
✅ Get all chunks works

🗑️  Test 6: Deleting chunks by file...
  Deleted 2 chunks
  Remaining chunks: 1
✅ Delete by file works

🏥 Test 7: Health check...
  Database healthy: True
✅ Health check works

♻️  Test 8: Resetting database...
  Chunks after reset: 0
✅ Reset works

🎉 All tests passed!
🧹 Cleaned up test directory
```

### Test 2: CLI Integration

Test the LanceDB backend with actual CLI commands:

```bash
# Set environment variable
export MCP_VECTOR_SEARCH_BACKEND=lancedb

# Initialize a test project
cd /tmp
mkdir test-lancedb-project
cd test-lancedb-project
echo "def hello(): pass" > test.py

mcp-vector-search init

# Index the project
mcp-vector-search index

# Verify status shows LanceDB backend
mcp-vector-search status

# Search
mcp-vector-search search "hello function"

# Check stats
mcp-vector-search status
```

Expected status output should show:
```
Backend: lancedb
```

### Test 3: Migration

Test migration from ChromaDB to LanceDB:

```bash
# First, create a ChromaDB index
unset MCP_VECTOR_SEARCH_BACKEND
cd /tmp/test-project
mcp-vector-search init
mcp-vector-search index

# Migrate to LanceDB
mcp-vector-search migrate db chromadb-to-lancedb .

# Set environment and verify
export MCP_VECTOR_SEARCH_BACKEND=lancedb
mcp-vector-search search "your query"
mcp-vector-search status
```

### Test 4: Switching Backends

Test switching between backends:

```bash
cd /tmp/test-project

# Use ChromaDB
unset MCP_VECTOR_SEARCH_BACKEND
mcp-vector-search index
mcp-vector-search search "query"

# Switch to LanceDB
export MCP_VECTOR_SEARCH_BACKEND=lancedb
mcp-vector-search index
mcp-vector-search search "query"

# Both should work independently
```

## Unit Testing

Add unit tests for LanceDB (TODO):

```bash
pytest tests/unit/core/test_lancedb_backend.py -v
```

## Integration Testing

Test with MCP server (TODO):

```bash
# Start MCP server with LanceDB backend
export MCP_VECTOR_SEARCH_BACKEND=lancedb
mcp-vector-search-mcp

# In another terminal, test MCP client
# (add MCP client test commands here)
```

## Performance Testing

Compare performance between ChromaDB and LanceDB:

```bash
# Test ChromaDB
unset MCP_VECTOR_SEARCH_BACKEND
time mcp-vector-search index
time mcp-vector-search search "query" --limit 100

# Test LanceDB
export MCP_VECTOR_SEARCH_BACKEND=lancedb
time mcp-vector-search index
time mcp-vector-search search "query" --limit 100
```

## Known Issues

Track any issues found during testing:

1. **None yet** - Report issues as you find them

## Testing Checklist

- [ ] Manual test script passes
- [ ] CLI indexing works with LanceDB
- [ ] CLI search works with LanceDB
- [ ] Status command shows correct backend
- [ ] Migration ChromaDB → LanceDB works
- [ ] Migration LanceDB → ChromaDB works
- [ ] Can switch backends via env var
- [ ] Search results match ChromaDB quality
- [ ] Metadata filtering works
- [ ] File deletion works
- [ ] Database reset works
- [ ] Health check works
- [ ] MCP server works with LanceDB

## Files Created/Modified

### New Files:
- `src/mcp_vector_search/core/lancedb_backend.py` - LanceDB implementation
- `src/mcp_vector_search/cli/commands/migrate_db.py` - Migration CLI
- `tests/manual/test_lancedb_backend.py` - Manual test script
- `docs/LANCEDB_BACKEND.md` - User documentation
- `LANCEDB_TESTING.md` - This file

### Modified Files:
- `pyproject.toml` - Added lancedb>=0.6.0 dependency
- `src/mcp_vector_search/core/factory.py` - Added backend selection logic
- `src/mcp_vector_search/cli/commands/migrate.py` - Added migrate-db subcommand

## Next Steps

1. Run manual test script
2. Fix any issues found
3. Test CLI integration
4. Test migration commands
5. Add unit tests
6. Add integration tests
7. Update main documentation
8. Create PR

## Issues Encountered

Document any issues here as they're found during testing.

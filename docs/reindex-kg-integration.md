# Reindex KG Integration

## Summary

Added automatic knowledge graph rebuilding to `reindex -f` command for seamless full pipeline execution.

## What Changed

### Modified Files
- `src/mcp_vector_search/cli/commands/reindex.py`

### Changes Made

1. **Added imports** for KG subprocess approach:
   - `gc`, `json`, `shutil`, `subprocess`, `tempfile`, `threading`, `time`
   - `asdict` from `dataclasses`

2. **Added KG build step** to `_run_reindex()`:
   - Only runs when `fresh=True` (full reindex)
   - Executes after successful chunk + embed completion
   - Wrapped in try/except to prevent reindex failure if KG build fails
   - Shows helpful error message if KG build fails

3. **Added `_build_knowledge_graph()` helper function**:
   - Loads chunks from database
   - Serializes chunks to temp JSON file
   - Closes database and cleans up async resources
   - Spawns isolated subprocess to build KG (avoids thread conflicts with Kuzu)
   - Uses same subprocess approach as `kg build` command

## Flow

### Before (reindex -f)
```
1. chunk_files(fresh=True)   → Parse all files
2. embed_chunks(fresh=True)  → Generate all embeddings
✓ Done
```

### After (reindex -f)
```
1. chunk_files(fresh=True)   → Parse all files
2. embed_chunks(fresh=True)  → Generate all embeddings
3. build_knowledge_graph()   → Rebuild KG (NEW)
✓ Done
```

## Error Handling

- KG build failures don't crash the reindex command
- Shows warning message if KG build fails
- Suggests manual rebuild: `mcp-vector-search kg build --force`

## Technical Details

### Why Subprocess Approach?

The KG build uses Kuzu (Rust-based graph database) which has strict thread safety requirements:
- Cannot run in same process as LanceDB (creates background threads)
- Requires isolated subprocess to avoid segfaults
- Same approach used by `kg build` command

### Implementation Pattern

Follows the pattern from `kg.py` (`build_kg` command):
1. Load chunks in main process
2. Serialize to temp JSON file
3. Close database and cleanup async resources
4. Spawn subprocess with `_kg_subprocess.py`
5. Subprocess builds KG from JSON file
6. Cleanup temp file

## Testing

- ✅ Code compiles successfully
- ✅ Integration tests pass (no regressions)
- ✅ Pattern matches `kg build` command implementation

## Usage

### Full Reindex (with KG rebuild)
```bash
mcp-vector-search reindex -f
# or
mcp-vector-search reindex --fresh
```

### Incremental Reindex (no KG rebuild)
```bash
mcp-vector-search reindex --incremental
```

### Verbose Output
```bash
mcp-vector-search reindex -f --verbose
```

## Benefits

1. **Seamless workflow**: Full reindex now includes KG rebuild automatically
2. **Atomic operation**: One command for complete index + KG rebuild
3. **Safe defaults**: Only rebuilds KG on full reindex (not incremental)
4. **Graceful degradation**: KG build failure doesn't break reindex
5. **Clear feedback**: User notified of KG build progress and any errors

## Related Commands

- `mcp-vector-search kg build --force` - Manual KG rebuild
- `mcp-vector-search kg stats` - View KG statistics
- `mcp-vector-search kg status` - Detailed KG status

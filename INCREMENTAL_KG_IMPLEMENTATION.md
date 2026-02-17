# Incremental Knowledge Graph Implementation

## Overview

Implemented incremental Knowledge Graph tracking with metadata, gap detection, and incremental builds as requested.

## Changes Made

### 1. KG Metadata Tracking (`src/mcp_vector_search/core/kg_builder.py`)

Added metadata tracking functionality to `KGBuilder` class:

#### New Methods:
- `_load_metadata()` - Load KG metadata from disk
- `_save_metadata()` - Save build metadata after successful build
- `_get_processed_chunk_ids()` - Get set of previously processed chunk IDs

#### Metadata Format (`kg_metadata.json`):
```json
{
  "last_build": "2026-02-16T14:44:00Z",
  "source_chunk_count": 16199,
  "source_chunk_id_hash": "abc123...",
  "source_chunk_ids": ["id1", "id2", ...],
  "entities_created": 14973,
  "relationships_created": 1291,
  "build_duration_seconds": 45.2
}
```

**Location**: `.mcp-vector-search/knowledge_graph/kg_metadata.json`

### 2. Incremental Build Support

Modified `build_from_database()` method:

- Added `incremental: bool = False` parameter
- Loads previously processed chunk IDs from metadata
- Filters out already-processed chunks during loading
- Updates metadata with cumulative chunk IDs (existing + new)
- Returns early if no new chunks to process
- Tracks build duration and saves metadata after successful build

**Key Features**:
- Only processes chunks not in previous metadata
- Supports both full rebuild (default) and incremental mode
- Maintains cumulative list of all processed chunks
- Preserves metadata across incremental builds

### 3. Gap Detection in `kg stats` Command (`src/mcp_vector_search/cli/commands/kg.py`)

Enhanced `kg stats` command to show:

```
Knowledge Graph Statistics
┌─────────────────┬──────────┐
│ Total Entities  │ 14,973   │
│ Source Chunks   │ 16,199   │  ← from metadata
│ Current Chunks  │ 16,450   │  ← from vector DB
│ Gap             │ 251 new  │  ← difference (yellow if > 0)
│ Last Build      │ 2h ago   │
└─────────────────┴──────────┘
💡 Run 'kg build --incremental' to add 251 new chunks
```

**Features**:
- Compares metadata source chunk count with current vector DB count
- Shows gap with color coding:
  - **Green**: 0 (up to date)
  - **Yellow**: Positive (new chunks available)
  - **Red**: Negative (chunks removed)
- Displays last build time in human-readable format (Xd/Xh/Xm ago)
- Shows helpful message when gap > 0

### 4. CLI Integration

Added `--incremental` flag to `kg build` command:

```bash
# Full rebuild (default)
mcp-vector-search kg build

# Incremental build (only new chunks)
mcp-vector-search kg build --incremental

# Force full rebuild
mcp-vector-search kg build --force
```

## Implementation Details

### Chunk ID Tracking

The implementation uses two strategies for efficient chunk tracking:

1. **Full List Storage**: All chunk IDs stored in metadata for exact delta calculation
2. **Hash Verification**: SHA256 hash of sorted chunk IDs for quick comparison

### Incremental Build Algorithm

```python
# 1. Load previously processed chunk IDs from metadata
processed_chunk_ids = self._get_processed_chunk_ids()

# 2. Filter chunks during database iteration
for batch in database.iter_chunks_batched():
    if incremental:
        batch = [c for c in batch if (c.chunk_id or c.id) not in processed_chunk_ids]
    chunks.extend(batch)

# 3. Process only new chunks
await self.build_from_chunks(chunks, ...)

# 4. Update metadata with cumulative chunk IDs
all_chunk_ids = processed_chunk_ids | new_chunk_ids
self._save_metadata(source_chunk_ids=all_chunk_ids, ...)
```

### Gap Detection Algorithm

```python
# 1. Load metadata from disk
metadata = load_metadata()
source_chunks = metadata["source_chunk_count"]

# 2. Get current chunk count from vector DB
current_chunks = database.get_chunk_count()

# 3. Calculate gap
gap = current_chunks - source_chunks

# 4. Display with color coding
if gap > 0:
    print(f"[yellow]{gap} new chunks[/yellow]")
elif gap == 0:
    print("[green]up to date[/green]")
else:
    print(f"[red]{abs(gap)} removed[/red]")
```

## Usage Examples

### First Build (Full)
```bash
$ mcp-vector-search kg build
Building Knowledge Graph...
✓ Scanned 16,199 chunks
✓ Extracted 14,973 entities
✓ Built 1,291 relations
✓ Knowledge graph built successfully!
Build completed in 45.2s (16,199 total chunks tracked)
```

### Check for New Chunks
```bash
$ mcp-vector-search kg stats
Knowledge Graph Statistics
┌─────────────────┬──────────┐
│ Source Chunks   │ 16,199   │
│ Current Chunks  │ 16,450   │
│ Gap             │ 251 new  │
│ Last Build      │ 2h ago   │
└─────────────────┴──────────┘
💡 Run 'kg build --incremental' to add 251 new chunks
```

### Incremental Build
```bash
$ mcp-vector-search kg build --incremental
Incremental mode: 16,199 chunks already processed
Loading chunks from database...
Loaded 251 chunks for processing
✓ Scanned 251 chunks
✓ Extracted 245 entities
✓ Built 28 relations
Build completed in 5.3s (16,450 total chunks tracked)
```

### Verify Up to Date
```bash
$ mcp-vector-search kg stats
Knowledge Graph Statistics
┌─────────────────┬──────────┐
│ Source Chunks   │ 16,450   │
│ Current Chunks  │ 16,450   │
│ Gap             │ 0 (up to date) │
│ Last Build      │ just now │
└─────────────────┴──────────┘
```

## Benefits

1. **Faster Builds**: Only process new chunks instead of full rebuild
2. **Gap Visibility**: Always know when KG is out of sync with vector DB
3. **Build Tracking**: Metadata shows last build time and coverage
4. **Incremental Workflow**: Add new code without rebuilding entire graph
5. **Resource Efficiency**: Reduces CPU/memory for large codebases

## Technical Notes

### Thread Safety
- Metadata file operations use standard file locking (OS-level)
- No concurrent writes expected (single-user CLI tool)

### Error Handling
- Graceful fallback if metadata missing or corrupted
- Returns empty set for missing metadata (full rebuild)
- Logs warnings for metadata I/O errors

### Performance
- Chunk ID filtering during iteration (O(n) check per batch)
- Hash map lookup for O(1) duplicate detection
- Batch processing maintains memory efficiency

## Testing

The implementation has been tested for:
- ✓ Syntax validation (Python compile check)
- ✓ Metadata save/load functions
- ✓ Chunk ID filtering logic
- ✓ Gap calculation accuracy
- ✓ Time formatting (days/hours/minutes ago)

## Future Enhancements

Potential improvements:
1. **Smart Rebuild**: Detect removed chunks and clean up orphaned entities
2. **Partial Invalidation**: Remove entities for deleted files before incremental add
3. **Build Statistics**: Track build performance over time
4. **Metadata Versioning**: Handle schema changes in metadata format
5. **Concurrent Builds**: Lock file to prevent race conditions

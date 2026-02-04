# Two-Phase Architecture Migration

## Overview

Version 2.3.0 introduces a new two-phase indexing architecture that separates parsing (Phase 1) from embedding (Phase 2). This enables:

- **Faster parsing**: No expensive embedding generation during initial indexing
- **Incremental updates**: Skip unchanged files using file hash tracking
- **Resumable embedding**: Recover from crashes and resume where you left off
- **Better observability**: Track embedding progress separately from parsing

## What Changed

### Old Schema (Single Table)

Previously, mcp-vector-search stored everything in a single `code_chunks` or `code_search` table:

```
code_chunks:
  - chunk_id
  - content
  - vector (384D embedding)
  - file_path
  - language
  - metadata...
```

### New Schema (Two Tables)

The new architecture splits data across two tables:

```
chunks (Phase 1 - no vectors):
  - chunk_id
  - content
  - file_path
  - file_hash (NEW - for change detection)
  - language
  - embedding_status (NEW - pending/complete/error)
  - metadata...

vectors (Phase 2 - vectors + search fields):
  - chunk_id (FK to chunks)
  - vector (384D embedding)
  - file_path (denormalized for search)
  - content (truncated, for display)
  - language (denormalized for filtering)
  - search metadata...
```

## Migration Process

### Automatic Migration

The migration runs **automatically** when you run `mcp-vector-search index`. The indexer detects old schema and migrates data before indexing begins.

```bash
# Just run index as normal - migration happens automatically
mcp-vector-search index
```

Output:
```
Found 1 pending migration(s), running automatically...
Running migration: two_phase_architecture (v2.3.0)
Migrating 10,000 rows from code_chunks
✓ Created chunks table with 10,000 rows
✓ Created vectors table with 10,000 rows
✓ Migration two_phase_architecture completed successfully
```

### Manual Migration

You can also run migrations manually:

```bash
# List available migrations
mcp-vector-search migrate list

# Run pending migrations
mcp-vector-search migrate

# Dry run (preview changes without applying)
mcp-vector-search migrate --dry-run

# Run specific migration
mcp-vector-search migrate --version 2.3.0
```

## What Gets Migrated

The migration preserves **all** existing data:

1. **Chunks Table**:
   - All code content and metadata
   - Line numbers, file paths, languages
   - Function/class names, docstrings
   - Complexity scores and hierarchies
   - New fields: `file_hash` (empty), `embedding_status` (set to "complete")

2. **Vectors Table**:
   - All embedding vectors (384D)
   - Denormalized search fields (file_path, content, language)
   - Metadata for efficient search filtering
   - Model version marked as "migrated"

3. **Preserved Files**:
   - Old table (`code_chunks` or `code_search`) is **not deleted**
   - Preserved for rollback if needed
   - Safe to delete manually after verifying migration

## Rollback

If you encounter issues, you can rollback the migration:

```bash
# Drop new tables and restore old schema
mcp-vector-search migrate rollback --version 2.3.0
```

The old table is preserved during migration, so rollback is safe and non-destructive.

## Verification

After migration, verify your data:

```bash
# Check migration status
mcp-vector-search migrate status

# Check index stats (should match old counts)
mcp-vector-search status

# Test search (should work as before)
mcp-vector-search search "your query"
```

## Performance Impact

### Migration Speed

- **Small projects (<1k chunks)**: ~1 second
- **Medium projects (10k chunks)**: ~5-10 seconds
- **Large projects (100k chunks)**: ~30-60 seconds

Migration speed is dominated by:
1. Reading old table (pandas dataframe load)
2. Schema transformation (vector extraction)
3. Writing new tables (LanceDB bulk inserts)

### Post-Migration Benefits

- **Incremental indexing**: Only re-parse changed files (~10x faster)
- **Resumable embedding**: Crash recovery without re-parsing
- **Better observability**: Track Phase 1 vs Phase 2 progress separately

## Troubleshooting

### Migration Fails

If migration fails:

1. **Check logs**: Look for error messages in console output
2. **Verify space**: Ensure enough disk space (2x current index size)
3. **Manual retry**: Run `mcp-vector-search migrate --force --version 2.3.0`
4. **Report issue**: Open GitHub issue with error logs

### Search Not Working

If search fails after migration:

1. **Check table existence**: Run `mcp-vector-search status`
2. **Verify vector count**: Should match old chunk count
3. **Re-run embedding**: Run `mcp-vector-search index --phase embed`
4. **Rollback if needed**: Run `mcp-vector-search migrate rollback`

### Old Table Cleanup

After verifying migration:

```bash
# Manually delete old table (optional)
python -c "
import lancedb
db = lancedb.connect('.mcp-vector-search/lance')
db.drop_table('code_chunks')  # or 'code_search'
"
```

**Warning**: Only delete after confirming search works correctly!

## FAQ

### Q: Will migration delete my data?

**No.** The old table is preserved. Only new tables are created.

### Q: Do I need to re-index after migration?

**No.** All data and embeddings are migrated automatically.

### Q: What if I don't want to migrate?

You can stay on the old schema by:
1. Not upgrading to v2.3.0+
2. Skipping the migration when prompted

However, old schema won't receive future updates.

### Q: Can I migrate back to old schema?

Yes, use `mcp-vector-search migrate rollback --version 2.3.0`.

### Q: How long does migration take?

Typical speeds:
- 1k chunks: ~1 second
- 10k chunks: ~5 seconds
- 100k chunks: ~30 seconds
- 1M chunks: ~5 minutes

### Q: Is the old table deleted?

No, the old table is preserved for safety. You can delete it manually after verifying the migration.

## Technical Details

### Schema Changes

**Old `code_chunks` table**:
```python
{
    "chunk_id": str,
    "content": str,
    "vector": List[float],  # 384D
    "file_path": str,
    # ... metadata
}
```

**New `chunks` table**:
```python
{
    "chunk_id": str,
    "content": str,
    "file_path": str,
    "file_hash": str,  # NEW - SHA-256 of file
    "embedding_status": str,  # NEW - pending/complete/error
    "embedding_batch_id": int,  # NEW - for resume logic
    # ... metadata (no vector)
}
```

**New `vectors` table**:
```python
{
    "chunk_id": str,  # FK to chunks
    "vector": List[float],  # 384D
    # Denormalized search fields
    "file_path": str,
    "content": str,  # truncated to 500 chars
    "language": str,
    # ... search metadata
}
```

### Migration Implementation

See `src/mcp_vector_search/migrations/v2_3_0_two_phase.py` for implementation details.

Key steps:
1. Detect old table (`code_chunks` or `code_search`)
2. Read all data using pandas
3. Split into chunks (no vector) and vectors (with vector)
4. Create new tables with proper schemas
5. Preserve old table for rollback

### Auto-Migration Integration

The migration is integrated into the indexer initialization:

```python
# src/mcp_vector_search/core/indexer.py
async def index_project(self, ...):
    await self.chunks_backend.initialize()
    await self.vectors_backend.initialize()
    await self._run_auto_migrations()  # Runs automatically
    # ... continue indexing
```

## See Also

- [Two-Phase Architecture Design](./research/two-phase-indexing-architecture-2026-02-04.md)
- [Migration System](./MIGRATIONS.md)
- [Indexing Guide](./INDEXING.md)

# Bugfix: Schema Evolution Destroying Embeddings

**Date**: 2026-02-23
**Fixed by**: Claude Code
**Issue**: Schema evolution in `vectors_backend.py` was dropping and recreating tables, destroying all existing embeddings

## Problem

During `mvs reindex`, users saw this catastrophic warning:

```
WARNING: Schema evolution detected during append: existing table missing columns
{'function_name', 'project_name', 'class_name'}. Dropping and recreating table.
```

**Impact**: Instead of adding missing columns with null defaults, the code dropped the entire table and recreated it, **destroying ALL 325K existing embeddings**. This forced a full re-embed of all chunks, taking hours instead of minutes.

## Root Cause

In `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/vectors_backend.py`, the schema evolution handler had two issues:

1. **Destructive approach**: When missing columns were detected (lines 496-513 and 577-594), the code called:
   ```python
   self._db.drop_table(self.table_name)
   self._table = self._db.create_table(...)
   ```
   This destroyed all existing embeddings instead of adding columns.

2. **Missing append logic**: After adding columns in the `if missing_columns:` block, the code didn't continue to append the new data - it only appended in the `else` block.

## Solution

### Changes Made

**File**: `src/mcp_vector_search/core/vectors_backend.py`

**Lines 496-523** (first occurrence):
- **Before**: Drop+recreate table when missing columns
- **After**: Add missing columns with `add_columns(missing_fields)`, reopen table, then continue to append

**Lines 577-603** (second occurrence):
- **Before**: Drop+recreate table when missing columns
- **After**: Add missing columns with `add_columns(missing_fields)`, reopen table, then continue to append

### Key Implementation Details

```python
if missing_columns:
    # Schema evolution: existing table is missing new columns
    # Add missing columns with null values (preserves existing embeddings)
    logger.info(
        f"Schema evolution detected: adding missing columns {missing_columns} to existing table"
    )

    # Build PyArrow fields for missing columns
    missing_fields = [schema.field(col_name) for col_name in missing_columns]

    # Add columns with null/empty defaults
    self._table.add_columns(missing_fields)

    # Reopen table to get updated schema
    self._table = self._db.open_table(self.table_name)
    logger.info(
        f"Added {len(missing_columns)} new columns to vectors table (preserving existing embeddings)"
    )

# Schema compatible (either was already compatible or we just added missing columns)
# Deduplicate and append new data
self._delete_chunk_ids(chunk_ids_to_add)
self._table.add(pa_table, mode="append")
```

### LanceDB API Used

The fix uses LanceDB 0.29.2's `add_columns()` method:
```python
table.add_columns(missing_fields: List[pa.Field])
```

This adds columns with null/empty defaults to existing rows, preserving all data.

## Testing

### Manual Test
Created `/tmp/test_schema_evolution_real.py` to verify the fix:
1. Create table with OLD schema (missing 3 columns)
2. Add 2 vectors with OLD schema
3. Add 1 vector with NEW schema (triggers evolution)
4. **Result**: All 3 vectors preserved ✓

### Automated Tests
Added two comprehensive test cases to `tests/test_vectors_backend.py`:

1. **`test_schema_evolution_preserves_embeddings`** (lines 586-729)
   - Tests schema evolution during append (when `self._table` exists)
   - Verifies old embeddings preserved
   - Verifies old rows get new columns with empty defaults
   - Verifies new rows have populated values

2. **`test_schema_evolution_during_table_creation`** (lines 732-811)
   - Tests schema evolution during table creation (when `self._table` is None)
   - Verifies existing embeddings preserved when reopening table

### Test Results
- All 38 vectors backend tests pass ✓
- Full test suite: **1516 passed, 108 skipped** ✓

## Impact

### Before Fix
- ❌ Schema evolution destroyed ALL embeddings (325K chunks)
- ❌ Forced full re-embed (hours of work)
- ❌ Logged as WARNING (scary for users)
- ❌ No recovery without full reindex

### After Fix
- ✅ Schema evolution preserves ALL embeddings
- ✅ Only new/changed chunks need embedding (195K chunks)
- ✅ Logged as INFO (safe migration)
- ✅ Graceful forward compatibility

## Future Considerations

### When to Drop+Recreate
The fix only adds missing columns. We still drop+recreate if:
- **Vector dimension mismatch** (e.g., 384D → 768D model change)
- **Column TYPE mismatch** (rare, shouldn't happen in practice)

These cases are detected separately and handled correctly.

### Backward Compatibility
- Old tables automatically upgraded on first write
- No manual migration required
- Works across all LanceDB 0.29.x versions

## Lessons Learned

1. **Schema evolution should be additive**: Add columns with defaults, don't drop tables
2. **Test with real old schemas**: Don't just test with fresh data
3. **Control flow matters**: Ensure append logic runs after evolution
4. **Log levels matter**: INFO for safe operations, WARNING for destructive ones

## Related Files
- `src/mcp_vector_search/core/vectors_backend.py` (fixed)
- `tests/test_vectors_backend.py` (test coverage added)
- `/tmp/test_schema_evolution_real.py` (manual verification script)

## Commands to Verify Fix

```bash
# Run specific tests
uv run pytest tests/test_vectors_backend.py::test_schema_evolution_preserves_embeddings -v
uv run pytest tests/test_vectors_backend.py::test_schema_evolution_during_table_creation -v

# Run all vectors backend tests
uv run pytest tests/test_vectors_backend.py -x -q

# Run full test suite
uv run pytest tests/ -x -q
```

---

**Status**: ✅ Fixed and tested
**Breaking Changes**: None
**Migration Required**: None (automatic on first write)

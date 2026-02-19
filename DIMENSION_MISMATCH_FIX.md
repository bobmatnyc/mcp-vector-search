# Fix: Vector Dimension Mismatch (768D vs 384D)

## Problem

Users were getting dimension mismatch errors after switching default model from MiniLM (384D) to GraphCodeBERT (768D):

```
ERROR: Failed to add vectors: lance error: LanceError(Arrow): ListType can only be casted to FixedSizeListType if the lists are all the expected size.
input list: fixed_size_list<item: float>[768]
output list: fixed_size_list<item: float>[384]
```

## Root Causes

1. **Stale vectors.lance table**: The `vectors.lance` table was created with 384D schema when MiniLM was the default model, but new vectors being added were 768D from GraphCodeBERT.

2. **Missing dimension check in append path**: The code checked dimensions when opening tables (`self._table is None`), but skipped the check when appending to an already-opened table (`self._table` already set). This meant dimension mismatches weren't detected during batch operations.

3. **Hardcoded 384D defaults**: Several places in the code had 384D references from the MiniLM era that needed updating to 768D for GraphCodeBERT.

## Solutions Applied

### 1. Added Dimension Check to Append Path

**File**: `src/mcp_vector_search/core/vectors_backend.py` (lines 370-397)

**Change**: Added dimension verification BEFORE appending to existing table:

```python
else:
    # Check dimension mismatch before appending
    existing_schema = self._table.schema
    vector_field = existing_schema.field("vector")
    existing_dim = None
    if hasattr(vector_field.type, "list_size"):
        existing_dim = vector_field.type.list_size

    if existing_dim != self.vector_dim:
        # Dimension mismatch detected - drop and recreate table
        logger.warning(...)
        self._db.drop_table(self.TABLE_NAME)
        self._table = self._db.create_table(self.TABLE_NAME, pa_table, schema=schema)
    else:
        # Dimensions match, safe to append
        self._table.add(pa_table, mode="append")
```

**Why**: This ensures dimension mismatches are caught even when the table was opened in a previous call with stale dimension info.

### 2. Updated Default Schema to 768D

**File**: `src/mcp_vector_search/core/vectors_backend.py` (line 59-61)

**Change**:
```python
# Before
VECTORS_SCHEMA = _create_vectors_schema(384)

# After
VECTORS_SCHEMA = _create_vectors_schema(768)
```

**Why**: The default schema should match the default model (GraphCodeBERT = 768D, not MiniLM = 384D).

### 3. Updated Documentation and Comments

**Files**:
- `src/mcp_vector_search/core/vectors_backend.py` (multiple lines)

**Changes**:
- Updated docstring comments to say "dimension varies by model" instead of hardcoding 384D
- Updated PQ sub-vectors from 96 (384/4) to 192 (768/4)
- Updated comments to reference GraphCodeBERT as the default

## Testing

1. **Manual Verification**: Deleted `vectors.lance` table and re-indexed successfully with 768D vectors
2. **Dimension Auto-Detection**: Confirmed model produces 768D vectors
3. **Schema Validation**: Verified new table has correct 768D schema

```bash
# Verify model dimension
uv run python -c "from mcp_vector_search.core.embeddings import create_embedding_function; ef, _ = create_embedding_function(); vec = ef(['test']); print(f'Model dimension: {len(vec[0])}')"
# Output: Model dimension: 768

# Verify table schema
uv run python -c "import lancedb; db = lancedb.connect('.mcp-vector-search/lance'); table = db.open_table('vectors'); schema = table.schema; vector_field = schema.field('vector'); print(f'Vector dimension: {vector_field.type.list_size}')"
# Output: Vector dimension: 768
```

## Migration Path

For users with existing indexes:

1. **Option A: Delete and Reindex** (fastest, but loses existing data):
   ```bash
   rm -rf .mcp-vector-search/lance/vectors.lance
   uv run mcp-vector-search index -f
   ```

2. **Option B: Re-embed with New Model** (preserves data, slower):
   ```bash
   uv run mcp-vector-search re-embed --model microsoft/graphcodebert-base
   ```

3. **Option C: Let Auto-Fix Handle It** (automatic, next index run):
   - The dimension check in append path will auto-detect mismatch and recreate table
   - No manual intervention needed

## Verification Commands

```bash
# Check current vector dimension in table
uv run python -c "import lancedb; db = lancedb.connect('.mcp-vector-search/lance'); table = db.open_table('vectors'); schema = table.schema; vector_field = schema.field('vector'); print(f'Table dimension: {vector_field.type.list_size}')"

# Check model dimension
uv run python -c "from mcp_vector_search.core.embeddings import create_embedding_function; ef, dim = create_embedding_function(); print(f'Model dimension: {dim or len(ef([\"test\"])[0])}')"

# Both should output: 768
```

## Related Files

- `src/mcp_vector_search/core/vectors_backend.py` - Main fix (dimension check + defaults)
- `src/mcp_vector_search/core/lancedb_backend.py` - Has auto-detection already (no changes needed)
- `src/mcp_vector_search/core/embeddings.py` - Model selection (already defaulting to GraphCodeBERT)

## Version

Fixed in: **v2.5.45** (unreleased)

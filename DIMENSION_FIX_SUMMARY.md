# Dimension Mismatch Fix Summary (v2.5.45)

## Problem Fixed ✅

**Issue**: "ListType can only be casted to FixedSizeListType" errors when indexing

**Symptom**:
```
input list: fixed_size_list<item: float>[768]
output list: fixed_size_list<item: float>[384]
```

## Root Cause Analysis

The bug was **NOT** in the creation of new tables (that already had dimension auto-detection), but in the **append path** for existing tables:

1. **Stale Table Handle**: When `self._table` was already opened, the code skipped dimension checking
2. **384D Legacy**: The `vectors.lance` table was created with 384D schema (from MiniLM era)
3. **768D New Data**: GraphCodeBERT produces 768D vectors
4. **Dimension Check Bypassed**: The dimension verification only ran when `self._table is None`, not during append operations

## The Fix

### Code Changes

**File**: `src/mcp_vector_search/core/vectors_backend.py`

**Change 1**: Added dimension check to append path (lines 372-401)
- Verify existing table dimension matches new data dimension BEFORE append
- Auto-drop and recreate table if mismatch detected
- Handles stale table handles gracefully

**Change 2**: Updated default schema from 384D to 768D (line 59-61)
- `VECTORS_SCHEMA` now uses GraphCodeBERT dimension (768D)
- Updated PQ sub-vectors from 96 to 192 (768/4)
- Updated all docstrings to remove hardcoded 384D references

### Why This Works

**Before Fix**:
```python
# vectors_backend.py (old code)
else:
    self._table.add(pa_table, mode="append")  # ❌ No dimension check!
```

**After Fix**:
```python
# vectors_backend.py (new code)
else:
    # Check dimension mismatch before appending
    existing_dim = existing_schema.field("vector").type.list_size
    if existing_dim != self.vector_dim:
        # ✅ Auto-detect mismatch and recreate table
        self._db.drop_table(self.TABLE_NAME)
        self._table = self._db.create_table(...)
    else:
        self._table.add(pa_table, mode="append")
```

## Migration Options

### Option A: Manual Delete (Fastest) ⚡
```bash
rm -rf .mcp-vector-search/lance/vectors.lance
uv run mcp-vector-search index -f
```

### Option B: Re-embed Command (Preserves Data) 📦
```bash
uv run mcp-vector-search re-embed --model microsoft/graphcodebert-base
```

### Option C: Automatic (No Action) 🤖
Just update to v2.5.45 and run next index. The auto-fix will handle it.

## Verification

### Test 1: Check Model Dimension
```bash
uv run python -c "from mcp_vector_search.core.embeddings import create_embedding_function; ef, _ = create_embedding_function(); vec = ef(['test']); print(f'Model dimension: {len(vec[0])}')"
```
Expected: `Model dimension: 768`

### Test 2: Check Table Dimension
```bash
uv run python -c "import lancedb; db = lancedb.connect('.mcp-vector-search/lance'); table = db.open_table('vectors'); schema = table.schema; vector_field = schema.field('vector'); print(f'Table dimension: {vector_field.type.list_size}')"
```
Expected: `Table dimension: 768`

### Test 3: Index Successfully
```bash
uv run mcp-vector-search index
```
Expected: No dimension mismatch errors

## Related Commits

- `b1e24b0`: fix: add dimension mismatch detection to append path and update 768D defaults
- Previous: `vectors_backend.py` already had dimension check for NEW tables (lines 146-178)
- This fix: Added dimension check for EXISTING tables during append (lines 372-401)

## Technical Details

### Why Dimension Mismatch Happens

1. **Table Creation**: `vectors.lance` created with schema at table creation time
2. **Model Switch**: Default model changed from MiniLM (384D) to GraphCodeBERT (768D)
3. **Stale Schema**: Old table retained 384D schema
4. **New Vectors**: New embeddings generated with 768D
5. **LanceDB Cast Error**: LanceDB tried to cast 768D vectors to fit 384D schema → ERROR

### Why Previous Fix Didn't Catch This

The previous dimension auto-detection code (lines 146-178) only ran when:
- Opening a NEW table (`self._table is None`)
- Table doesn't exist yet

It didn't run when:
- Table was already opened (`self._table` already set)
- Appending to existing table handle

This fix closes that gap by checking dimensions BEFORE EVERY append operation.

## Version Info

- **Fixed in**: v2.5.45
- **Released**: 2026-02-19
- **Commit**: b1e24b0

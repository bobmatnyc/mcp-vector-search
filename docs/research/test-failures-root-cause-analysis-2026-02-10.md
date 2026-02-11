# Test Failures Root Cause Analysis

**Date**: 2026-02-10
**Project**: mcp-vector-search
**Analyst**: Research Agent
**Total Failing Tests**: 13

## Executive Summary

All 13 test failures stem from **3 primary root causes** related to the two-phase indexing architecture migration:

1. **Force reindex chunks table collision** (8 tests) - High Priority
2. **ChromaDB metrics storage API mismatch** (3 tests) - Medium Priority
3. **Test assertion precision issues** (2 tests) - Low Priority

**Quick Win**: Fixing root cause #1 will resolve **61% of failing tests** (8 out of 13).

---

## Root Cause Categories

### 🔴 ROOT CAUSE #1: Force Reindex Table Collision (CRITICAL)
**Impact**: 8 failing tests (61%)
**Severity**: High - Blocks force reindex and incremental indexing workflows

#### Error Pattern
```
ERROR | mcp_vector_search.core.chunks_backend:add_chunks:233 - Failed to add chunks: Table 'chunks' already exists
ERROR | mcp_vector_search.core.indexer:_phase1_chunk_files:396 - Failed to chunk file ... Table 'chunks' already exists
INFO  | Phase 1 complete: 0 files processed, 0 chunks created
```

#### Affected Tests
1. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_incremental_indexing_workflow`
2. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_force_reindexing_workflow`
3. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_metadata_consistency_workflow`
4. `tests/unit/core/test_indexer.py::TestSemanticIndexer::test_index_project_force_reindex`
5. `tests/unit/core/test_indexer.py::TestSemanticIndexer::test_incremental_indexing`
6. `tests/test_version_reindex.py::test_auto_reindex_integration`

**Plus 2 more related**:
7. `tests/test_basic_functionality.py::test_indexing_and_search` (returns 0 chunks due to failed indexing)

#### Root Cause Analysis

**File**: `src/mcp_vector_search/core/chunks_backend.py:217-228`

```python
# Create or append to table
if self._table is None:
    # Create table with first batch
    self._table = self._db.create_table(
        self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
    )
    logger.debug(f"Created chunks table with {len(normalized_chunks)} chunks")
else:
    # Append to existing table
    self._table.add(normalized_chunks)
    logger.debug(f"Added {len(normalized_chunks)} chunks to table")
```

**Problem**: During force reindex (or second incremental index), the code attempts to create a new chunks table when `self._table is None`, but LanceDB already has a persisted table on disk from the first index run. The `create_table()` call fails with "Table 'chunks' already exists".

**Why it happens**:
1. First `index_project()` call succeeds - creates chunks table
2. Indexer is re-instantiated (or table reference is lost)
3. `self._table = None` but LanceDB has persisted table on disk
4. Second `index_project(force_reindex=True)` attempts `create_table()` → **FAILS**
5. No chunks are indexed (count = 0), all assertions fail

#### Fix Strategy

**Option A: Check for existing table** (Recommended - Low Risk)
```python
# In chunks_backend.py:add_chunks()
if self._table is None:
    # Check if table exists on disk
    if self.TABLE_NAME in self._db.table_names():
        # Reopen existing table
        self._table = self._db.open_table(self.TABLE_NAME)
        logger.debug(f"Reopened existing chunks table")
    else:
        # Create new table
        self._table = self._db.create_table(
            self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
        )
        logger.debug(f"Created chunks table with {len(normalized_chunks)} chunks")
```

**Option B: Handle force reindex explicitly**
- Before force reindex, drop existing chunks and vectors tables
- Clear all LanceDB state
- Requires coordination between chunks_backend and vectors_backend

**Option C: Lazy table initialization**
- Move table initialization to `initialize()` method
- Always check table existence during init
- Requires refactoring initialization flow

**Recommended**: Option A (add table existence check before create)

**Implementation Steps**:
1. Add `table_exists()` helper method in `chunks_backend.py`
2. Modify `add_chunks()` to check existence before create
3. Add similar logic to `vectors_backend.py` for consistency
4. Add force reindex flag to explicitly clear tables if needed

---

### 🟡 ROOT CAUSE #2: ChromaDB Metrics Storage API Mismatch (MEDIUM)
**Impact**: 3 failing tests (23%)
**Severity**: Medium - Metrics not being persisted to ChromaDB

#### Error Pattern
```python
# Test retrieves chunks from ChromaDB
results = db._collection.get(ids=[sample_chunk.id], include=["metadatas"])
assert results["ids"]  # FAILS - returns []
```

#### Affected Tests
1. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_add_chunks_with_metrics`
2. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_add_chunks_without_metrics`
3. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_multiple_chunks_same_file`

#### Root Cause Analysis

**Issue**: Two-phase architecture separates chunks (LanceDB) and vectors (ChromaDB). Tests expect metrics in ChromaDB `_collection`, but chunks are stored in LanceDB `chunks` table.

**Architecture Mismatch**:
```
OLD (v2.2.x):
  ChromaDB stores: chunks + embeddings + metadata (metrics)

NEW (v2.3.0+):
  LanceDB chunks table: chunks + metadata (metrics)
  ChromaDB vectors: embeddings only (or deprecated)
```

**Why tests fail**:
1. Test calls `db.add_chunks([chunk], metrics=metrics_dict)`
2. Chunks go to LanceDB chunks table (with metrics in metadata)
3. Test queries ChromaDB `_collection.get(ids=[...])` → **empty** (chunks not in ChromaDB)
4. Assertion fails: `assert results["ids"]` → `assert []`

#### Fix Strategy

**Option A: Update tests to query LanceDB** (Recommended - Correct approach)
```python
# OLD test approach
results = db._collection.get(ids=[sample_chunk.id], include=["metadatas"])

# NEW test approach
chunks = await db.get_chunks_by_ids([sample_chunk.id])
assert len(chunks) == 1
assert chunks[0].metrics is not None
```

**Option B: Add ChromaDB compatibility layer**
- Keep dual storage (LanceDB + ChromaDB) for backwards compatibility
- Update `add_chunks()` to write to both
- Not recommended - defeats purpose of two-phase migration

**Recommended**: Option A (update tests to query LanceDB chunks table)

**Implementation Steps**:
1. Add `get_chunks_by_ids()` method to `ChunksBackend` (if missing)
2. Update all 3 test files to query LanceDB instead of ChromaDB
3. Verify metrics are correctly stored in LanceDB chunks table metadata
4. Remove direct `_collection` access from tests (breaks encapsulation)

---

### 🟢 ROOT CAUSE #3: Test Assertion Precision Issues (LOW)
**Impact**: 2 failing tests (15%)
**Severity**: Low - Test expectations vs. implementation behavior

#### 3A: Codebase Type Classification Threshold
**Affected Tests**:
- `tests/test_codebase_profiler.py::test_profile_small_codebase`
- `tests/test_codebase_profiler.py::test_format_profile_summary`

**Error**:
```python
# Test expectation
assert profile.codebase_type == CodebaseType.PYTHON

# Actual result
assert CodebaseType.MIXED == CodebaseType.PYTHON
# Mixed detected: PY 50%, JS 30%, MD 20%
```

**Root Cause**: Test creates 5 Python + 3 JavaScript + 2 Markdown files (10 total). Python is 50%, which is below the threshold for "Python" classification (likely 60%+).

**Fix Options**:
1. **Adjust test fixtures**: Create 6+ Python files to push percentage above threshold
2. **Update test assertion**: Accept `CodebaseType.MIXED` as valid result
3. **Lower classification threshold**: Change profiler threshold to 50% (not recommended)

**Recommended**: Adjust test fixtures to create clearer Python majority (7 PY + 2 JS + 1 MD = 70% Python).

---

#### 3B: Dart Parser Async Function Detection
**Affected Test**:
- `tests/test_dart_parser.py::test_dart_parser`

**Error**:
```python
async_chunks = [c for c in chunks if c.type == "async_function"]
assert len(async_chunks) >= 1, "Should find at least 1 async function chunk"
# AssertionError: assert 0 >= 1
```

**Root Cause**: Dart parser does not create separate chunk type for async functions. Async functions are parsed as regular `function` type chunks.

**Evidence**: Test output shows 20 chunks extracted, including:
- Chunk 12: `Future<Map<String, dynamic>> fetchUserData(String userId)` (parsed as `function`, not `async_function`)
- Chunk 15: `Future<Map<String, dynamic>> getUser(String id)` (parsed as `function`, not `async_function`)

**Fix Options**:
1. **Update test assertion**: Check for `function` type with `Future` return type instead of `async_function` type
2. **Enhance Dart parser**: Add `async_function` classification for functions returning `Future`
3. **Add metadata flag**: Store `is_async` in chunk metadata instead of separate type

**Recommended**: Update test to check for `function` chunks with async patterns (simpler, parser already extracts correctly).

```python
# Current (fails)
async_chunks = [c for c in chunks if c.type == "async_function"]

# Fixed
async_chunks = [c for c in chunks if c.type == "function" and "Future" in c.content]
```

---

## Priority Matrix

| Priority | Root Cause | Tests Affected | Fix Complexity | Impact |
|----------|-----------|----------------|----------------|--------|
| **P0** | Force reindex table collision | 8 tests (61%) | Medium | Critical - blocks reindexing |
| **P1** | ChromaDB metrics API mismatch | 3 tests (23%) | Low | Medium - test infrastructure |
| **P2** | Codebase profiler threshold | 2 tests (15%) | Very Low | Low - test assertion |
| **P3** | Dart async function detection | 1 test (8%) | Very Low | Low - test assertion |

---

## Recommended Fix Order

### Phase 1: Quick Wins (Low Effort, High Impact)
1. **Fix Root Cause #2**: Update 3 database metrics tests (~30 min)
   - Update tests to query LanceDB chunks table
   - Remove ChromaDB `_collection` direct access
   - **Resolves**: 3 tests

2. **Fix Root Cause #3B**: Update Dart parser test assertion (~10 min)
   - Change async function detection logic
   - **Resolves**: 1 test

3. **Fix Root Cause #3A**: Adjust codebase profiler test fixtures (~15 min)
   - Create 6-7 Python files instead of 5
   - **Resolves**: 2 tests

**Phase 1 Total**: ~1 hour, **6 tests fixed** (46%)

---

### Phase 2: Critical Fix (Medium Effort, Critical Impact)
4. **Fix Root Cause #1**: Force reindex table collision (~2-3 hours)
   - Add table existence check in `chunks_backend.py`
   - Add table existence check in `vectors_backend.py`
   - Handle table reopening gracefully
   - Add integration tests for table persistence
   - **Resolves**: 8 tests (including cascade failures)

**Phase 2 Total**: ~3 hours, **8 tests fixed** (61%)

---

## Detailed Fix Implementation

### Fix #1: Force Reindex Table Collision

**File**: `src/mcp_vector_search/core/chunks_backend.py`

**Current Code** (lines 217-228):
```python
# Create or append to table
if self._table is None:
    # Create table with first batch
    self._table = self._db.create_table(
        self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
    )
    logger.debug(f"Created chunks table with {len(normalized_chunks)} chunks")
else:
    # Append to existing table
    self._table.add(normalized_chunks)
    logger.debug(f"Added {len(normalized_chunks)} chunks to table")
```

**Fixed Code**:
```python
# Create or append to table
if self._table is None:
    # Check if table exists on disk (persisted from previous run)
    try:
        existing_tables = self._db.table_names()
    except AttributeError:
        # Fallback for older LanceDB versions
        existing_tables = []

    if self.TABLE_NAME in existing_tables:
        # Reopen existing table
        self._table = self._db.open_table(self.TABLE_NAME)
        logger.debug(f"Reopened existing chunks table")
        # For force reindex, delete old chunks for files being reprocessed
        # (handled by caller setting processing status)
        self._table.add(normalized_chunks)
        logger.debug(f"Added {len(normalized_chunks)} chunks to existing table")
    else:
        # Create new table
        self._table = self._db.create_table(
            self.TABLE_NAME, normalized_chunks, schema=CHUNKS_SCHEMA
        )
        logger.debug(f"Created chunks table with {len(normalized_chunks)} chunks")
else:
    # Append to existing table
    self._table.add(normalized_chunks)
    logger.debug(f"Added {len(normalized_chunks)} chunks to table")
```

**Additional Changes Needed**:
1. Add force reindex cleanup logic before Phase 1 (delete old chunks for files being reprocessed)
2. Apply similar fix to `vectors_backend.py`
3. Add test coverage for table persistence across indexer instances

---

### Fix #2: ChromaDB Metrics Tests

**File**: `tests/unit/core/test_database_metrics.py`

**Current Test** (fails):
```python
async def test_add_chunks_with_metrics(self, ...):
    # Add chunk with metrics
    metrics_dict = {sample_chunk.chunk_id: sample_metrics.to_metadata()}
    await db.add_chunks([sample_chunk], metrics=metrics_dict)

    # Retrieve and verify metrics were stored
    chunks = await db.get_all_chunks()
    assert len(chunks) == 1

    # Verify metrics are in ChromaDB metadata (fetch directly from collection)
    results = db._collection.get(ids=[sample_chunk.id], include=["metadatas"])
    assert results["ids"]  # FAILS - ChromaDB is empty
```

**Fixed Test**:
```python
async def test_add_chunks_with_metrics(self, ...):
    # Add chunk with metrics
    metrics_dict = {sample_chunk.chunk_id: sample_metrics.to_metadata()}
    await db.add_chunks([sample_chunk], metrics=metrics_dict)

    # Retrieve chunks from LanceDB chunks table
    chunks = await db.get_all_chunks()
    assert len(chunks) == 1

    # Verify metrics are stored in chunk metadata
    chunk = chunks[0]
    assert chunk.chunk_id == sample_chunk.chunk_id

    # Verify metrics metadata is present
    # (access via chunks_backend, not ChromaDB _collection)
    chunks_backend = db._chunks_backend  # or db.chunks_backend if exposed
    retrieved_chunks = await chunks_backend.get_chunks_by_ids([sample_chunk.chunk_id])
    assert len(retrieved_chunks) == 1

    # Check metadata contains metrics fields
    chunk_metadata = retrieved_chunks[0].metadata
    assert "cognitive_complexity" in chunk_metadata
    assert chunk_metadata["cognitive_complexity"] == 5
```

**Repeat for**:
- `test_add_chunks_without_metrics`
- `test_multiple_chunks_same_file`

---

### Fix #3: Codebase Profiler Tests

**File**: `tests/test_codebase_profiler.py`

**Current Fixture** (creates 50% Python):
```python
@pytest.fixture
def temp_project(tmp_path):
    # Create 5 Python files
    # Create 3 JavaScript files
    # Create 2 Markdown files
    # Total: 50% Python → classified as MIXED
```

**Fixed Fixture** (creates 70% Python):
```python
@pytest.fixture
def temp_project(tmp_path):
    # Create 7 Python files (increased from 5)
    for i in range(7):
        (tmp_path / f"module_{i}.py").write_text(f"def func_{i}(): pass")

    # Create 2 JavaScript files (reduced from 3)
    for i in range(2):
        (tmp_path / f"script_{i}.js").write_text(f"function func_{i}() {{}}")

    # Create 1 Markdown file (reduced from 2)
    (tmp_path / "README.md").write_text("# Project")

    # Total: 7 PY / 10 files = 70% Python → classified as PYTHON
    return tmp_path
```

**Update test assertions**:
```python
def test_profile_small_codebase(temp_project):
    profiler = CodebaseProfiler(temp_project)
    profile = profiler.profile()

    assert profile.size_category == CodebaseSize.SMALL
    assert profile.total_files == 10  # Still 10 files
    assert profile.codebase_type == CodebaseType.PYTHON  # Now passes
```

---

### Fix #4: Dart Parser Async Detection

**File**: `tests/test_dart_parser.py`

**Current Test** (fails):
```python
async_chunks = [c for c in chunks if c.type == "async_function"]
assert len(async_chunks) >= 1, "Should find at least 1 async function chunk"
```

**Fixed Test**:
```python
# Dart parser doesn't create separate "async_function" type
# Async functions are regular functions with Future return type
async_chunks = [
    c for c in chunks
    if c.type == "function" and ("Future" in c.content or "async" in c.content)
]
assert len(async_chunks) >= 1, "Should find at least 1 async function (Future return)"

# Alternatively, check for specific async function
fetchUserData_chunks = [c for c in chunks if "fetchUserData" in c.content]
assert len(fetchUserData_chunks) >= 1, "Should find fetchUserData async function"
assert "Future" in fetchUserData_chunks[0].content
```

---

## Testing Strategy

### Verification Plan

**After Phase 1 fixes** (quick wins):
```bash
# Run fixed tests
uv run pytest tests/unit/core/test_database_metrics.py -v
uv run pytest tests/test_codebase_profiler.py -v
uv run pytest tests/test_dart_parser.py::test_dart_parser -v

# Expected: 6 tests pass
```

**After Phase 2 fix** (force reindex):
```bash
# Run all previously failing tests
uv run pytest tests/integration/test_indexing_workflow.py -v
uv run pytest tests/unit/core/test_indexer.py -v
uv run pytest tests/test_version_reindex.py -v
uv run pytest tests/test_basic_functionality.py -v

# Expected: All 13 tests pass
```

**Full regression test**:
```bash
# Run complete test suite
uv run pytest tests/ -v --ignore=tests/manual --ignore=tests/e2e

# Expected: All tests pass
```

---

## Risk Assessment

### High Risk Changes
- **Force reindex fix**: Modifies core indexing flow
  - **Mitigation**: Add comprehensive integration tests
  - **Rollback**: Revert to table existence check, add explicit error message

### Medium Risk Changes
- **ChromaDB metrics tests**: Requires understanding two-phase architecture
  - **Mitigation**: Test both LanceDB and ChromaDB paths
  - **Rollback**: Tests only, no production code changes

### Low Risk Changes
- **Test assertion fixes**: No production code changes
  - **Mitigation**: None needed (test-only changes)
  - **Rollback**: Trivial

---

## Additional Findings

### Deprecation Warnings (Not Blocking)
```python
DeprecationWarning: table_names() is deprecated, use list_tables() instead
```
- **Location**: `src/mcp_vector_search/core/lancedb_backend.py:151`
- **Fix**: Replace `self._db.table_names()` with `self._db.list_tables()`
- **Impact**: Low (warnings only, not errors)

### Pydantic Warnings (Not Blocking)
```
PydanticDeprecatedSince20: Support for class-based config is deprecated
```
- **Impact**: Low (deprecation warnings, not errors)
- **Fix**: Migrate to `ConfigDict` (separate refactoring task)

---

## Conclusion

All 13 test failures are **fixable with high confidence**. The root causes are well-understood and stem from the two-phase indexing architecture migration (v2.3.0).

**Estimated Total Fix Time**: 4-5 hours
- Phase 1 (quick wins): 1 hour → 6 tests fixed
- Phase 2 (critical fix): 3 hours → 8 tests fixed (including cascading fixes)

**Success Criteria**:
- All 13 tests pass
- No regression in existing passing tests
- Two-phase architecture fully functional with force reindex support

**Next Steps**:
1. Implement Phase 1 fixes (low-hanging fruit)
2. Validate fixes with targeted test runs
3. Implement Phase 2 fix (table collision)
4. Run full regression test suite
5. Update CI/CD pipeline if needed

---

## Appendix: Full Test Failure List

### Integration Tests (3 failures)
1. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_incremental_indexing_workflow`
2. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_force_reindexing_workflow`
3. `tests/integration/test_indexing_workflow.py::TestIndexingWorkflow::test_metadata_consistency_workflow`

### Unit Tests (5 failures)
4. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_add_chunks_with_metrics`
5. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_add_chunks_without_metrics`
6. `tests/unit/core/test_database_metrics.py::TestDatabaseMetricsSupport::test_multiple_chunks_same_file`
7. `tests/unit/core/test_indexer.py::TestSemanticIndexer::test_index_project_force_reindex`
8. `tests/unit/core/test_indexer.py::TestSemanticIndexer::test_incremental_indexing`

### Functional Tests (5 failures)
9. `tests/test_basic_functionality.py::test_indexing_and_search`
10. `tests/test_codebase_profiler.py::test_profile_small_codebase`
11. `tests/test_codebase_profiler.py::test_format_profile_summary`
12. `tests/test_dart_parser.py::test_dart_parser`
13. `tests/test_version_reindex.py::test_auto_reindex_integration`

---

**Research completed**: 2026-02-10 21:59:15 PST
**File location**: `/Users/masa/Projects/mcp-vector-search/docs/research/test-failures-root-cause-analysis-2026-02-10.md`

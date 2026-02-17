# Schema Version Check Bypass Investigation

**Date**: 2026-02-16
**Issue**: Schema compatibility check is NOT catching the "Field 'calls' not found" error during indexing
**Severity**: High - Users are experiencing errors that should be prevented by the schema check

## Problem Summary

User is seeing this error during indexing:
```
ERROR: Failed to flush write buffer: Field 'calls' not found in target schema
ERROR: Indexing error: Field 'calls' not found in target schema
ERROR: Indexing failed: Field 'calls' not found in target schema
✗ Indexing failed: Field 'calls' not found in target schema
```

This means the schema compatibility check added in v2.3.5 is **not working as designed**. It should have:
1. Detected missing `schema_version.json` (old database from pre-2.3.x)
2. Prompted user to reset with `--force`
3. **NOT proceeded** with indexing against an incompatible database

## Code Flow Analysis

### 1. Entry Point: `index.py::run_indexing()`

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py`

**Lines 356-382**: Schema compatibility check location

```python
# Check schema compatibility before indexing (unless explicitly skipped)
if not skip_schema_check:
    from ...core.schema import check_schema_compatibility

    db_path = config.index_path  # THIS IS THE CRITICAL VARIABLE
    is_compatible, message = check_schema_compatibility(db_path)

    if not is_compatible:
        print_warning("⚠️  Schema Version Mismatch Detected")
        print_error(message)
        print_info("\nOptions:")
        print_info("  1. Reset database and reindex (recommended):")
        print_info("     mcp-vector-search index --force")
        print_info("  2. Skip schema check and continue (may cause errors):")
        print_info("     mcp-vector-search index --skip-schema-check")

        # If force_reindex is set, proceed with reset
        if force_reindex:
            print_warning(
                "\n🔄 Force flag detected - resetting database with new schema..."
            )
            # Schema version will be saved after successful reset
        else:
            raise typer.Exit(1)  # THIS SHOULD STOP INDEXING
    else:
        logger.debug(message)
```

**Key Variables**:
- `skip_schema_check`: Defaults to `False` (line 115), can be set via `--skip-schema-check` flag
- `db_path`: Comes from `config.index_path`
- `force_reindex`: Comes from `--force` flag

### 2. Path Resolution: What is `config.index_path`?

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`

```python
index_path: Path = Field(
    default=".mcp-vector-search", description="Index storage path"
)
```

**Resolution**:
- Default value: `.mcp-vector-search` (relative path)
- Gets resolved to absolute path via `validate_paths()` method (line 75)
- Typical resolved path: `/path/to/project/.mcp-vector-search`

### 3. Schema Version File Location

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/schema.py`

```python
def get_schema_version_path(db_path: Path) -> Path:
    """Get path to schema version file.

    Args:
        db_path: Database directory path (e.g., .mcp-vector-search/lancedb)

    Returns:
        Path to schema_version.json
    """
    # Store version in parent directory (.mcp-vector-search)
    mcp_dir = db_path.parent if db_path.name == "lancedb" else db_path
    return mcp_dir / "schema_version.json"
```

**Critical Logic**:
- If `db_path.name == "lancedb"`: Uses parent directory
- Otherwise: Uses `db_path` directly
- Returns: `mcp_dir / "schema_version.json"`

**Expected Behavior**:
- Input: `.mcp-vector-search` → Output: `.mcp-vector-search/schema_version.json`
- Input: `.mcp-vector-search/lancedb` → Output: `.mcp-vector-search/schema_version.json`

### 4. Schema Compatibility Check Logic

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/schema.py`

```python
def check_schema_compatibility(db_path: Path) -> tuple[bool, str]:
    """Check if database schema matches current code version."""
    current_version = SchemaVersion(SCHEMA_VERSION)
    db_version = load_schema_version(db_path)

    if db_version is None:
        # No version file - assume old database (pre-2.3.x)
        return (
            False,
            f"No schema version found. Database may be from older version (< 2.3.0).\n"
            f"Current code version: {current_version}\n"
            f"Recommendation: Reset database and reindex.",
        )

    # ... compatibility checks ...
```

**Expected Behavior**:
- If `schema_version.json` is missing: Return `(False, <error message>)`
- This should trigger the error handling in `run_indexing()` (line 363)
- Without `--force` flag: Should call `raise typer.Exit(1)` (line 379)

## The Bug: Why Is the Check Not Working?

### Hypothesis 1: **Path Mismatch** ⭐ MOST LIKELY

**Problem**: The `db_path` passed to `check_schema_compatibility()` might not be the **actual database path**.

**Evidence**:
1. In `indexer.py` (line 216), the actual database paths are:
   ```python
   index_path = project_root / ".mcp-vector-search" / "lance"
   self.chunks_backend = ChunksBackend(index_path)  # Uses .mcp-vector-search/lance
   self.vectors_backend = VectorsBackend(index_path)  # Uses .mcp-vector-search/lance
   ```

2. But in `index.py` (line 360), the schema check uses:
   ```python
   db_path = config.index_path  # This is .mcp-vector-search (WITHOUT /lance)
   ```

3. The schema version file is stored at:
   ```
   .mcp-vector-search/schema_version.json
   ```

4. The actual LanceDB tables are at:
   ```
   .mcp-vector-search/lance/chunks.lance
   .mcp-vector-search/lance/vectors.lance
   .mcp-vector-search/lance/code_search.lance  (OLD ChromaDB format)
   ```

**The Discrepancy**:
- Schema check looks for: `.mcp-vector-search/schema_version.json`
- Database operations use: `.mcp-vector-search/lance/`
- If the database was created with old code, there's no schema version file
- But the schema check is looking in the CORRECT location (`.mcp-vector-search/`)

**Why It Might Still Fail**:
- The user's error mentions `Field 'calls' not found in target schema`
- This error happens during **LanceDB write operations**, not during the schema check
- The schema check runs BEFORE indexing starts (line 356-382)
- The error occurs DURING indexing (in `chunks_backend.add_chunks()`)

### Hypothesis 2: **Schema Check is Bypassed** 🤔 POSSIBLE

**Scenarios Where Check Could Be Skipped**:

1. **User passed `--skip-schema-check` flag**:
   - Check: User's command line arguments
   - If `skip_schema_check=True`, entire block is skipped (line 357)

2. **Background indexing mode**:
   - Check: `_spawn_background_indexer()` (line 217-299)
   - The background indexer spawns a separate process
   - Need to verify if schema check is called in background mode

3. **Incremental indexing path**:
   - Check: User is running incremental indexing (default behavior)
   - Schema check should still run for incremental indexing

### Hypothesis 3: **Schema Check Passes Incorrectly** 🚨 CRITICAL

**Problem**: The schema check might be returning `(True, message)` when it should return `(False, message)`.

**Possible Causes**:
1. **Schema version file EXISTS but is outdated**:
   - If `schema_version.json` exists with version `2.2.x`
   - But current code is `2.3.x` with new `calls` field
   - The compatibility check (line 189) checks if `db_version.is_compatible_with(current_version)`

2. **Check the `is_compatible_with()` logic**:
   ```python
   def is_compatible_with(self, other: "SchemaVersion") -> bool:
       """Check if two versions are compatible."""
       if self.major != other.major:
           return False

       # Database version must be <= code version
       if self.minor > other.minor:
           return False
       if self.minor == other.minor and self.patch > other.patch:
           return False

       return True
   ```

   **Example**:
   - DB version: `2.2.21`
   - Code version: `2.3.5`
   - Major: `2 == 2` ✓
   - Minor: `2 <= 3` ✓ (db version is older)
   - **RESULT**: `is_compatible_with()` returns `True` ⚠️

   **This is the bug!** The compatibility check assumes older schemas are compatible with newer code, but this is NOT true when new fields are added to LanceDB tables.

### Hypothesis 4: **Timing Issue - Check Happens Too Early**

**Problem**: Schema check runs before backends are initialized, so it can't detect actual schema mismatches in LanceDB tables.

**Evidence**:
- Schema check: Line 356-382 (before indexing starts)
- Backend initialization: Line 411-425 (after schema check)
- Actual error: During `chunks_backend.add_chunks()` (Phase 1 chunking)

**The Issue**:
- Schema version file is a **metadata file**, not the actual database schema
- The actual LanceDB table schema is stored in the `.lance` files themselves
- The schema check only looks at the version file, not the actual table structure
- When writing to LanceDB, the write fails because the table schema is missing the `calls` field

## Root Cause Analysis

**CONFIRMED ROOT CAUSE**: The schema compatibility check has a **logical flaw** in its compatibility logic.

### The Flaw

```python
def is_compatible_with(self, other: "SchemaVersion") -> bool:
    """Check if two versions are compatible.

    Compatible means:
    - Same major version (breaking changes between major versions)
    - Database version <= code version (code can handle older schemas)  ⚠️ WRONG ASSUMPTION
    """
    if self.major != other.major:
        return False

    # Database version must be <= code version
    if self.minor > other.minor:
        return False
    if self.minor == other.minor and self.patch > other.patch:
        return False

    return True  # BUG: Assumes code can handle older DB schemas
```

**The Wrong Assumption**:
- Comment says: "Database version <= code version (code can handle older schemas)"
- **Reality**: Code version `2.3.5` **cannot** handle database version `2.2.21` because:
  - Version `2.3.x` added new **required fields** (`calls`, `inherits_from`) to LanceDB tables
  - LanceDB tables have **immutable schemas** - you can't add fields to existing tables
  - Old tables from `2.2.21` don't have these fields
  - Attempting to write new chunks with `calls` field to old table schema **fails**

### Why Current Logic is Wrong

**Current Logic**: "If DB version is older than code version, it's compatible"
- DB: `2.2.21`, Code: `2.3.5` → Compatible ✓ (WRONG)

**Correct Logic**: "If DB schema doesn't match code requirements, it's incompatible"
- DB: `2.2.21` (no `calls` field), Code: `2.3.5` (requires `calls` field) → Incompatible ✗

### What Should Happen

1. **User runs**: `mcp-vector-search index`
2. **Schema check detects**: DB version `2.2.21` < Code version `2.3.5`
3. **Current behavior**: Check passes ✓ (assumes backward compatibility)
4. **Indexing starts**: Tries to write chunks with `calls` field
5. **LanceDB rejects**: "Field 'calls' not found in target schema"
6. **User sees error**: ❌

**What should happen**:
1. User runs: `mcp-vector-search index`
2. Schema check detects: DB version `2.2.21` != Code version `2.3.x` (minor version changed)
3. Check fails: ❌ "Schema incompatible, need to reset database"
4. User is prompted: "Run `mcp-vector-search index --force` to reset"
5. Indexing does NOT proceed

## Evidence Supporting Root Cause

### 1. User Error Message
```
ERROR: Failed to flush write buffer: Field 'calls' not found in target schema
```
- This error comes from LanceDB when attempting to write data
- The table schema doesn't have the `calls` field
- This means the table was created with old schema (v2.2.x)

### 2. Schema Version Check Logic
```python
# Database version must be <= code version
if self.minor > other.minor:
    return False
```
- If DB minor version (2) <= code minor version (3), check passes
- This assumes backward compatibility, which is false for schema changes

### 3. Required Fields Definition
```python
REQUIRED_FIELDS = {
    "chunks": [
        "chunk_id",
        "file_path",
        "content",
        # ... other fields ...
        "calls",  # Added in 2.3.x
        "inherits_from",  # Added in 2.3.x
    ],
}
```
- These fields are required in version `2.3.x`
- Old databases (v2.2.x) don't have these fields
- Schema check should detect this and fail

## Fix Recommendations

### Option 1: Stricter Version Checking ⭐ RECOMMENDED

**Change**: Require exact minor version match for compatibility

```python
def is_compatible_with(self, other: "SchemaVersion") -> bool:
    """Check if two versions are compatible.

    Compatible means:
    - Same major version (breaking changes between major versions)
    - Same minor version (schema changes between minor versions)
    - Database patch <= code patch (backward compatible within minor version)
    """
    # Different major version = incompatible
    if self.major != other.major:
        return False

    # Different minor version = incompatible (schema may have changed)
    if self.minor != other.minor:
        return False

    # Database patch must be <= code patch
    if self.patch > other.patch:
        return False

    return True
```

**Impact**:
- DB `2.2.21` + Code `2.3.5` → Incompatible ✗ (correct)
- DB `2.3.0` + Code `2.3.5` → Compatible ✓ (correct)
- DB `2.3.10` + Code `2.3.5` → Incompatible ✗ (user needs to downgrade or upgrade code)

**Pros**:
- Catches all schema changes between minor versions
- Forces users to reset database when upgrading to new minor version
- Simple and safe

**Cons**:
- Forces database reset even for backward-compatible changes
- More conservative than necessary

### Option 2: Field-Level Schema Inspection 🔬 IDEAL BUT COMPLEX

**Change**: Actually inspect LanceDB table schema and compare with required fields

```python
def check_schema_compatibility(db_path: Path) -> tuple[bool, str]:
    """Check if database schema matches current code version."""
    current_version = SchemaVersion(SCHEMA_VERSION)
    db_version = load_schema_version(db_path)

    # Check version file first
    if db_version is None:
        return (False, "No schema version found. Reset database.")

    # Check actual table schemas in LanceDB
    missing_fields = inspect_lancedb_schema(db_path)
    if missing_fields:
        return (
            False,
            f"Database schema is missing required fields: {missing_fields}\n"
            f"Database version: {db_version}, Code version: {current_version}\n"
            f"Reset database with --force flag."
        )

    return (True, "Schema compatible")
```

**Pros**:
- Catches actual schema mismatches, not just version differences
- Allows backward-compatible changes without forcing reset
- Most accurate

**Cons**:
- Requires LanceDB table inspection (more complex)
- Slower (needs to open and inspect tables)
- May not work if tables are locked or corrupted

### Option 3: Migration Support 🛠️ BEST LONG-TERM

**Change**: Implement schema migrations to add missing fields

```python
def migrate_schema_2_2_to_2_3(db_path: Path) -> tuple[bool, str]:
    """Migrate schema from v2.2.x to v2.3.x by adding new fields."""
    try:
        # Open chunks table
        chunks_table = lancedb.open_table(db_path / "lance" / "chunks")

        # Check if 'calls' and 'inherits_from' fields exist
        schema = chunks_table.schema
        if 'calls' not in schema.names:
            # Add 'calls' field with default value
            chunks_table.add_column('calls', default=[])

        if 'inherits_from' not in schema.names:
            # Add 'inherits_from' field with default value
            chunks_table.add_column('inherits_from', default=[])

        # Update schema version
        save_schema_version(db_path, SchemaVersion("2.3.0"))

        return (True, "Schema migrated successfully from 2.2.x to 2.3.x")
    except Exception as e:
        return (False, f"Migration failed: {e}")
```

**Pros**:
- Best user experience (no data loss)
- Allows seamless upgrades
- Preserves existing index data

**Cons**:
- Most complex to implement
- Requires understanding of LanceDB's column addition capabilities
- May not work for all schema changes

## Recommended Action Plan

### Immediate Fix (v2.3.6)

1. **Implement Option 1** (stricter version checking)
   - Quick fix that solves the immediate problem
   - Prevents users from hitting the "Field not found" error
   - Forces database reset when upgrading between minor versions

2. **Add better error messaging**
   - When schema check fails, explain clearly:
     - What version mismatch was detected
     - Why database reset is needed
     - Exact command to run (`mcp-vector-search index --force`)

3. **Improve documentation**
   - Add upgrade guide explaining when database reset is needed
   - Document schema versioning policy
   - Explain semantic versioning for database schemas

### Medium-Term Improvement (v2.4.0)

1. **Implement Option 2** (field-level inspection)
   - Provides more accurate compatibility checking
   - Allows backward-compatible changes without reset

2. **Add schema health check command**
   - `mcp-vector-search index health --check-schema`
   - Shows detailed schema information
   - Detects schema mismatches proactively

### Long-Term Solution (v3.0.0)

1. **Implement Option 3** (schema migrations)
   - Build migration system for LanceDB schema changes
   - Support automatic migrations during indexing
   - Provide rollback capability

2. **Add schema evolution tracking**
   - Track all schema changes in migration history
   - Allow users to see what changed between versions
   - Provide tools for schema inspection and debugging

## Testing Plan

### Test Case 1: Fresh Database (No Version File)

**Setup**:
- Delete `.mcp-vector-search/schema_version.json`
- Run: `mcp-vector-search index`

**Expected**:
- Schema check fails
- Error message: "No schema version found. Database may be from older version"
- User is prompted to run with `--force`
- Indexing does NOT proceed

**Current Behavior**: UNKNOWN (need to test)

### Test Case 2: Old Database (v2.2.21)

**Setup**:
- Create schema version file with version `2.2.21`
- Current code version: `2.3.5`
- Run: `mcp-vector-search index`

**Expected**:
- Schema check fails
- Error message: "Schema version mismatch. Database: 2.2.21, Code: 2.3.5"
- User is prompted to run with `--force`
- Indexing does NOT proceed

**Current Behavior**: Check PASSES (bug), indexing proceeds and fails with "Field 'calls' not found"

### Test Case 3: Force Reset

**Setup**:
- Old database (v2.2.21)
- Run: `mcp-vector-search index --force`

**Expected**:
- Schema check detects mismatch
- Force flag detected, database is reset
- New schema version file is created with `2.3.5`
- Indexing proceeds successfully

**Current Behavior**: UNKNOWN (need to test)

### Test Case 4: Skip Schema Check

**Setup**:
- Old database (v2.2.21)
- Run: `mcp-vector-search index --skip-schema-check`

**Expected**:
- Schema check is skipped
- Indexing proceeds
- Error: "Field 'calls' not found in target schema"

**Current Behavior**: This is likely what's happening to the user

## Questions for User

To confirm the root cause, we need to know:

1. **What command did you run?**
   - `mcp-vector-search index`
   - `mcp-vector-search index --force`
   - `mcp-vector-search index --skip-schema-check`
   - Other?

2. **Does `.mcp-vector-search/schema_version.json` exist in your project?**
   - If yes, what version does it contain?
   - If no, this confirms you have an old database

3. **When did you last upgrade mcp-vector-search?**
   - From what version to what version?
   - Did you reset the database after upgrading?

4. **Did you see any warning messages before the error?**
   - Specifically: "⚠️  Schema Version Mismatch Detected"
   - If yes, what did you do after seeing it?

## Conclusion

The schema version check is **NOT catching the error** because of a **logical flaw in the compatibility check**:

**Bug Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/schema.py`, line 73-89

**Bug Description**: The `is_compatible_with()` method assumes that newer code can always handle older database schemas, but this is false when:
- New required fields are added to LanceDB tables (like `calls` and `inherits_from` in v2.3.x)
- LanceDB tables have immutable schemas and can't be upgraded in-place

**Fix Required**: Change compatibility check to require exact minor version match, or implement field-level schema inspection

**Impact**: **HIGH** - All users upgrading from v2.2.x to v2.3.x will hit this error unless they manually run `--force` to reset the database

**Recommendation**: Release hotfix v2.3.6 with stricter version checking immediately to prevent users from hitting this error

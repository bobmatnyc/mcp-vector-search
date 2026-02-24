# Schema Versioning

## Overview

MCP Vector Search implements schema versioning to prevent users from encountering infinite retry loops or crashes when the database schema doesn't match the code version.

## Problem

When the codebase adds new fields (e.g., `calls`, `inherits_from` in v2.3.x), users with older databases may encounter:

- **Infinite retry loops** - Code tries to access fields that don't exist
- **Silent failures** - Missing fields cause unexpected behavior
- **Data corruption** - Writes fail due to schema mismatch

## Solution

Schema versioning tracks the database schema version and validates compatibility before indexing operations.

### Key Features

1. **Automatic Version Tracking** - Schema version is saved after every successful indexing operation
2. **Pre-Index Validation** - Compatibility check before indexing prevents errors
3. **Clear Error Messages** - Users get actionable guidance when schema mismatch is detected
4. **Graceful Degradation** - Option to skip checks for advanced users

## Architecture

### Schema Version File

Location: `.mcp-vector-search/schema_version.json`

```json
{
  "version": "2.3.4",
  "updated_at": "2024-02-16T19:45:00Z"
}
```

### Version Compatibility Rules

- **Major Version**: Breaking changes (e.g., 1.x.x → 2.x.x)
  - Incompatible - requires full reset
- **Minor Version**: New features/fields (e.g., 2.3.x → 2.4.x)
  - Database version ≤ code version: Compatible (code handles older schemas)
  - Database version > code version: Incompatible (database has features code doesn't understand)
- **Patch Version**: Bug fixes (e.g., 2.3.4 → 2.3.5)
  - Always compatible within same minor version

### Compatibility Check Flow

```
┌─────────────────────────┐
│ User runs: index        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Load schema_version.json│
└────────────┬────────────┘
             │
             ▼
    ┌────────┴────────┐
    │ Version exists? │
    └────────┬────────┘
         No  │  Yes
             │
    ┌────────┴────────┐
    │ Old database    │
    │ (pre-2.3.0)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐      ┌──────────────────┐
    │ Compatible?     │─Yes──▶│ Proceed normally │
    └────────┬────────┘      └──────────────────┘
         No  │
             ▼
    ┌─────────────────┐
    │ Show error      │
    │ & options       │
    └─────────────────┘
```

## Usage

### For Users

**Normal Operation** (compatible database):
```bash
mcp-vector-search index
# ✅ Schema version 2.3.4 is compatible with code version 2.3.4
# Proceeds with indexing
```

**Schema Mismatch** (old database):
```bash
mcp-vector-search index
# ⚠️  Schema Version Mismatch Detected
#
# ❌ Schema version mismatch!
#
# Database version: 2.2.0
# Code version: 2.3.4
#
# Missing fields in 'chunks' table:
#   - calls
#   - inherits_from
#
# Options:
#   1. Reset database and reindex (recommended):
#      mcp-vector-search index --force
#   2. Skip check and continue (may cause errors):
#      mcp-vector-search index --skip-schema-check
```

**Force Reset** (recommended solution):
```bash
mcp-vector-search index --force
# 🔄 Force flag detected - resetting database with new schema...
# ✅ Database reset complete
# 📊 Indexing with schema version 2.3.4
```

**Skip Check** (for advanced users):
```bash
mcp-vector-search index --skip-schema-check
# ⚠️  Schema check skipped - proceeding at your own risk
# (May fail if schema is incompatible)
```

### For Developers

#### Adding New Required Fields

When adding new required fields to the schema:

1. **Update `REQUIRED_FIELDS`** in `core/schema.py`:
```python
REQUIRED_FIELDS = {
    "chunks": [
        # ... existing fields ...
        "new_field",  # Add your new field here
    ],
}
```

2. **Bump Version** appropriately:
   - **Minor version** if backward-compatible (code can handle old schemas)
   - **Major version** if breaking change (old code can't handle new schema)

3. **Update Models** in `core/models.py`:
```python
@dataclass
class CodeChunk:
    # ... existing fields ...
    new_field: str | None = None  # Add with default for compatibility
```

4. **Update Database Backends**:
   - `core/lancedb_backend.py` - Add field to record creation
   - `core/database.py` (ChromaDB) - Add field to metadata

5. **Test Compatibility**:
```bash
pytest tests/test_schema_version.py
```

#### Schema Version Lifecycle

1. **Initialization** (`mcp-vector-search init`):
   - Creates `.mcp-vector-search/schema_version.json`
   - Sets version to current package version

2. **Indexing** (`mcp-vector-search index`):
   - Checks compatibility before indexing
   - Updates version after successful indexing

3. **Database Close**:
   - Saves current schema version
   - Ensures version is updated after operations

## Error Scenarios

### Scenario 1: Old Database (No Version File)

**Cause**: Database created before schema versioning (< v2.3.0)

**Detection**: `schema_version.json` doesn't exist

**Message**:
```
No schema version found. Database may be from older version (< 2.3.0).
Current code version: 2.3.4
Recommendation: Reset database and reindex.
```

**Solution**: `mcp-vector-search index --force`

### Scenario 2: Major Version Mismatch

**Cause**: Upgraded from v1.x.x to v2.x.x

**Detection**: Database major version ≠ code major version

**Message**:
```
❌ Schema version mismatch!

Database version: 1.0.0
Code version: 2.3.4

Major version mismatch - database schema is incompatible.
Recommendation: Reset database with --force flag.
```

**Solution**: `mcp-vector-search index --force`

### Scenario 3: Newer Database

**Cause**: Downgraded package or shared database from newer version

**Detection**: Database version > code version

**Message**:
```
⚠️  Database schema is newer than code!

Database version: 2.4.0
Code version: 2.3.4

Recommendation: Upgrade mcp-vector-search to latest version:
  pip install --upgrade mcp-vector-search
```

**Solution**: `pip install --upgrade mcp-vector-search`

### Scenario 4: Missing Fields (Future Enhancement)

**Cause**: Code requires fields not present in database

**Detection**: Field inspection (not yet implemented)

**Message**:
```
⚠️  Schema mismatch detected!

Missing fields in 'chunks' table:
  - calls
  - inherits_from

Options:
  1. Reset database and reindex (recommended)
  2. Skip check and continue (may cause errors)
```

**Solution**: `mcp-vector-search index --force`

## Implementation Details

### Files

- **`src/mcp_vector_search/core/schema.py`** - Schema versioning logic
- **`src/mcp_vector_search/cli/commands/index.py`** - Pre-index compatibility check
- **`src/mcp_vector_search/cli/commands/init.py`** - Initial version save
- **`src/mcp_vector_search/core/lancedb_backend.py`** - Version save on close
- **`src/mcp_vector_search/core/database.py`** - Version save on close (ChromaDB)
- **`tests/test_schema_version.py`** - Test suite

### Schema Version Structure

```python
class SchemaVersion:
    """Schema version information."""

    def __init__(self, version_str: str) -> None:
        self.major = int(...)  # e.g., 2
        self.minor = int(...)  # e.g., 3
        self.patch = int(...)  # e.g., 4

    def is_compatible_with(self, other: SchemaVersion) -> bool:
        """Check compatibility between versions."""
        # Major version must match
        if self.major != other.major:
            return False

        # Database version must be <= code version
        if self.minor > other.minor:
            return False

        return True
```

### Compatibility Check API

```python
from mcp_vector_search.core.schema import check_schema_compatibility

# Check compatibility
is_compatible, message = check_schema_compatibility(db_path)

if not is_compatible:
    print(message)  # Show user-friendly error message
    # Handle incompatibility (prompt for reset, exit, etc.)
```

### Version Saving

```python
from mcp_vector_search.core.schema import save_schema_version

# Save current schema version
save_schema_version(db_path)

# Save specific version
version = SchemaVersion("2.3.4")
save_schema_version(db_path, version)
```

## Testing

Run the test suite:

```bash
pytest tests/test_schema_version.py -v
```

Test coverage:
- Version parsing and comparison
- Compatibility checks (same, minor upgrade, major mismatch, newer DB)
- File save/load operations
- Error scenarios

## Future Enhancements

### 1. Automatic Migration (Low Priority)

For minor version bumps, attempt automatic schema migration:

```python
def migrate_schema(db_path: Path) -> tuple[bool, str]:
    """Attempt automatic schema migration."""
    # For LanceDB, this is challenging because:
    # - Apache Arrow schemas are immutable
    # - Adding columns requires recreating tables
    #
    # Therefore, reset/reindex is the safest option
    return (False, "Migration requires reset")
```

### 2. Field-Level Detection (Medium Priority)

Inspect actual database schema to detect missing fields:

```python
def get_missing_fields(db_path: Path) -> dict[str, list[str]]:
    """Get fields that exist in code but not in database."""
    # Open LanceDB table
    # Get schema via table.schema
    # Compare with REQUIRED_FIELDS
    return {"chunks": ["calls", "inherits_from"]}
```

### 3. Schema History Tracking (Low Priority)

Track schema changes over time:

```json
{
  "current_version": "2.3.4",
  "history": [
    {"version": "2.2.0", "updated_at": "..."},
    {"version": "2.3.0", "updated_at": "..."},
    {"version": "2.3.4", "updated_at": "..."}
  ]
}
```

## FAQ

**Q: What happens if I delete `schema_version.json`?**

A: The system treats it as an old database (pre-2.3.0) and prompts for reset. Safe to delete if you want to force a reset.

**Q: Can I manually edit `schema_version.json`?**

A: Yes, but not recommended. The file is regenerated after indexing operations.

**Q: What if I want to skip the check every time?**

A: Set environment variable: `export MCP_SKIP_SCHEMA_CHECK=1` (not yet implemented)

**Q: Will this slow down indexing?**

A: No. Version check adds <10ms overhead (single JSON read).

**Q: What about ChromaDB vs LanceDB?**

A: Both backends save schema version on close. The check is database-agnostic.

## Related Issues

- #XX - Add schema version tracking to prevent infinite retry loops
- #YY - Users getting stuck with missing `calls` field

## Changelog

- **v2.3.4** - Initial schema versioning implementation
  - Added `SchemaVersion` class
  - Added compatibility checking before indexing
  - Added `--skip-schema-check` flag
  - Saves version after successful operations

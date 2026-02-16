# Schema Versioning Improvements

This document describes the schema versioning improvements implemented in mcp-vector-search v2.3.x.

## Summary

Two major improvements to schema versioning:

1. **Auto-Reset on Schema Mismatch**: Database automatically resets when schema is incompatible
2. **Separate Schema Version**: Schema version is independent of code version

## Problem Solved

### Before (Issues)
- Schema version tied to package version (e.g., 2.3.7)
- Every package release triggered schema mismatch
- Users had to manually run `--force` flag
- Unnecessary reindexing for non-schema changes

### After (Solutions)
- Schema version (2.3.0) separate from package version (2.3.7)
- Auto-reset on schema mismatch (no manual intervention)
- Schema only bumps when database fields actually change
- Code can update freely without database resets

## Implementation Details

### 1. Auto-Reset on Schema Mismatch

**File**: `/src/mcp_vector_search/cli/commands/index.py` (lines 356-389)

**Behavior**:
```
Schema mismatch detected
  ↓
Show warning with reason
  ↓
Auto-delete old database (shutil.rmtree)
  ↓
Save new schema version
  ↓
Continue indexing
```

**User Experience**:
```bash
# Old behavior (manual intervention required)
$ mcp-vector-search index
⚠️  Schema Version Mismatch Detected
❌ Use --force flag to reset database
$ mcp-vector-search index --force  # User must re-run

# New behavior (automatic)
$ mcp-vector-search index
⚠️  Schema Version Mismatch Detected
🔄 Auto-resetting database for new schema...
✓ Database reset complete, proceeding with indexing...
# Indexing continues automatically
```

### 2. Separate Schema Version

**File**: `/src/mcp_vector_search/core/schema.py` (lines 15-30)

**Schema Version**: `"2.3.0"` (manually maintained)
- Only bumps when database fields change
- Independent of package version

**Schema Changelog**:
```python
SCHEMA_CHANGELOG = {
    "2.3.0": "Added 'calls' and 'inherits_from' fields to chunks table",
    "2.2.0": "Initial schema with basic chunk fields",
}
```

**Schema Compatibility**: Exact version match required
```python
# Old: Backward compatible (database 2.3.0 works with code 2.4.0)
# New: Exact match required (database 2.3.0 only works with schema 2.3.0)

def is_compatible_with(self, other: SchemaVersion) -> bool:
    return (
        self.major == other.major
        and self.minor == other.minor
        and self.patch == other.patch
    )
```

## When to Bump Schema Version

**Bump Schema Version When**:
- Adding new required fields to database schema
- Removing fields from database schema
- Changing field types or constraints
- Incompatible data structure changes

**Don't Bump Schema Version When**:
- Fixing bugs in code logic
- Adding new CLI features
- Improving performance
- Updating documentation
- Refactoring code without schema changes

## Example Scenarios

### Scenario 1: Bug Fix Release (No Schema Change)
```
Package: 2.3.7 → 2.3.8 (bug fix)
Schema:  2.3.0 → 2.3.0 (unchanged)
Result:  No database reset, upgrade seamless
```

### Scenario 2: Schema Change (New Field)
```
Package: 2.3.8 → 2.4.0 (add "complexity_score" field)
Schema:  2.3.0 → 2.4.0 (schema change)
Result:  Auto-reset database, reindex required
Message: "Database schema: 2.3.0 (Added 'calls' and 'inherits_from' fields to chunks table)
          Current schema: 2.4.0 (Added 'complexity_score' field to chunks table)
          Database will be automatically reset."
```

### Scenario 3: First Install (No Existing Database)
```
Package: 2.3.8
Schema:  2.3.0
Result:  Creates new database with schema 2.3.0
```

## Testing

All schema versioning tests pass:
- `tests/test_schema_version.py` (13 tests)
- `tests/test_schema_auto_reset.py` (3 integration tests)

**Test Coverage**:
- Exact version matching
- Auto-reset on mismatch
- No reset when compatible
- Schema version persistence
- Old database migration

## Developer Guide

### How to Add New Schema Version

1. **Update Schema Fields** in `/src/mcp_vector_search/core/schema.py`:
   ```python
   REQUIRED_FIELDS = {
       "chunks": [
           # ... existing fields ...
           "new_field",  # Add new required field
       ],
   }
   ```

2. **Bump Schema Version**:
   ```python
   SCHEMA_VERSION = "2.4.0"  # Increment minor version
   ```

3. **Update Changelog**:
   ```python
   SCHEMA_CHANGELOG = {
       "2.4.0": "Added 'new_field' to chunks table",
       "2.3.0": "Added 'calls' and 'inherits_from' fields to chunks table",
       "2.2.0": "Initial schema with basic chunk fields",
   }
   ```

4. **Test**:
   ```bash
   # Run schema tests
   pytest tests/test_schema_version.py -v
   pytest tests/test_schema_auto_reset.py -v

   # Manual test with old database
   mcp-vector-search index  # Should auto-reset
   ```

5. **Update Package Version** (separate from schema version):
   ```python
   # In /src/mcp_vector_search/__init__.py
   __version__ = "2.4.0"  # Matches schema change for major/minor features
   ```

### Common Pitfalls to Avoid

❌ **Don't**: Bump schema version for every code release
✅ **Do**: Only bump when database structure changes

❌ **Don't**: Use package version as schema version
✅ **Do**: Maintain schema version separately

❌ **Don't**: Allow backward compatibility for schemas
✅ **Do**: Require exact schema version match

## Benefits

1. **User Experience**: No manual `--force` flags, automatic recovery
2. **Developer Experience**: Code can update without schema resets
3. **Maintenance**: Clear separation of concerns (code vs. data)
4. **Debugging**: Schema changelog explains what changed
5. **Testing**: Integration tests verify auto-reset behavior

## Migration Path

Existing users with databases:
- Schema version file missing → Auto-reset on first `mcp-vector-search index`
- Schema version 2.2.x → Auto-reset to 2.3.0
- Schema version 2.3.0 → No reset needed

## Files Changed

- `/src/mcp_vector_search/core/schema.py` (schema versioning logic)
- `/src/mcp_vector_search/cli/commands/index.py` (auto-reset on mismatch)
- `/tests/test_schema_version.py` (updated tests)
- `/tests/test_schema_auto_reset.py` (new integration tests)

## Related Issues

- Fixes infinite retry loops from schema mismatches
- Reduces unnecessary reindexing for code-only updates
- Improves upgrade experience for users

# Research: "Project not initialized" Error Investigation

**Date**: 2026-02-16
**Issue**: User ran `mcp-vector-search index` and got "Project not initialized" error after auto-reset from schema mismatch
**Project**: mcp-vector-search
**Status**: Root cause identified, solution proposed

---

## Executive Summary

The auto-reset feature (added to handle schema version mismatches) deletes the entire `.mcp-vector-search/` directory but does NOT recreate the required `config.json` file. This breaks the initialization check, preventing subsequent indexing operations.

**Root Cause**: `shutil.rmtree(config.index_path)` removes `.mcp-vector-search/` directory entirely, including `config.json`, but only recreates the directory structure—not the configuration file.

**Impact**: Users experience broken state after automatic schema reset, requiring manual `init` command to recover.

---

## 1. Project Initialization Check

### What the Check Looks For

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py:74-79`

```python
def is_initialized(self) -> bool:
    """Check if project is initialized for MCP Vector Search."""
    config_path = get_default_config_path(self.project_root)
    index_path = get_default_index_path(self.project_root)

    return config_path.exists() and index_path.exists()
```

### Required Files/Directories

From `config/defaults.py:406-413`:

1. **Config File**: `{project_root}/.mcp-vector-search/config.json`
2. **Index Directory**: `{project_root}/.mcp-vector-search/`

**Initialization Check**: BOTH must exist to pass `is_initialized()` check.

### Where the Check is Used

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py:349-352`

```python
project_manager = ProjectManager(project_root)

if not project_manager.is_initialized():
    raise ProjectNotFoundError(
        f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
    )
```

This check happens at the start of the `run_indexing()` function, before any indexing work begins.

---

## 2. Auto-Reset Behavior (Schema Mismatch)

### When Auto-Reset Triggers

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py:356-387`

```python
if not skip_schema_check:
    from ...core.schema import check_schema_compatibility, save_schema_version

    db_path = config.index_path
    is_compatible, message = check_schema_compatibility(db_path)

    if not is_compatible:
        print_warning("⚠️  Schema Version Mismatch Detected")
        print_error(message)
        print_warning("\n🔄 Auto-resetting database for new schema...")

        # Automatically reset database on schema mismatch
        import shutil

        if config.index_path.exists():
            try:
                shutil.rmtree(config.index_path)  # ⚠️  DELETES ENTIRE DIRECTORY
                logger.info(f"Removed old database at {config.index_path}")
            except Exception as e:
                logger.error(f"Failed to remove database: {e}")
                raise

        # Ensure index path exists for new database
        config.index_path.mkdir(parents=True, exist_ok=True)  # ✅ Creates directory

        # Save new schema version
        save_schema_version(db_path)  # ✅ Creates schema metadata
        print_info("✓ Database reset complete, proceeding with indexing...")

        # Force reindex after reset
        force_reindex = True
```

### What Gets Deleted

`shutil.rmtree(config.index_path)` removes:
- `.mcp-vector-search/` directory
- `.mcp-vector-search/config.json` ⚠️  **CRITICAL FILE DELETED**
- `.mcp-vector-search/chroma.sqlite3` (vector database)
- `.mcp-vector-search/chunk_index.db` (chunk database)
- `.mcp-vector-search/relationships.json` (knowledge graph data)
- All other files and subdirectories

### What Gets Recreated

After `shutil.rmtree(config.index_path)`:

1. ✅ `config.index_path.mkdir(parents=True, exist_ok=True)` - Recreates directory structure
2. ✅ `save_schema_version(db_path)` - Creates schema version metadata file
3. ❌ **`config.json` NOT recreated** - This breaks `is_initialized()` check

---

## 3. What `mcp-vector-search init` Creates

### Initialization Files and Directories

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py:81-182`

The `initialize()` method creates:

1. **Config Backup** (if `force=True` and config exists):
   ```python
   backup_path = config_path.with_suffix(".json.bak")
   shutil.copy2(config_path, backup_path)
   ```

2. **Index Directory**:
   ```python
   index_path = get_default_index_path(self.project_root)
   index_path.mkdir(parents=True, exist_ok=True)
   ```

3. **Gitignore Entry**:
   ```python
   ensure_gitignore_entry(
       self.project_root,
       pattern=".mcp-vector-search/",
       comment="MCP Vector Search index directory",
   )
   ```

4. **Configuration File** (via `save_config()` at line 166):
   ```python
   config = ProjectConfig(
       project_root=self.project_root,
       index_path=index_path,
       file_extensions=resolved_extensions,
       embedding_model=embedding_model,
       similarity_threshold=similarity_threshold,
       languages=detected_languages,
   )

   self.save_config(config)  # Creates .mcp-vector-search/config.json
   ```

5. **Schema Version Metadata** (from `init.py:199-202`):
   ```python
   from ...core.schema import save_schema_version

   config = project_manager.load_config()
   save_schema_version(config.index_path)
   ```

### Critical Files for Initialization

**Required for `is_initialized()` to return `True`**:
- `.mcp-vector-search/` directory (exists check)
- `.mcp-vector-search/config.json` (exists check)

**Created during initialization but not required for initialization check**:
- `.mcp-vector-search/schema_version.json` (schema metadata)
- `.gitignore` entry for `.mcp-vector-search/`

---

## 4. Is the Init Check Necessary?

### Current Behavior

**Strict Requirement**: The `index` command requires explicit `init` first.

**Justification**:
- Configuration must be established (file extensions, embedding model, similarity threshold)
- Project metadata should be recorded (detected languages, file count)
- User should consciously enable vector search for their project

### Could `index` Work Without Explicit `init`?

**Technically Yes**, with auto-initialization:

**Pros**:
- Better user experience (one-command workflow)
- Prevents broken state after auto-reset
- Aligns with "convention over configuration" philosophy

**Cons**:
- Less explicit about what's being set up
- Loses opportunity to show project detection results
- May surprise users who didn't realize vector search was being initialized

### Design Decision

The initialization check is **intentional design**, not a technical limitation. It:
1. Makes vector search setup explicit and visible
2. Allows users to review detected languages and file types
3. Provides clear point for configuration customization

However, the **auto-reset behavior violates this design** by deleting config without user knowledge.

---

## 5. Root Cause Analysis

### Problem Flow

```
User runs: mcp-vector-search index
    ↓
index command loads config (config.json exists, init check passes)
    ↓
Schema compatibility check runs
    ↓
Schema mismatch detected (e.g., v1 → v2 migration)
    ↓
Auto-reset triggers: shutil.rmtree(config.index_path)
    ↓
    DELETES:
    - .mcp-vector-search/ directory
    - config.json ← CRITICAL LOSS
    - All database files
    ↓
Auto-reset recreates: config.index_path.mkdir(parents=True, exist_ok=True)
    ↓
    CREATES:
    - .mcp-vector-search/ directory
    - schema_version.json (via save_schema_version)
    ↓
    MISSING:
    - config.json ← BREAKS is_initialized()
    ↓
Indexing proceeds (logic bug - should fail here)
    ↓
User runs: mcp-vector-search index (second time)
    ↓
is_initialized() check: config.json missing → False
    ↓
ERROR: "Project not initialized at {path}. Run 'mcp-vector-search init' first."
```

### The Bug

**Location**: Lines 371-384 in `index.py`

**Issue**: Auto-reset deletes config but doesn't recreate it, assuming the config object in memory is sufficient. However:

1. The config object is only in memory for the current run
2. Future commands cannot pass `is_initialized()` without config.json
3. User is left in broken state requiring manual recovery

### Why This Wasn't Caught Earlier

**Timing**: The auto-reset happens AFTER `is_initialized()` check passes:

```python
# Line 349: Check passes because config.json exists
if not project_manager.is_initialized():
    raise ProjectNotFoundError(...)

# Line 354: Load config (reads from config.json)
config = project_manager.load_config()

# Lines 356-384: Schema check → auto-reset → config.json deleted
# But indexing continues using in-memory config object
```

The first run after schema migration SUCCEEDS because:
- Config was loaded into memory before deletion
- Indexing uses the in-memory config object
- No subsequent `is_initialized()` check happens

Only the SECOND run fails because config.json is now missing.

---

## 6. Proposed Solutions

### Solution 1: Recreate Config After Auto-Reset (Recommended)

**Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py:371-384`

**Change**: After `shutil.rmtree()` and `mkdir()`, re-save the config:

```python
if config.index_path.exists():
    try:
        # Store config in memory before deletion
        config_backup = config.model_dump()

        shutil.rmtree(config.index_path)
        logger.info(f"Removed old database at {config.index_path}")
    except Exception as e:
        logger.error(f"Failed to remove database: {e}")
        raise

# Ensure index path exists for new database
config.index_path.mkdir(parents=True, exist_ok=True)

# ✅ RECREATE CONFIG FILE
project_manager.save_config(config)
logger.info("Recreated configuration file after auto-reset")

# Save new schema version
save_schema_version(db_path)
```

**Pros**:
- Minimal code change
- Preserves user's configuration (extensions, embedding model, threshold)
- Maintains initialization state across reset

**Cons**:
- Requires access to `project_manager` in auto-reset code block

---

### Solution 2: Only Delete Database Files, Not Config

**Change**: Instead of `shutil.rmtree(config.index_path)`, selectively delete database files:

```python
if config.index_path.exists():
    try:
        # Delete only database files, preserve config
        db_files = [
            config.index_path / "chroma.sqlite3",
            config.index_path / "chunk_index.db",
            config.index_path / "relationships.json",
            config.index_path / "directory_index.json",
        ]

        for db_file in db_files:
            if db_file.exists():
                db_file.unlink()
                logger.info(f"Removed {db_file.name}")

        # Remove chroma directory if exists
        chroma_dir = config.index_path / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            logger.info("Removed chroma directory")

    except Exception as e:
        logger.error(f"Failed to remove database files: {e}")
        raise
```

**Pros**:
- Preserves config.json (no re-creation needed)
- Cleaner separation: reset affects data, not configuration
- Avoids touching non-database files

**Cons**:
- Must maintain explicit list of database files
- May miss new database files added in future
- Chroma might create unexpected files we don't know about

---

### Solution 3: Auto-Initialize If Config Missing

**Change**: Make `index` command auto-initialize with default config if missing:

```python
project_manager = ProjectManager(project_root)

if not project_manager.is_initialized():
    print_warning("Project not initialized. Auto-initializing with defaults...")

    # Auto-initialize with defaults
    project_manager.initialize(
        file_extensions=DEFAULT_FILE_EXTENSIONS,
        embedding_model=DEFAULT_EMBEDDING_MODELS["code"],
        similarity_threshold=0.5,
        force=False,
    )

    print_success("Project auto-initialized successfully")

config = project_manager.load_config()
```

**Pros**:
- Eliminates "Project not initialized" errors entirely
- Better user experience (no manual recovery needed)
- Handles any case where config is missing, not just auto-reset

**Cons**:
- Changes explicit initialization design philosophy
- User loses visibility into what's being set up
- May auto-initialize with defaults user doesn't want

---

### Solution 4: Warn and Require Re-Init After Auto-Reset

**Change**: After auto-reset, abort indexing and require explicit re-init:

```python
if not is_compatible:
    print_warning("⚠️  Schema Version Mismatch Detected")
    print_error(message)
    print_warning("\n🔄 Database must be reset for new schema")

    import shutil

    if config.index_path.exists():
        try:
            shutil.rmtree(config.index_path)
            logger.info(f"Removed old database at {config.index_path}")
        except Exception as e:
            logger.error(f"Failed to remove database: {e}")
            raise

    print_error("Project has been reset. Please run 'mcp-vector-search init' to reinitialize.")
    print_info("This will recreate configuration and prepare for indexing.")
    raise typer.Exit(1)
```

**Pros**:
- Makes schema reset explicit and visible to user
- User consciously re-initializes (can customize config)
- Clear separation: reset is destructive, init is constructive

**Cons**:
- Worse user experience (requires two commands)
- More disruptive to workflow
- Fails during automated/scripted usage

---

## 7. Recommendation

### Primary Solution: Solution 1 (Recreate Config After Auto-Reset)

**Reasoning**:
1. **Maintains current design**: Explicit initialization philosophy preserved
2. **Minimal disruption**: Auto-reset continues to work seamlessly
3. **Preserves user config**: Extensions, embedding model, threshold maintained
4. **Simple implementation**: 3-4 lines of code

**Implementation Priority**: High (fixes active bug affecting users)

### Secondary Enhancement: Solution 2 (Selective Deletion)

**Reasoning**:
1. **Better separation of concerns**: Config vs. data clearly separated
2. **Safer operation**: Less risk of accidental file deletion
3. **More maintainable**: Explicit about what auto-reset affects

**Implementation Priority**: Medium (improvement but requires more testing)

---

## 8. Testing Checklist

After implementing fix:

### Test Case 1: Auto-Reset Preserves Initialization
```bash
# Setup
cd /tmp/test-project
mcp-vector-search init --extensions .py,.js

# Trigger schema mismatch (simulate by manually editing schema_version.json)
# Edit .mcp-vector-search/schema_version.json → change version

# Run index (should trigger auto-reset)
mcp-vector-search index

# Verify: Run index again (should NOT fail with "not initialized")
mcp-vector-search index
```

**Expected Result**: Second `index` command succeeds without error.

### Test Case 2: Config Preserved After Auto-Reset
```bash
# Setup with custom config
mcp-vector-search init --extensions .py,.md --embedding-model all-mpnet-base-v2

# Trigger auto-reset (schema mismatch)
# ...

# Verify config preserved
cat .mcp-vector-search/config.json | grep "file_extensions"
# Should contain: [".py", ".md"]

cat .mcp-vector-search/config.json | grep "embedding_model"
# Should contain: "all-mpnet-base-v2"
```

**Expected Result**: Custom configuration settings preserved across auto-reset.

### Test Case 3: Database Files Properly Reset
```bash
# Index some files
mcp-vector-search init
mcp-vector-search index

# Record file sizes
ls -lh .mcp-vector-search/

# Trigger auto-reset
# ...

# Verify database files recreated fresh
ls -lh .mcp-vector-search/
# Database files should be much smaller (empty/minimal state)
```

**Expected Result**: Database files reset to fresh state while config.json exists.

---

## 9. Additional Context

### Schema Version System

**Purpose**: Track database schema changes across mcp-vector-search versions.

**Location**: `.mcp-vector-search/schema_version.json`

**Format**:
```json
{
  "version": "2.0.0",
  "created_at": "2025-02-16T10:30:00Z",
  "compatibility": ["2.0.0", "2.1.0"]
}
```

**Compatibility Check**: `check_schema_compatibility()` compares stored version against current codebase expectations.

### When Schema Mismatches Occur

1. **User upgrades mcp-vector-search**: New version expects different database structure
2. **Breaking changes**: Vector dimension changes, metadata schema changes, etc.
3. **Database corruption**: Schema file missing or malformed

### Why Auto-Reset Exists

**Alternative to Manual Migration**:
- Schema migrations are complex and error-prone
- Vector databases are fast to rebuild from source code
- Auto-reset provides smooth upgrade experience

**Trade-off**: Convenience vs. data preservation

---

## 10. Related Files

### Core Files Involved
- `src/mcp_vector_search/cli/commands/index.py` (auto-reset logic)
- `src/mcp_vector_search/core/project.py` (initialization check)
- `src/mcp_vector_search/config/defaults.py` (path helpers)
- `src/mcp_vector_search/core/schema.py` (schema version checking)

### Configuration Files
- `.mcp-vector-search/config.json` (project configuration)
- `.mcp-vector-search/schema_version.json` (schema metadata)

### Database Files (Reset by Auto-Reset)
- `.mcp-vector-search/chroma.sqlite3` (vector embeddings)
- `.mcp-vector-search/chunk_index.db` (code chunks)
- `.mcp-vector-search/relationships.json` (knowledge graph)
- `.mcp-vector-search/directory_index.json` (file metadata)

---

## 11. Next Steps

1. **Implement Solution 1**: Add `project_manager.save_config(config)` after auto-reset
2. **Test thoroughly**: Verify config preservation and initialization state
3. **Consider Solution 2**: Evaluate selective deletion approach for future enhancement
4. **Documentation**: Update migration guide to explain auto-reset behavior
5. **User Communication**: Release notes should mention this fix

---

## Appendix: Error Message Analysis

### Original Error Message
```
ERROR: Indexing failed: Project not initialized at /Users/masa/Clients/Duetto/duetto.
Run 'mcp-vector-search init' first.
```

### Why It's Confusing
- User DID initialize (ran `init` successfully earlier)
- User just ran `index` successfully (first time after schema migration)
- Error appears on SECOND run, not immediately after reset

### Improved Error Message (Optional Enhancement)
```
ERROR: Configuration file missing. This may have occurred due to schema migration reset.

Please run 'mcp-vector-search init' to recreate configuration, or:
  • Check if .mcp-vector-search/config.json exists
  • Verify .mcp-vector-search/ directory permissions

For automated workflows, consider using 'mcp-vector-search init --force' in CI/CD.
```

---

**Research Complete**
**Confidence Level**: High (root cause confirmed through code analysis)
**Solution Feasibility**: High (straightforward implementation)

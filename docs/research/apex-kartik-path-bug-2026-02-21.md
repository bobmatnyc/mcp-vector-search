# APEX Indexing Failure: `/Users/kartik` Permission Error

**Date:** 2026-02-21
**Project:** mcp-vector-search
**Issue:** Indexing APEX project at `/Users/masa/Duetto/repos/APEX` fails with permission error accessing `/Users/kartik`

---

## Problem Summary

The `mcp-vector-search index` command on the APEX project fails with:

```
🔄 Auto-resetting database for new schema...
ERROR: Indexing failed: [Errno 13] Permission denied: '/Users/kartik'
```

The indexer is incorrectly attempting to access `/Users/kartik/duettoresearch/APEX` (another user's home directory) instead of the actual project path `/Users/masa/Duetto/repos/APEX`.

---

## Root Cause

The APEX project's `.mcp-vector-search/` directory contains **stale configuration files** with hardcoded paths from a previous user (`kartik`). When the auto-reset logic attempts to delete and recreate the database, it uses these stale paths instead of the current project location.

---

## Detailed Analysis

### 1. Stale Configuration Files

**File:** `/Users/masa/Duetto/repos/APEX/.mcp-vector-search/config.json`

```json
{
  "project_root": "/Users/kartik/duettoresearch/APEX",
  "index_path": "/Users/kartik/duettoresearch/APEX/.mcp-vector-search",
  ...
}
```

**Issue:** The config contains absolute paths to kartik's system, not masa's.

**File:** `/Users/masa/Duetto/repos/APEX/.mcp-vector-search/index_metadata.json`

```json
{
  "index_version": "2.3.53",
  "indexed_at": "2026-02-17T23:12:59.101704+00:00",
  "file_mtimes": {
    "/Users/kartik/duettoresearch/APEX/CLAUDE.md": 1771275190.6495752,
    "/Users/kartik/duettoresearch/APEX/README.md": 1771304617.6555767,
    ...
  }
}
```

**Issue:** All file paths reference kartik's directory structure.

### 2. Auto-Reset Code Path

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py`

**Lines 771-796:**

```python
db_path = config.index_path  # ← Loads stale path from config.json
is_compatible, message = check_schema_compatibility(db_path)

if not is_compatible:
    print_warning("⚠️  Schema Version Mismatch Detected")
    print_error(message)
    print_warning("\n🔄 Auto-resetting database for new schema...")

    # Automatically reset database on schema mismatch
    import shutil

    if config.index_path.exists():  # ← Uses stale path
        try:
            shutil.rmtree(config.index_path)  # ← Tries to delete /Users/kartik/...
            logger.info(f"Removed old database at {config.index_path}")
        except Exception as e:
            logger.error(f"Failed to remove database: {e}")
            raise
```

**Issue:** The `config.index_path` contains `/Users/kartik/duettoresearch/APEX/.mcp-vector-search`, causing `shutil.rmtree()` to attempt deletion of an inaccessible path.

### 3. Config Loading Logic

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py`

**Lines 191-214:**

```python
def load_config(self) -> ProjectConfig:
    """Load project configuration."""
    ...
    config_path = get_default_config_path(self.project_root)

    try:
        with open(config_path, "rb") as f:
            config_data = orjson.loads(f.read())

        # Convert paths back to Path objects
        config_data["project_root"] = Path(config_data["project_root"])  # ← Loads /Users/kartik/...
        config_data["index_path"] = Path(config_data["index_path"])      # ← Loads /Users/kartik/...
```

**Issue:** The config loader blindly trusts the paths stored in `config.json`, even when they don't match the actual project location.

---

## Why This Happened

**Scenario:**
1. User `kartik` initialized the APEX project on their system at `/Users/kartik/duettoresearch/APEX`
2. The `.mcp-vector-search/` directory was committed to version control (or copied)
3. User `masa` cloned the APEX repo to `/Users/masa/Duetto/repos/APEX`
4. When `masa` runs `mcp-vector-search index`, the tool loads the stale config with kartik's paths
5. Schema version mismatch triggers auto-reset logic
6. Auto-reset tries to delete `/Users/kartik/...` → **Permission denied**

---

## Impact

**Affected Components:**
- `config.json`: Contains stale `project_root` and `index_path`
- `index_metadata.json`: Contains stale file paths in `file_mtimes`
- LanceDB tables (`lance/`, `code_search.lance/`): Contain embedded file paths
- Knowledge graph (`knowledge_graph.new/code_kg`): Binary file with embedded paths

**Failure Mode:**
- Schema upgrade/auto-reset operations fail with permission errors
- Cannot reindex project without manual cleanup
- Error message is confusing (references another user's directory)

---

## Solution

### Immediate Fix (Manual Cleanup)

Delete the entire stale index directory and reinitialize:

```bash
# Remove stale index with hardcoded paths
rm -rf /Users/masa/Duetto/repos/APEX/.mcp-vector-search

# Reinitialize project (creates fresh config with correct paths)
mcp-vector-search init --force --project /Users/masa/Duetto/repos/APEX

# Reindex project
mcp-vector-search index --project /Users/masa/Duetto/repos/APEX
```

**Why this works:**
- Deletes all stale configuration files
- Forces recreation of config.json with correct paths (`/Users/masa/...`)
- Fresh index will use correct paths throughout

### Verification

After cleanup, verify the config contains correct paths:

```bash
cat /Users/masa/Duetto/repos/APEX/.mcp-vector-search/config.json
```

**Expected output:**
```json
{
  "project_root": "/Users/masa/Duetto/repos/APEX",
  "index_path": "/Users/masa/Duetto/repos/APEX/.mcp-vector-search",
  ...
}
```

---

## Recommended Code Fixes

### Fix 1: Path Validation in Config Loader

**File:** `src/mcp_vector_search/core/project.py`

**Current behavior:**
```python
config_data["project_root"] = Path(config_data["project_root"])
config_data["index_path"] = Path(config_data["index_path"])
```

**Proposed fix:**
```python
# Validate that loaded paths match actual project location
loaded_root = Path(config_data["project_root"])
if loaded_root.resolve() != self.project_root.resolve():
    logger.warning(
        f"Config file contains stale path ({loaded_root}). "
        f"Updating to actual project root: {self.project_root}"
    )
    # Override with correct paths
    config_data["project_root"] = self.project_root
    config_data["index_path"] = get_default_index_path(self.project_root)

    # Save corrected config
    config = ProjectConfig(**config_data)
    self.save_config(config)
```

**Benefits:**
- Auto-corrects stale paths without manual intervention
- Prevents permission errors on schema reset
- Maintains backward compatibility

### Fix 2: Better Error Message

**File:** `src/mcp_vector_search/cli/commands/index.py`

**Current behavior:**
```python
except Exception as e:
    logger.error(f"Failed to remove database: {e}")
    raise
```

**Proposed fix:**
```python
except PermissionError as e:
    logger.error(
        f"Failed to remove database at {config.index_path}: {e}\n"
        f"The config file contains a stale path that doesn't match your project location.\n"
        f"Solution: Run 'rm -rf {config.index_path.parent}/.mcp-vector-search && mcp-vector-search init --force'"
    )
    raise
except Exception as e:
    logger.error(f"Failed to remove database: {e}")
    raise
```

**Benefits:**
- Provides actionable error message
- Explains root cause clearly
- Suggests specific fix command

### Fix 3: Prevent Committing Index to Version Control

**File:** `.gitignore` (project-level)

**Add rule:**
```gitignore
# MCP Vector Search index directory
.mcp-vector-search/
```

**Benefits:**
- Prevents stale configs from spreading via version control
- Each user maintains their own index with correct paths
- Reduces repository size (indexes can be large)

**Note:** The tool already attempts to add this rule automatically during `init` (via `ensure_gitignore_entry`), but this may fail if `.gitignore` doesn't exist or lacks write permissions.

---

## Testing Verification

### Test Case 1: Stale Config Detection

1. Initialize project at path A
2. Copy `.mcp-vector-search/` to path B
3. Run indexer at path B
4. **Expected:** Auto-correction warning + successful indexing
5. **Current:** Permission error (if paths don't exist)

### Test Case 2: Schema Reset with Stale Config

1. Initialize project at path A
2. Copy `.mcp-vector-search/` to path B
3. Upgrade tool version (trigger schema mismatch)
4. Run indexer at path B
5. **Expected:** Auto-correction + successful reset
6. **Current:** Permission error on `shutil.rmtree()`

---

## Related Issues

**Symptom:** "Permission denied" during indexing
**Root Cause:** Stale absolute paths in config files
**Trigger:** Schema version mismatch + auto-reset logic
**Affected Files:**
- `config.json`
- `index_metadata.json`
- LanceDB tables (binary)
- Knowledge graph (binary)

**Similar Issues:**
- Users moving projects between machines
- Projects shared via cloud sync (Dropbox, iCloud)
- Docker containers with bind-mounted volumes
- CI/CD pipelines with ephemeral paths

---

## Action Items

**Immediate (User):**
- [ ] Delete `/Users/masa/Duetto/repos/APEX/.mcp-vector-search`
- [ ] Run `mcp-vector-search init --force --project /Users/masa/Duetto/repos/APEX`
- [ ] Run `mcp-vector-search index --project /Users/masa/Duetto/repos/APEX`
- [ ] Verify `config.json` contains correct paths

**Short-term (Development):**
- [ ] Implement path validation in `load_config()` (Fix 1)
- [ ] Improve error messages for permission errors (Fix 2)
- [ ] Add tests for stale config detection
- [ ] Document `.gitignore` best practices in README

**Long-term (Development):**
- [ ] Consider using relative paths in config (relative to project root)
- [ ] Add config migration tool for schema upgrades
- [ ] Implement config health check command
- [ ] Add warnings for mismatched paths during init/index

---

## Conclusion

The permission error is caused by stale absolute paths in the APEX project's `.mcp-vector-search/config.json` file. The immediate fix is to delete the stale index directory and reinitialize. The long-term fix requires implementing path validation in the config loader to auto-correct mismatched paths.

**Key Takeaway:** Never commit `.mcp-vector-search/` to version control. Each user/machine should maintain their own index with machine-specific absolute paths.

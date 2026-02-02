# System Directory Scanning Bug Investigation

**Date:** 2026-01-31
**Issue:** mcp-vector-search setup scanning system directories like `/Users/mattrosenberg/Library/VoiceTrigger/SAT`
**Error:** `[Errno 1] Operation not permitted`

## Executive Summary

The `mcp-vector-search setup` command is attempting to scan system directories outside the project root, causing permission errors on macOS. This occurs because `Path.rglob()` follows symlinks by default, and the code does not skip symlinks during directory traversal.

## Root Cause Analysis

### Problem Locations

1. **File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py`
   - **Line:** 327
   - **Method:** `_iter_source_files()`
   - **Issue:** Uses `self.project_root.rglob("*")` which follows symlinks

2. **File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/setup.py`
   - **Line:** 234
   - **Method:** `scan_project_file_extensions()`
   - **Issue:** Uses `project_root.rglob("*")` which follows symlinks

### Code Analysis

#### Current Implementation (Broken)

**`project.py:327` in `_iter_source_files()`:**
```python
def _iter_source_files(self) -> list[Path]:
    """Iterate over source files in the project."""
    files = []

    for path in self.project_root.rglob("*"):  # ❌ Follows symlinks
        if not path.is_file():
            continue

        if self._should_ignore_path(path, is_directory=False):
            continue

        files.append(path)

    return files
```

**`setup.py:234` in `scan_project_file_extensions()`:**
```python
for path in project_root.rglob("*"):  # ❌ Follows symlinks
    # Check timeout
    if time.time() - start_time > timeout:
        logger.debug(f"File extension scan timed out...")
        return None

    if not path.is_file():
        continue

    if project_manager._should_ignore_path(path, is_directory=False):
        continue
```

### Why This Happens

1. **`Path.rglob()` behavior:** Python's `Path.rglob()` follows symlinks by default (unlike `os.walk()` which has a `followlinks` parameter)
2. **Symlink scenarios:**
   - User has symlinks in their project pointing to external directories
   - Project root itself is accessed via a symlink
   - Home directory or parent directories contain symlinks
3. **macOS protected directories:** System Integrity Protection (SIP) prevents access to directories like:
   - `/Users/*/Library/VoiceTrigger/`
   - `/System/Library/`
   - Other protected system paths

### Error Flow

```
setup.py:938 → detect_languages()
    ↓
project.py:256 → _iter_source_files()
    ↓
project.py:327 → project_root.rglob("*")
    ↓
    Follows symlink → /Users/mattrosenberg/Library/VoiceTrigger/SAT
    ↓
    Attempts to access → PermissionError: [Errno 1] Operation not permitted
    ↓
setup.py:898 → Exception caught
    ↓
setup.py:899 → ERROR: Unexpected error during setup
```

## Evidence

### Error Message
```
🔍 Detecting project...
ℹ    Detecting languages...
ERROR: Unexpected error during setup: [Errno 1] Operation not permitted: '/Users/mattrosenberg/Library/VoiceTrigger/SAT'
```

### Affected Functions

1. **`detect_languages()`** - Called from `setup.py:939`
2. **`scan_project_file_extensions()`** - Called from `setup.py:951`
3. **`_iter_source_files()`** - Helper method used by detection

### Call Stack

```
setup_command()                          # setup.py:904
└─> _run_smart_setup()                   # setup.py:922
    ├─> project_manager.detect_languages()   # setup.py:939
    │   └─> _iter_source_files()             # project.py:256
    │       └─> project_root.rglob("*")      # project.py:327 ❌
    │
    └─> scan_project_file_extensions()       # setup.py:951
        └─> project_root.rglob("*")          # setup.py:234 ❌
```

## Recommended Fixes

### Fix 1: Skip Symlinks (Recommended)

Add symlink detection before processing paths:

**`project.py` - Fix `_iter_source_files()`:**
```python
def _iter_source_files(self) -> list[Path]:
    """Iterate over source files in the project."""
    files = []

    for path in self.project_root.rglob("*"):
        # Skip symlinks to prevent traversing outside project
        if path.is_symlink():
            continue

        if not path.is_file():
            continue

        if self._should_ignore_path(path, is_directory=False):
            continue

        files.append(path)

    return files
```

**`setup.py` - Fix `scan_project_file_extensions()`:**
```python
for path in project_root.rglob("*"):
    # Check timeout
    if time.time() - start_time > timeout:
        logger.debug(f"File extension scan timed out...")
        return None

    # Skip symlinks to prevent traversing outside project
    if path.is_symlink():
        continue

    if not path.is_file():
        continue

    if project_manager._should_ignore_path(path, is_directory=False):
        continue
```

### Fix 2: Add Symlink Check to `_should_ignore_path()`

Centralize symlink handling in the ignore logic:

**`project.py` - Update `_should_ignore_path()`:**
```python
def _should_ignore_path(self, path: Path, is_directory: bool | None = None) -> bool:
    """Check if a path should be ignored.

    Args:
        path: Path to check
        is_directory: Optional hint if path is a directory (avoids filesystem check)

    Returns:
        True if path should be ignored
    """
    # Skip symlinks to prevent traversing outside project boundaries
    if path.is_symlink():
        return True

    # First check gitignore rules if available
    if self.gitignore_parser and self.gitignore_parser.is_ignored(
        path, is_directory=is_directory
    ):
        return True

    # ... rest of existing logic
```

### Fix 3: Add Permission Error Handling

Wrap filesystem operations in try-except for better error recovery:

**`project.py` - Add error handling:**
```python
def _iter_source_files(self) -> list[Path]:
    """Iterate over source files in the project."""
    files = []

    try:
        for path in self.project_root.rglob("*"):
            # Skip symlinks
            if path.is_symlink():
                continue

            # Catch permission errors on individual files
            try:
                if not path.is_file():
                    continue
            except PermissionError:
                logger.debug(f"Permission denied accessing: {path}")
                continue

            if self._should_ignore_path(path, is_directory=False):
                continue

            files.append(path)
    except PermissionError as e:
        logger.warning(f"Permission error during file iteration: {e}")

    return files
```

## Priority Recommendation

**Use Fix 1** - Skip symlinks explicitly in both locations:
- **Pros:**
  - Minimal code change
  - Clear and explicit
  - Prevents the issue at the source
  - No performance impact
- **Cons:**
  - Need to update two locations
  - Won't index symlinked files (but this is safer behavior)

**Alternative:** Combine Fix 1 and Fix 2 for comprehensive protection:
- Add symlink check in `_should_ignore_path()` (centralized)
- Keep explicit checks in both `rglob()` loops (defense in depth)
- Add permission error handling (graceful degradation)

## Testing Recommendations

### Test Cases

1. **Symlink in project directory:**
   ```bash
   ln -s ~/Library test-project/external-link
   mcp-vector-search setup
   # Should skip symlink, not traverse into ~/Library
   ```

2. **Project root is symlink:**
   ```bash
   ln -s /real/project /tmp/symlink-project
   cd /tmp/symlink-project
   mcp-vector-search setup
   # Should work correctly
   ```

3. **Symlink to code file (valid use case):**
   ```bash
   ln -s ~/shared/utils.py project/utils.py
   mcp-vector-search setup
   # Should either:
   # - Skip the symlink (safer), OR
   # - Follow but only if target is readable
   ```

4. **Protected system directory:**
   ```bash
   # Test on macOS with SIP-protected directories
   ln -s ~/Library/VoiceTrigger project/trigger-link
   mcp-vector-search setup
   # Should skip symlink, complete setup successfully
   ```

## Additional Findings

### Other `rglob()` Usage in Codebase

The following files also use `rglob()` and may need review:

1. **`src/mcp_vector_search/utils/monorepo.py:233`**
   - Context: `package.json` detection in monorepos
   - Risk: Medium (could follow symlinks in node_modules)

2. **`src/mcp_vector_search/analysis/entry_points.py:168`**
   - Context: Finding Python files
   - Risk: Medium (could traverse into virtual environments)

3. **`src/mcp_vector_search/cli/commands/analyze.py:863`**
   - Context: File analysis
   - Risk: Medium (could access restricted directories)

4. **`src/mcp_vector_search/cli/commands/chat.py:939`**
   - Context: File search for chat context
   - Risk: Low (already filtered by extension)

5. **`src/mcp_vector_search/core/corruption_recovery.py:336`**
   - Context: Scanning persist directory
   - Risk: Low (controlled directory, not user input)

6. **`src/mcp_vector_search/migrations/v1_2_2_codexembed.py:191`**
   - Context: Finding parquet files
   - Risk: Low (controlled directory)

### Recommendation for Other Files

Apply symlink checks to:
- **High Priority:** `monorepo.py`, `entry_points.py`, `analyze.py`
- **Medium Priority:** `chat.py`
- **Low Priority:** `corruption_recovery.py`, migrations (controlled paths)

## Implementation Plan

### Phase 1: Critical Fix (Immediate)
1. Add symlink checks to `project.py:_iter_source_files()`
2. Add symlink checks to `setup.py:scan_project_file_extensions()`
3. Test on macOS with symlinks to protected directories
4. Release as patch version

### Phase 2: Comprehensive Fix (Next Release)
1. Add symlink check to `_should_ignore_path()` method
2. Review and fix other `rglob()` usage in codebase
3. Add permission error handling where appropriate
4. Add integration tests for symlink scenarios

### Phase 3: Documentation (Follow-up)
1. Document symlink behavior in README
2. Add warning about symlinks in troubleshooting guide
3. Update setup command documentation

## Related Issues

- Permission errors on macOS with SIP-protected directories
- Potential slow setup times if traversing large external directories
- Risk of indexing unintended files from symlinked locations

## Impact Assessment

### Severity: **HIGH**
- Blocks setup command on affected systems
- User cannot initialize mcp-vector-search
- Affects macOS users with certain directory structures

### Scope: **MEDIUM**
- Affects users with symlinks in project directories
- Affects macOS users with SIP-protected paths
- Does not affect users without symlinks

### Urgency: **HIGH**
- Complete blocker for affected users
- Simple fix available
- Should be patched immediately

## Conclusion

The root cause is clear: `Path.rglob()` follows symlinks and accesses directories outside the project root. The fix is straightforward: skip symlinks during traversal. This should be implemented in both `_iter_source_files()` and `scan_project_file_extensions()`, with optional centralization in `_should_ignore_path()` for consistency.

**Recommended Action:** Implement Fix 1 immediately as a patch release, then consider Fix 2 for long-term maintenance.

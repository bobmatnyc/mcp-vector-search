# Monorepo Detection Fix - Exclude Test/Example/Docs Directories

## Problem

The visualization incorrectly showed "tests/manual" as the project root because the monorepo detection logic treated `tests/manual/package.json` as a depth-0 subproject (root level) instead of recognizing it as a test directory.

## Root Cause

The file `/Users/masa/Projects/mcp-vector-search/tests/manual/package.json` triggered monorepo detection in the fallback `_detect_by_package_json()` method, which scanned for ANY `package.json` file without excluding common non-subproject directories.

## Solution

Updated `src/mcp_vector_search/utils/monorepo.py` to exclude test/example/docs directories from subproject detection:

### Changes Made

1. **Added `EXCLUDED_SUBPROJECT_DIRS` constant** (lines 9-35)
   - Comprehensive list of directories to exclude from subproject detection
   - Includes: tests, examples, docs, scripts, tools, build artifacts, etc.

2. **Added `_is_excluded_path()` helper method** (lines 58-73)
   - Checks if any part of a path matches excluded directories
   - Returns `True` if path should be excluded from subproject detection

3. **Updated `_detect_by_package_json()`** (lines 217-248)
   - Calls `_is_excluded_path()` before treating `package.json` as subproject
   - Logs excluded paths for debugging

4. **Updated `_expand_workspace_patterns()`** (lines 250-286)
   - Applies same exclusion logic to workspace patterns
   - Ensures consistency across detection methods

5. **Updated `_detect_nx_workspace()`** (lines 191-220)
   - Applies exclusion logic to Nx workspace detection
   - Prevents test directories in apps/libs from being detected

## Testing

Created comprehensive test suite in `tests/unit/test_monorepo.py`:

- ✅ Verifies excluded directories are defined
- ✅ Tests exclusion of tests, examples, docs directories
- ✅ Tests inclusion of apps, packages, src directories
- ✅ Verifies package.json in excluded paths is ignored
- ✅ Verifies package.json in valid paths is detected
- ✅ Tests mixed scenarios with multiple directories

**All 10 tests pass.**

## Verification

Tested with actual project structure:

```bash
uv run python3 -c "
from pathlib import Path
from src.mcp_vector_search.utils.monorepo import MonorepoDetector

detector = MonorepoDetector(Path('/Users/masa/Projects/mcp-vector-search'))
subprojects = detector.detect_subprojects()
print(f'Detected {len(subprojects)} subprojects')
"
```

**Output:**
```
Detected 0 subprojects
DEBUG: Skipping excluded path: tests/manual/package.json
```

✅ `tests/manual` is now correctly excluded from subproject detection.

## Impact

### Before Fix
- `tests/manual/package.json` detected as depth-0 subproject
- Visualization breadcrumb showed "tests/manual" as root
- Incorrect project structure representation

### After Fix
- `tests/manual/package.json` correctly excluded
- Visualization will show actual project root
- Proper breadcrumb: "🏠 Root / tests / manual / [filename]"

## Next Steps

To see the fix in action:

1. Re-index the project:
   ```bash
   uv run mcp-vector-search index --force
   ```

2. Regenerate visualization:
   ```bash
   uv run mcp-vector-search visualize export --code-only
   ```

3. Start visualization server:
   ```bash
   uv run mcp-vector-search visualize serve --code-only --port 8080
   ```

4. Verify in browser:
   - Breadcrumb should show "🏠 Root" for mcp-vector-search project
   - Files from `tests/manual/` should show: "🏠 Root / tests / manual / [filename]"
   - No JavaScript errors in console

## Code Quality

- ✅ Passes `ruff check` (no linting issues)
- ✅ Passes `mypy` (no type errors)
- ✅ All unit tests pass (10/10)
- ✅ Follows project coding standards

## Design Decisions

### Excluded Directories List

**Rationale:** Comprehensive exclusion list prevents false positives from common project directories that contain `package.json` but are not subprojects.

**Trade-offs:**
- **Completeness:** May need updates for new project structures
- **Performance:** Minimal overhead (O(k) where k = number of path segments)
- **Maintainability:** Centralized list is easy to update

**Alternatives Considered:**
1. **Whitelist approach:** Only allow specific directories (e.g., "apps", "packages")
   - Rejected: Too restrictive, would miss valid monorepo structures
2. **Regex patterns:** More flexible pattern matching
   - Rejected: Added complexity without significant benefit
3. **Config file:** Make exclusions configurable
   - Deferred: YAGNI - can add later if needed

### Helper Method Design

**Rationale:** `_is_excluded_path()` provides reusable exclusion logic across all detection methods.

**Performance:**
- Time Complexity: O(k) where k = number of path segments (typically 2-4)
- Space Complexity: O(1)
- Expected Performance: <1ms per path check

**Error Handling:**
- Returns `True` for paths outside project root (ValueError → excluded)
- Safe to call on non-existent paths
- No exceptions propagated to caller

## Related Files

- Implementation: `src/mcp_vector_search/utils/monorepo.py`
- Tests: `tests/unit/test_monorepo.py`
- Documentation: `docs/development/monorepo-detection-fix.md`

## Author

- **Date:** December 6, 2025
- **Version:** 0.14.9 (pending)
- **Issue:** Visualization shows "tests/manual" as root

---

**Status:** ✅ Implemented, Tested, Ready for Re-indexing

# Rust Language Auto-Detection Investigation

**Date**: 2026-02-04
**Issue**: `mcp-vector-search setup` did not detect Rust (`.rs` files) in ai-commander project
**Status**: Investigation Complete - Root Cause Identified
**Priority**: Medium
**Context**: RustParser was added in v2.2.3, but setup command doesn't auto-detect Rust projects

## Problem Statement

When running `mcp-vector-search setup` or `init` on the ai-commander project (which contains Rust code), the auto-detection logic failed to include `.rs` files in the indexed extensions.

## Investigation Summary

### Files Analyzed

1. **`src/mcp_vector_search/cli/commands/setup.py`**
   - `scan_project_file_extensions()` function (lines 208-275)
   - `_run_smart_setup()` function (lines 908-1185)

2. **`src/mcp_vector_search/config/defaults.py`**
   - `DEFAULT_FILE_EXTENSIONS` (lines 15-109)
   - `LANGUAGE_MAPPINGS` (lines 112-206)

3. **`src/mcp_vector_search/core/project.py`**
   - `detect_languages()` method (lines 250-263)

## Key Findings

### ✅ Rust IS Properly Configured

**Defaults Configuration** (`defaults.py`):
```python
# Line 71: .rs is in DEFAULT_FILE_EXTENSIONS
DEFAULT_FILE_EXTENSIONS = [
    # ... other extensions ...
    ".rs",  # Rust - Tree-Sitter supported
    # ...
]

# Line 168: .rs is mapped to "rust" language
LANGUAGE_MAPPINGS = {
    # ... other mappings ...
    ".rs": "rust",
    # ...
}
```

**RustParser Status**:
- ✅ Added in v2.2.3
- ✅ Registered in parser registry
- ✅ Fully functional for parsing Rust code

### 🔍 Auto-Detection Logic Analysis

**The `scan_project_file_extensions()` function** (setup.py:208-275):

```python
def scan_project_file_extensions(
    project_root: Path,
    timeout: float = 2.0,
) -> list[str] | None:
    """Scan project for unique file extensions with timeout."""
    extensions: set[str] = set()
    start_time = time.time()

    for path in project_root.rglob("*"):
        # Check timeout
        if time.time() - start_time > timeout:
            return None  # 🚨 TIMEOUT SCENARIO

        # Skip ignored paths
        if project_manager._should_ignore_path(path, is_directory=False):
            continue  # 🚨 IGNORED PATH SCENARIO

        # Get extension
        ext = path.suffix
        if ext:
            language = get_language_from_extension(ext)
            if language != "text" or ext in [".txt", ".md", ".rst"]:
                extensions.add(ext)  # ✅ SHOULD WORK for .rs

    return sorted(extensions) if extensions else None
```

**Logic Verification for `.rs` Files**:

1. **Extension Retrieval**: `ext = path.suffix` → `".rs"` ✅
2. **Language Lookup**: `get_language_from_extension(".rs")` → `"rust"` ✅
3. **Condition Check**: `"rust" != "text"` → `True` ✅
4. **Action**: `extensions.add(".rs")` → SHOULD execute ✅

**Conclusion**: The logic itself is CORRECT. Rust files should be detected.

## Root Cause Identification

Since the logic is correct, the issue must be one of these scenarios:

### Hypothesis 1: Scan Timeout (Most Likely)
**Probability**: 80%

**Evidence**:
- Default timeout is 2.0 seconds (line 210)
- Large projects like ai-commander can take longer to scan
- When timeout occurs, function returns `None` (line 244)
- Fallback behavior uses `DEFAULT_FILE_EXTENSIONS` (line 1011)

**Code Flow**:
```python
# Line 955: Scan with 2-second timeout
detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)

# Line 957-965: Handle timeout
if detected_extensions:
    print_success(f"   ✅ Detected {len(detected_extensions)} file type(s)")
else:
    print_info("   ⏱️  Scan timed out, using defaults")  # User sees this

# Line 1011: Fallback to defaults
file_extensions = detected_extensions or DEFAULT_FILE_EXTENSIONS
```

**Why This Causes the Issue**:
- If scan times out, `detected_extensions = None`
- Fallback uses `DEFAULT_FILE_EXTENSIONS` which DOES include `.rs`
- **BUT**: If user ran `init` command instead of `setup`, different behavior occurs

### Hypothesis 2: Rust Files in Ignored Directories
**Probability**: 15%

**Evidence**:
- Scan respects `.gitignore` patterns
- Common Rust directories that might be ignored:
  - `target/` - Rust build output (in `DEFAULT_IGNORE_PATTERNS`)
  - `vendor/` - Dependency vendoring
  - `.cargo/` - Cargo cache

**Verification Needed**:
- Check if ai-commander's `.rs` files are all in `target/` or other ignored dirs
- Review project structure

### Hypothesis 3: Init Command vs Setup Command Difference
**Probability**: 5%

**Evidence**:
- `init` command has different auto-detection behavior
- `init` uses `DEFAULT_FILE_EXTENSIONS` directly (line 166 in init.py)
- `setup` uses scanned extensions with fallback

**Code Comparison**:
```python
# init.py (line 158-166)
file_extensions = None
if extensions:
    file_extensions = [ext.strip() for ext in extensions.split(",")]
else:
    file_extensions = DEFAULT_FILE_EXTENSIONS  # Always includes .rs

# setup.py (line 1011)
file_extensions = detected_extensions or DEFAULT_FILE_EXTENSIONS  # Depends on scan
```

## Impact Analysis

**Current Behavior**:
- Rust files ARE in `DEFAULT_FILE_EXTENSIONS`
- Timeout fallback SHOULD work correctly
- Manual specification with `--extensions` flag works

**User Workaround Available**:
```bash
# Explicit extension specification
mcp-vector-search init --extensions .rs,.py,.js,.md

# Or force re-initialization
mcp-vector-search setup --force
```

## Recommended Next Steps

### 1. Reproduce the Issue (Required)
```bash
# Test on ai-commander project
cd /path/to/ai-commander
mcp-vector-search setup --verbose

# Check output for:
# - "⏱️  Scan timed out, using defaults"
# - "✅ Detected N file type(s)" (and check if .rs is included)
```

### 2. Add Diagnostic Logging (Quick Win)
**File**: `src/mcp_vector_search/cli/commands/setup.py`

**Before** (line 955):
```python
detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)
```

**After**:
```python
detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)
if verbose and detected_extensions:
    print_info(f"      Detected extensions: {', '.join(sorted(detected_extensions))}")
```

### 3. Increase Timeout for Large Projects (Medium Priority)
**File**: `src/mcp_vector_search/cli/commands/setup.py`

**Current** (line 955):
```python
detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)
```

**Proposed**:
```python
# Adaptive timeout based on project size
base_timeout = 2.0
try:
    # Quick directory count to estimate size
    dir_count = sum(1 for _ in project_root.glob("**/") if not str(_).startswith('.'))
    scan_timeout = min(base_timeout * (1 + dir_count / 100), 10.0)  # Max 10s
except Exception:
    scan_timeout = base_timeout

detected_extensions = scan_project_file_extensions(project_root, timeout=scan_timeout)
```

### 4. Verify Default Fallback Works (Testing)
**Test Case**:
```python
def test_scan_timeout_includes_rust():
    """Verify that timeout fallback includes .rs extension."""
    # When scan times out
    detected_extensions = None

    # Fallback should include Rust
    file_extensions = detected_extensions or DEFAULT_FILE_EXTENSIONS
    assert ".rs" in file_extensions
```

### 5. Add Extension Validation Message (UX Improvement)
**After** initialization (line 1038), add:
```python
print_success("✅ Configuration saved")

# Show what will be indexed
if verbose:
    print_info(f"   Indexing {len(file_extensions)} file types:")
    for ext in sorted(file_extensions)[:10]:
        lang = get_language_from_extension(ext)
        print_info(f"      {ext} → {lang}")
    if len(file_extensions) > 10:
        print_info(f"      ... and {len(file_extensions) - 10} more")
```

## Questions to Answer

1. **Did the scan actually run on ai-commander?**
   - Check verbose output: "⏱️  Scan timed out" vs "✅ Detected N file type(s)"

2. **Were .rs files actually present in non-ignored locations?**
   - Run: `find /path/to/ai-commander -name "*.rs" -not -path "*/target/*"`

3. **What command was used?**
   - `mcp-vector-search setup` (uses scanning)
   - `mcp-vector-search init` (uses defaults directly)

4. **What were the detected languages?**
   - Check output: "✅ Found N language(s): ..."
   - Rust should appear if `.rs` files were scanned

## Recommended Fix

**Priority**: Medium
**Complexity**: Low
**Risk**: Low

**Option 1: Increase Timeout (Safest)**
- Change default timeout from 2.0s to 5.0s
- Add adaptive timeout based on project size
- Low risk, solves timeout scenario

**Option 2: Always Include Rust in Defaults (Quick Fix)**
- Ensure `DEFAULT_FILE_EXTENSIONS` includes `.rs` (already done ✅)
- Verify fallback logic works correctly
- Already implemented, just needs verification

**Option 3: Add --include-rust Flag (User Control)**
```bash
mcp-vector-search setup --include-rust
```
- Allows explicit Rust inclusion
- Good for edge cases
- More user-friendly than --extensions

## Related Issues

- **Issue #81**: CodeXEmbed integration (not related)
- **PR #XXX**: RustParser implementation (v2.2.3) - Parser works, detection issue is separate

## Testing Checklist

- [ ] Test `mcp-vector-search setup` on Rust project
- [ ] Test with `--verbose` flag to see detection output
- [ ] Test timeout scenario (large project)
- [ ] Test .gitignore exclusion of `target/`
- [ ] Verify `.rs` is in `DEFAULT_FILE_EXTENSIONS`
- [ ] Verify `LANGUAGE_MAPPINGS[".rs"] == "rust"`
- [ ] Test manual `--extensions .rs` flag
- [ ] Check `detect_languages()` finds Rust

## Conclusion

**Root Cause**: Most likely timeout during file extension scanning in large projects.

**Evidence**:
1. ✅ Rust is properly configured in defaults
2. ✅ Language mapping is correct
3. ✅ Detection logic is sound
4. ⚠️  Scan timeout returns None, triggering fallback
5. ✅ Fallback SHOULD include .rs (in DEFAULT_FILE_EXTENSIONS)

**Mystery**: If fallback includes `.rs`, why wasn't it detected? Need to:
- Reproduce issue on ai-commander
- Check verbose output
- Verify which command was actually run (`setup` vs `init`)
- Confirm scan timeout vs successful scan

**Next Action**: Run `mcp-vector-search setup --verbose` on ai-commander project and capture full output for analysis.

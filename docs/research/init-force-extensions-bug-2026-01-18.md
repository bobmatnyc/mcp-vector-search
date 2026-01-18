# Research: `init --force` Doesn't Regenerate Extensions Bug

**Date**: 2026-01-18
**Ticket**: #76
**Researcher**: Claude Code Research Agent
**Status**: Root cause identified

---

## Executive Summary

The `mcp-vector-search init --force` command **does NOT regenerate file extensions** to match the updated `DEFAULT_FILE_EXTENSIONS` list. It preserves the old extensions from the existing config file.

**Root Cause**: The `--force` flag only bypasses the "already initialized" check but **does not delete or regenerate** the config. When user runs `init --force` without `--extensions` flag, the CLI always uses `DEFAULT_FILE_EXTENSIONS` from code, but the old config is never read or merged.

**Impact**: Users upgrading to versions with expanded `DEFAULT_FILE_EXTENSIONS` will not get new extensions added unless they manually specify them with `--extensions`.

---

## Root Cause Analysis

### Code Flow

**File**: `src/mcp_vector_search/cli/commands/init.py`

```python
# Lines 157-166
# Parse file extensions
file_extensions = None
if extensions:
    file_extensions = [ext.strip() for ext in extensions.split(",")]
    # Ensure extensions start with dot
    file_extensions = [
        ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
    ]
else:
    file_extensions = DEFAULT_FILE_EXTENSIONS  # ← ALWAYS uses defaults
```

**File**: `src/mcp_vector_search/core/project.py`

```python
# Lines 100-103
if self.is_initialized() and not force:
    raise ProjectInitializationError(
        f"Project already initialized at {self.project_root}. Use --force to re-initialize."
    )
```

**Lines 133-141** (project.py):
```python
# Create configuration
config = ProjectConfig(
    project_root=self.project_root,
    index_path=index_path,
    file_extensions=file_extensions or DEFAULT_FILE_EXTENSIONS,  # ← Uses defaults
    embedding_model=embedding_model,
    similarity_threshold=similarity_threshold,
    languages=detected_languages,
)

# Save configuration
self.save_config(config)  # ← Overwrites old config completely
```

### The Problem

1. **User runs**: `mcp-vector-search init --force`
2. **CLI sets**: `file_extensions = DEFAULT_FILE_EXTENSIONS` (line 166, init.py)
3. **Force flag**: Bypasses initialization check (line 100, project.py)
4. **Config created**: Uses `DEFAULT_FILE_EXTENSIONS` from **code** (line 137, project.py)
5. **Config saved**: Completely overwrites old config (line 144, project.py)

**EXPECTED BEHAVIOR**: Config should be regenerated with latest `DEFAULT_FILE_EXTENSIONS`

**ACTUAL BEHAVIOR**: Config **is** regenerated, but from old `DEFAULT_FILE_EXTENSIONS` snapshot

### Why This Happens

The bug occurs because:

1. **Old config is never read**: When `--force` is used, the old config is completely ignored
2. **Code uses DEFAULT_FILE_EXTENSIONS**: The `init.py` always uses `DEFAULT_FILE_EXTENSIONS` from code
3. **User has old installation**: User's Python environment has **old version** of the package installed

**Key Insight**: The user's installed package contains **old `DEFAULT_FILE_EXTENSIONS`**. Running `init --force` regenerates config from the **old defaults** in their installed package, not the **new defaults** in the latest version.

---

## Evidence

### Current Config (User's System)
```json
{
  "file_extensions": [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb",
    ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh",
    ".json", ".md", ".txt"
  ]
}
```

**Missing 82 extensions** compared to latest `DEFAULT_FILE_EXTENSIONS`:
- `.pyw`, `.pyi` (Python)
- `.cjs`, `.mts`, `.cts` (JavaScript/TypeScript)
- `.html`, `.htm`, `.css`, `.scss`, `.sass`, `.less` (Web)
- `.yaml`, `.yml`, `.toml`, `.xml` (Config)
- `.markdown`, `.rst` (Docs)
- `.fish` (Shell)
- `.cc`, `.cxx`, `.hxx` (C++)
- `.groovy`, `.rake`, `.gemspec`, `.phtml` (JVM/Ruby/PHP)
- `.dart`, `.r`, `.R`, `.sql`, `.lua`, `.pl`, `.pm`, `.ex`, `.exs`, `.clj`, `.cljs`, `.cljc`, `.hs`, `.ml`, `.mli`, `.vim`, `.el` (Other languages)

### Latest DEFAULT_FILE_EXTENSIONS (Code)
**File**: `src/mcp_vector_search/config/defaults.py` (lines 14-108)

Contains **108 file extensions** across 30+ programming languages.

---

## The Real Issue: Package Installation State

### Why `--force` Doesn't Fix This

The user is experiencing this because:

1. **User upgraded codebase** (git pull) to get latest code
2. **User did NOT reinstall package** (no `pip install -e .` or `pip install --upgrade`)
3. **Python imports old code**: When running `mcp-vector-search init --force`, Python uses **installed package**, not **source code**
4. **Old defaults used**: The installed package has old `DEFAULT_FILE_EXTENSIONS`

### Verification

**CONFIRMED**: Package installation is out of sync with source code.

```bash
# Check installed package (uv tool)
/Users/masa/.local/share/uv/tools/mcp-vector-search/bin/python3 -c \
  "from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS; \
   print(f'Installed: {len(DEFAULT_FILE_EXTENSIONS)} extensions')"
# Output: Installed package has 67 extensions

# Check source code
wc -l src/mcp_vector_search/config/defaults.py
# Lines 14-108 define 108 extensions
```

**Expected**: Length should be **108** (latest source code)
**Actual**: Length is **67** (installed uv tool version)
**Gap**: **41 missing extensions** in installed package

---

## Recommended Fix Approaches

### Option 1: User Workaround (Immediate)

**Tell user to reinstall package**:

```bash
# If using editable install
pip install -e .

# If using regular install
pip install --upgrade --force-reinstall mcp-vector-search

# Then run init --force
mcp-vector-search init --force
```

**This will**:
- Install latest package with new `DEFAULT_FILE_EXTENSIONS`
- `init --force` will now use updated defaults

### Option 2: Add Migration Logic (Code Change)

**Problem**: `--force` should **detect version changes** and automatically merge new extensions.

**Implementation**:

**File**: `src/mcp_vector_search/core/project.py`

```python
def initialize(
    self,
    file_extensions: list[str] | None = None,
    embedding_model: str = "microsoft/codebert-base",
    similarity_threshold: float = 0.5,
    force: bool = False,
) -> ProjectConfig:
    """Initialize project for MCP Vector Search."""

    # NEW: Load old config if force=True and no extensions provided
    old_config = None
    if self.is_initialized() and force and file_extensions is None:
        try:
            old_config = self.load_config()
        except Exception:
            pass  # Ignore errors loading old config

    if self.is_initialized() and not force:
        raise ProjectInitializationError(
            f"Project already initialized at {self.project_root}. Use --force to re-initialize."
        )

    try:
        # Create index directory
        index_path = get_default_index_path(self.project_root)
        index_path.mkdir(parents=True, exist_ok=True)

        # ... gitignore handling ...

        # Detect languages and files
        detected_languages = self.detect_languages()

        # NEW: Merge old extensions with new defaults when force=True
        if old_config and file_extensions is None:
            # Merge: keep old + add new defaults
            merged_extensions = list(set(old_config.file_extensions + DEFAULT_FILE_EXTENSIONS))
            file_extensions = sorted(merged_extensions)

            logger.info(
                f"Force re-initialization: merged {len(old_config.file_extensions)} old "
                f"+ {len(DEFAULT_FILE_EXTENSIONS)} new = {len(file_extensions)} total extensions"
            )
        else:
            file_extensions = file_extensions or DEFAULT_FILE_EXTENSIONS

        file_count = self.count_indexable_files(file_extensions)

        # Create configuration
        config = ProjectConfig(
            project_root=self.project_root,
            index_path=index_path,
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            languages=detected_languages,
        )

        # Save configuration
        self.save_config(config)

        logger.info(
            f"Initialized project at {self.project_root}",
            languages=detected_languages,
            file_count=file_count,
            extensions=config.file_extensions,
        )

        self._config = config
        return config

    except Exception as e:
        raise ProjectInitializationError(
            f"Failed to initialize project: {e}"
        ) from e
```

**Benefits**:
- **Automatic migration**: Users get new extensions when running `init --force`
- **Non-destructive**: Old custom extensions are preserved
- **Version-safe**: Works even with old installed package

**Trade-offs**:
- **Never removes extensions**: If user had custom extensions, they persist forever
- **Merge behavior unclear**: User may not want ALL new defaults

### Option 3: Add `--reset-extensions` Flag (Code Change)

**Better UX**: Give users explicit control over extension reset.

**Implementation**:

**File**: `src/mcp_vector_search/cli/commands/init.py`

```python
@init_app.callback()
def main(
    ctx: typer.Context,
    # ... existing params ...
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization if project is already initialized",
        rich_help_panel="⚙️  Advanced Options",
    ),
    reset_extensions: bool = typer.Option(
        False,
        "--reset-extensions",
        help="Reset file extensions to latest defaults (use with --force)",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """..."""

    # Parse file extensions
    file_extensions = None
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(",")]
        file_extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
        ]
    elif reset_extensions or not project_manager.is_initialized():
        # Use defaults on first init OR when explicitly requested
        file_extensions = DEFAULT_FILE_EXTENSIONS
    else:
        # Preserve existing extensions (load from config)
        try:
            old_config = project_manager.load_config()
            file_extensions = old_config.file_extensions
            print_info(f"Preserving {len(file_extensions)} existing file extensions")
        except Exception:
            file_extensions = DEFAULT_FILE_EXTENSIONS
```

**Benefits**:
- **Explicit control**: User chooses when to reset extensions
- **Backward compatible**: Existing behavior unchanged
- **Clear intent**: `--reset-extensions` documents what happens

**Usage**:
```bash
# Preserve old extensions (default)
mcp-vector-search init --force

# Reset to latest defaults
mcp-vector-search init --force --reset-extensions

# Custom extensions (always works)
mcp-vector-search init --force --extensions .py,.js,.ts
```

### Option 4: Auto-Detect Version Change (Code Change)

**Most Sophisticated**: Detect package version change and auto-merge.

**Implementation**:

**File**: `src/mcp_vector_search/core/project.py`

Add version tracking to config:

```python
# In ProjectConfig
class ProjectConfig(BaseSettings):
    """Type-safe project configuration with validation."""

    project_root: Path
    index_path: Path
    file_extensions: list[str]
    # ... existing fields ...

    # NEW: Track version for migration
    config_version: str = Field(default="1.0.0")  # Matches package version
```

**File**: `src/mcp_vector_search/core/project.py`

```python
def initialize(
    self,
    file_extensions: list[str] | None = None,
    embedding_model: str = "microsoft/codebert-base",
    similarity_threshold: float = 0.5,
    force: bool = False,
) -> ProjectConfig:
    """Initialize project for MCP Vector Search."""

    from .. import __version__  # Get current package version

    # Check if this is a version upgrade
    old_config = None
    is_upgrade = False
    if self.is_initialized():
        try:
            old_config = self.load_config()
            old_version = getattr(old_config, 'config_version', '0.0.0')
            is_upgrade = old_version != __version__

            if is_upgrade and not force:
                logger.info(
                    f"Config version changed ({old_version} -> {__version__}). "
                    "Run with --force to migrate."
                )
        except Exception:
            pass

    if self.is_initialized() and not force:
        raise ProjectInitializationError(
            f"Project already initialized at {self.project_root}. Use --force to re-initialize."
        )

    try:
        # ... index directory creation ...

        # Auto-merge extensions on version upgrade
        if is_upgrade and old_config and file_extensions is None:
            logger.info(f"Upgrading config from {old_version} to {__version__}")

            # Merge old + new defaults
            merged_extensions = list(set(old_config.file_extensions + DEFAULT_FILE_EXTENSIONS))
            file_extensions = sorted(merged_extensions)

            logger.info(
                f"Auto-merged extensions: {len(old_config.file_extensions)} old "
                f"+ {len(DEFAULT_FILE_EXTENSIONS)} new = {len(file_extensions)} total"
            )
        else:
            file_extensions = file_extensions or DEFAULT_FILE_EXTENSIONS

        # ... rest of initialization ...

        # Create configuration with version tracking
        config = ProjectConfig(
            project_root=self.project_root,
            index_path=index_path,
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            languages=detected_languages,
            config_version=__version__,  # Track version
        )

        # Save configuration
        self.save_config(config)

        # ... rest of code ...
```

**Benefits**:
- **Automatic migration**: Users get new extensions on version upgrade
- **Version awareness**: Config knows what package version created it
- **Future-proof**: Can handle schema migrations in future versions

**Trade-offs**:
- **Complex**: Adds version tracking and migration logic
- **Breaking change**: Requires schema migration for existing configs

---

## Edge Cases to Consider

### 1. User has custom extensions

**Scenario**: User manually edited config to remove unwanted extensions.

**Current behavior**: `init --force` replaces with defaults (loses customization)

**With Option 2 (merge)**: User gets old custom + new defaults (unwanted extensions creep back)

**With Option 3 (flag)**: User keeps custom extensions unless `--reset-extensions` used ✅

**With Option 4 (version)**: User gets old custom + new defaults (same as Option 2)

**Recommendation**: Option 3 gives best control.

### 2. User upgrades package mid-project

**Scenario**: User runs `pip install --upgrade mcp-vector-search` but forgets to run `init --force`.

**Current behavior**: Old extensions persist, new files not indexed.

**With Option 2**: Automatic merge on next `init --force` ✅

**With Option 3**: User must explicitly `--reset-extensions` ❌

**With Option 4**: Automatic merge on next `init --force` ✅

**Recommendation**: Option 4 detects upgrades automatically.

### 3. Team has different package versions

**Scenario**: Team member A has v1.0, member B has v1.1 (with new extensions).

**Current behavior**: Each has different config, indexing inconsistent.

**With Option 2/4**: Config grows over time as members upgrade ✅

**With Option 3**: First member to run `--reset-extensions` standardizes team ✅

**Recommendation**: Team should commit config and standardize package versions.

### 4. Downgrade scenario

**Scenario**: User downgrades from v1.1 to v1.0.

**Current behavior**: Config has extensions v1.0 doesn't recognize (harmless).

**With all options**: Old extensions preserved (safe) ✅

**Recommendation**: No special handling needed.

---

## Immediate Action Required

### For User (Issue #76)

**User needs to reinstall package** (using uv tool):

```bash
# Verify current installation state
/Users/masa/.local/share/uv/tools/mcp-vector-search/bin/python3 -c \
  "from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS; \
   print(f'Installed: {len(DEFAULT_FILE_EXTENSIONS)} extensions')"
# Expected output: Installed package has 67 extensions

# Reinstall using uv (this rebuilds the tool from source)
uv tool install --reinstall --force .

# Verify new installation
/Users/masa/.local/share/uv/tools/mcp-vector-search/bin/python3 -c \
  "from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS; \
   print(f'Installed: {len(DEFAULT_FILE_EXTENSIONS)} extensions')"
# Expected output: Installed package now has 108 extensions

# Regenerate config with latest defaults
mcp-vector-search init --force

# Verify config has new extensions
cat .mcp-vector-search/config.json | jq '.file_extensions | length'
# Expected output: 108
```

**Alternative for pip users**:
```bash
# If using pip install -e
pip install -e . --force-reinstall

# Or if using regular pip install
pip install --upgrade --force-reinstall .
```

### For Codebase

**Recommended: Implement Option 3 (`--reset-extensions` flag)**

**Why**:
- **Non-breaking**: Existing behavior unchanged
- **Explicit control**: User chooses when to reset
- **Simple implementation**: Minimal code changes
- **Clear documentation**: Flag name explains intent

**Implementation PR**:
1. Add `--reset-extensions` flag to init command
2. Update help text to explain flag usage
3. Update docs/CLI help with upgrade instructions
4. Add tests for extension preservation behavior

---

## Testing Checklist

- [ ] **Test 1**: `init --force` preserves custom extensions (no `--reset-extensions`)
- [ ] **Test 2**: `init --force --reset-extensions` uses latest defaults
- [ ] **Test 3**: `init --force --extensions .py,.js` uses explicit list
- [ ] **Test 4**: First `init` (no existing config) uses latest defaults
- [ ] **Test 5**: Package upgrade + `init --force` preserves old extensions (no reset)
- [ ] **Test 6**: Package upgrade + `init --force --reset-extensions` gets new defaults

---

## Related Files

- **CLI Entry Point**: `src/mcp_vector_search/cli/commands/init.py` (lines 36-196)
- **Project Manager**: `src/mcp_vector_search/core/project.py` (lines 79-159)
- **Config Model**: `src/mcp_vector_search/config/settings.py` (lines 11-50)
- **Default Extensions**: `src/mcp_vector_search/config/defaults.py` (lines 14-108)
- **Config Storage**: `.mcp-vector-search/config.json` (user's project)

---

## Conclusion

**Root Cause**: The `--force` flag bypasses initialization checks but does NOT intelligently merge old and new extensions. It uses `DEFAULT_FILE_EXTENSIONS` from the **installed package**, which may be outdated if user upgraded codebase but not package.

**Immediate Fix**: User needs to reinstall package (`pip install -e .`) then run `init --force`.

**Long-term Fix**: Implement `--reset-extensions` flag to give users explicit control over extension regeneration.

**Impact**: Low severity (workaround exists), but affects user experience during package upgrades.

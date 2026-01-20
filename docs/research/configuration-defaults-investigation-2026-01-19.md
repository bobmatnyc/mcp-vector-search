# Configuration Defaults Investigation

**Date:** 2026-01-19
**Investigator:** Research Agent
**Purpose:** Locate default configuration values for `respect_gitignore` and directory exclude patterns

---

## Executive Summary

This investigation identified all locations where default configuration values are defined, particularly for `respect_gitignore` and directory exclude patterns. The system uses a multi-layered approach:

1. **Default values** defined in `/src/mcp_vector_search/config/defaults.py`
2. **Schema/validation** in `/src/mcp_vector_search/config/settings.py`
3. **Initialization logic** in `/src/mcp_vector_search/cli/commands/init.py`
4. **Usage in indexer** at `/src/mcp_vector_search/core/indexer.py`

---

## Key Findings

### 1. `respect_gitignore` Default Value

**Current Default:** `True`

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`

```python
# Line 50-53
respect_gitignore: bool = Field(
    default=True,
    description="Respect .gitignore patterns when indexing files",
)
```

**Purpose:** When `True`, the indexer checks `.gitignore` patterns and excludes matching files/directories.

**Usage in Indexer:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`

```python
# Lines 213-219 (Initialization)
# Initialize gitignore parser (only if respect_gitignore is True)
if config is None or config.respect_gitignore:
    try:
        self.gitignore_parser = create_gitignore_parser(project_root)
        logger.debug(
            f"Loaded {len(self.gitignore_parser.patterns)} gitignore patterns"
        )

# Lines 1170-1176 (File filtering)
# 2. Check gitignore rules if available and enabled
# PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
if self.config and self.config.respect_gitignore:
    if self.gitignore_parser and self.gitignore_parser.is_ignored(
        file_path, is_directory=is_directory
    ):
        logger.debug(f"Path ignored by .gitignore: {file_path}")
        return True
```

---

### 2. Directory Exclude Patterns (DEFAULT_IGNORE_PATTERNS)

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`

**Lines:** 248-295

```python
# Directories to ignore during indexing
DEFAULT_IGNORE_PATTERNS = [
    # Version control
    ".git",
    ".hg",
    ".svn",
    # Python caches and environments
    "__pycache__",
    ".hypothesis",  # Hypothesis property-based testing
    ".mypy_cache",  # mypy type checking cache
    ".nox",  # Nox test automation
    ".pytest_cache",
    ".ruff_cache",  # ruff linter cache
    ".tox",  # Tox testing environments
    ".venv",
    "venv",
    # JavaScript/Node.js
    ".npm",  # npm cache
    ".nyc_output",  # Istanbul/nyc coverage
    ".yarn",  # Yarn cache
    "bower_components",
    "coverage",  # Jest/Mocha coverage reports
    "node_modules",
    # Build outputs
    "_build",  # Sphinx and other doc builders
    "build",
    "dist",
    "htmlcov",  # Python coverage HTML reports
    "site",  # MkDocs and other static site builders
    "target",
    "wheels",  # Python wheel build artifacts
    # Generic caches
    ".cache",
    # IDEs and editors
    ".idea",
    ".vscode",
    # Environment and config
    ".env",
    # Build artifacts and packages
    "*.egg-info",
    "vendor",  # Dependency vendoring
    # OS files
    ".DS_Store",
    "Thumbs.db",
    # Tool-specific directories
    ".claude-mpm",  # Claude MPM directory
    ".mcp-vector-search",  # Our own index directory
]
```

**Notable Exclusions:**
- Version control directories (`.git`, `.hg`, `.svn`)
- Python virtual environments (`.venv`, `venv`)
- Node.js dependencies (`node_modules`)
- Build outputs (`build`, `dist`, `target`)
- Cache directories (`__pycache__`, `.cache`, `.mypy_cache`)
- IDE directories (`.idea`, `.vscode`)

---

### 3. Allowed Dotfiles (Whitelisted)

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`

**Lines:** 5-10

```python
# Dotfiles that should NEVER be skipped (CI/CD configurations)
ALLOWED_DOTFILES = {
    ".github",  # GitHub workflows/actions
    ".gitlab-ci",  # GitLab CI
    ".circleci",  # CircleCI config
}
```

**Purpose:** CI/CD configuration directories that should always be indexed despite starting with `.`

---

### 4. File Ignore Patterns (DEFAULT_IGNORE_FILES)

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`

**Lines:** 297-343

```python
# File patterns to ignore
DEFAULT_IGNORE_FILES = [
    # Python compiled and build artifacts
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.egg",  # Python egg files
    "*.whl",  # Python wheel files
    ".coverage",  # Python coverage data file
    "pip-wheel-metadata",  # pip wheel metadata
    # Native libraries and executables
    "*.a",
    "*.bin",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.lib",
    "*.o",
    "*.obj",
    "*.so",
    # Java archives
    "*.ear",
    "*.jar",
    "*.war",
    # Archive files
    "*.7z",
    "*.bz2",
    "*.gz",
    "*.iso",
    "*.rar",
    "*.tar",
    "*.xz",
    "*.zip",
    # Disk images
    "*.dmg",
    "*.img",
    # Editor swap and temporary files
    "*.sublime-*",  # Sublime Text
    "*.swo",  # Vim swap files
    "*.swp",  # Vim swap files
    # Temporary and cache files
    "*.cache",
    "*.lock",
    "*.log",
    "*.temp",
    "*.tmp",
]
```

---

### 5. Init Command Configuration

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py`

**Key Points:**

**Default File Extensions** (Line 166):
```python
else:
    file_extensions = DEFAULT_FILE_EXTENSIONS
```

**Configuration Defaults** (Lines 191-196):
```python
project_manager.initialize(
    file_extensions=file_extensions,
    embedding_model=embedding_model,
    similarity_threshold=similarity_threshold,
    force=force,
)
```

**No explicit `respect_gitignore` parameter** - relies on `ProjectConfig` default value (`True`)

---

### 6. ProjectManager Initialization

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py`

**Method:** `initialize()` (Lines 79-175)

**Creates ProjectConfig** (Lines 149-156):
```python
# Create configuration
config = ProjectConfig(
    project_root=self.project_root,
    index_path=index_path,
    file_extensions=resolved_extensions,
    embedding_model=embedding_model,
    similarity_threshold=similarity_threshold,
    languages=detected_languages,
)
```

**Note:** `respect_gitignore` is NOT explicitly set here, so it uses the Pydantic Field default (`True`)

**Configuration Persistence** (Lines 209-233):
```python
def save_config(self, config: ProjectConfig) -> None:
    """Save project configuration."""
    config_path = get_default_config_path(self.project_root)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert to JSON-serializable format
        config_data = config.model_dump()
        config_data["project_root"] = str(config.project_root)
        config_data["index_path"] = str(config.index_path)

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
```

**Saved to:** `{project_root}/.mcp-vector-search/config.json`

---

### 7. Usage in SemanticIndexer

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`

**Initialization** (Lines 213-219):
```python
# Initialize gitignore parser (only if respect_gitignore is True)
if config is None or config.respect_gitignore:
    try:
        self.gitignore_parser = create_gitignore_parser(project_root)
        logger.debug(
            f"Loaded {len(self.gitignore_parser.patterns)} gitignore patterns"
        )
```

**Ignore Patterns Setup** (Line 182):
```python
self._ignore_patterns = set(DEFAULT_IGNORE_PATTERNS)
```

**File Filtering** (Lines 1170-1176):
```python
# 2. Check gitignore rules if available and enabled
# PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
if self.config and self.config.respect_gitignore:
    if self.gitignore_parser and self.gitignore_parser.is_ignored(
        file_path, is_directory=is_directory
    ):
        logger.debug(f"Path ignored by .gitignore: {file_path}")
        return True
```

---

## Architecture Summary

### Configuration Flow

```
1. DEFAULT VALUES
   ├─ defaults.py
   │  ├─ DEFAULT_IGNORE_PATTERNS (list of directories)
   │  ├─ DEFAULT_IGNORE_FILES (list of file patterns)
   │  ├─ ALLOWED_DOTFILES (whitelist for .github, .gitlab-ci, .circleci)
   │  └─ DEFAULT_FILE_EXTENSIONS (supported file types)
   │
   └─ settings.py
      └─ ProjectConfig.respect_gitignore = True (Field default)

2. INITIALIZATION (init command)
   ├─ cli/commands/init.py
   │  └─ Calls project_manager.initialize()
   │
   └─ core/project.py
      ├─ Creates ProjectConfig with defaults
      └─ Saves to .mcp-vector-search/config.json

3. RUNTIME USAGE
   └─ core/indexer.py
      ├─ Loads config from .mcp-vector-search/config.json
      ├─ If respect_gitignore=True: Creates gitignore parser
      ├─ Always uses DEFAULT_IGNORE_PATTERNS
      └─ Filters files during indexing
```

### Filtering Layers

**The indexer applies multiple filtering layers:**

1. **DEFAULT_IGNORE_PATTERNS** (hardcoded directory list)
   - Always active
   - Excludes `.git`, `node_modules`, `.venv`, etc.

2. **`.gitignore` patterns** (if `respect_gitignore=True`)
   - Optional (controlled by config)
   - Parses `.gitignore` files in project
   - Default: **ENABLED** (`True`)

3. **`skip_dotfiles`** (if enabled)
   - Controlled by `ProjectConfig.skip_dotfiles`
   - Default: `True`
   - Whitelists: `.github`, `.gitlab-ci`, `.circleci`

---

## Files to Modify for Default Changes

### To Change `respect_gitignore` Default

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`
**Line:** 50-53

```python
# Change from:
respect_gitignore: bool = Field(
    default=True,  # ← CHANGE THIS
    description="Respect .gitignore patterns when indexing files",
)

# To:
respect_gitignore: bool = Field(
    default=False,  # New default
    description="Respect .gitignore patterns when indexing files",
)
```

---

### To Change Default Exclude Patterns

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`
**Lines:** 248-295

**Add new patterns to `DEFAULT_IGNORE_PATTERNS` list:**

```python
DEFAULT_IGNORE_PATTERNS = [
    # Version control
    ".git",
    ".hg",
    ".svn",
    # ... existing patterns ...

    # ADD NEW PATTERNS HERE:
    "new_directory_to_exclude",
    "another_directory",
]
```

**Or remove existing patterns** (e.g., to index `.vscode`):

```python
DEFAULT_IGNORE_PATTERNS = [
    # ... other patterns ...
    # ".vscode",  # COMMENTED OUT - now will be indexed
]
```

---

## Related Configuration Files

### 1. Skip Dotfiles Configuration

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`
**Lines:** 46-49

```python
skip_dotfiles: bool = Field(
    default=True,
    description="Skip files and directories starting with '.' (except whitelisted ones)",
)
```

---

### 2. Auto-Reindex on Upgrade

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`
**Lines:** 42-45

```python
auto_reindex_on_upgrade: bool = Field(
    default=True,
    description="Automatically reindex when tool version is upgraded (minor/major versions)",
)
```

---

## Testing Strategy

### To Test Default Value Changes

1. **Change default in source code** (e.g., `respect_gitignore=False`)

2. **Initialize new project:**
   ```bash
   cd /tmp/test-project
   mcp-vector-search init
   ```

3. **Verify config file:**
   ```bash
   cat .mcp-vector-search/config.json | jq '.respect_gitignore'
   # Should output: false (if default was changed to False)
   ```

4. **Test with `--force` (re-initialization):**
   ```bash
   mcp-vector-search init --force
   cat .mcp-vector-search/config.json | jq '.respect_gitignore'
   # Should pick up new default
   ```

---

### To Test Exclude Pattern Changes

1. **Modify `DEFAULT_IGNORE_PATTERNS`** (add or remove patterns)

2. **Create test structure:**
   ```bash
   mkdir -p /tmp/test-project/{.vscode,node_modules,src}
   touch /tmp/test-project/.vscode/settings.json
   touch /tmp/test-project/node_modules/package.json
   touch /tmp/test-project/src/main.py
   ```

3. **Initialize and index:**
   ```bash
   cd /tmp/test-project
   mcp-vector-search init
   mcp-vector-search index
   ```

4. **Verify indexed files:**
   ```bash
   mcp-vector-search status
   # Check which files were indexed vs. excluded
   ```

---

## Recommendations

### For Changing `respect_gitignore` Default

**Current Behavior:** `.gitignore` rules are respected by default (`True`)

**Impact of Changing to `False`:**
- Projects **without explicit configuration** will index gitignored files
- Existing projects (with saved config) will NOT be affected
- Users must run `mcp-vector-search init --force` to update existing projects

**Recommendation:**
- If changing default, document in CHANGELOG.md
- Provide migration guide for existing users
- Consider deprecation warning period

---

### For Modifying Exclude Patterns

**Current Behavior:** Hardcoded list excludes common directories

**Considerations:**
- `DEFAULT_IGNORE_PATTERNS` is **always active** (not configurable)
- Adding patterns: Low risk (more exclusions)
- Removing patterns: **High risk** (may index large directories like `node_modules`)

**Recommendation:**
- Make exclude patterns **configurable** in `ProjectConfig`
- Add field: `exclude_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))`
- Allow users to override defaults via CLI or config file

---

## Summary Table

| Setting | Current Default | File | Line Numbers | Impact |
|---------|----------------|------|--------------|--------|
| `respect_gitignore` | `True` | `config/settings.py` | 50-53 | HIGH - Controls gitignore filtering |
| `DEFAULT_IGNORE_PATTERNS` | List of 30+ dirs | `config/defaults.py` | 248-295 | HIGH - Always excludes these directories |
| `ALLOWED_DOTFILES` | `.github`, `.gitlab-ci`, `.circleci` | `config/defaults.py` | 5-10 | MEDIUM - Whitelists CI/CD directories |
| `skip_dotfiles` | `True` | `config/settings.py` | 46-49 | MEDIUM - Controls dotfile filtering |
| `DEFAULT_IGNORE_FILES` | List of 30+ patterns | `config/defaults.py` | 297-343 | LOW - File pattern exclusions |

---

## Conclusion

The configuration system uses a **two-tier approach**:

1. **Hard-coded defaults** in `config/defaults.py` (always active)
2. **Configurable settings** in `config/settings.py` (saved per-project)

To change defaults:
- **For `respect_gitignore`**: Modify `settings.py` line 51
- **For exclude patterns**: Modify `defaults.py` lines 248-295

**Critical Note:** Changing defaults only affects **NEW** project initializations. Existing projects use saved configuration in `.mcp-vector-search/config.json` unless re-initialized with `--force`.

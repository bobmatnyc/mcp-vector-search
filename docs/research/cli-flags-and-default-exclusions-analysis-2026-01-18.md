# CLI Flags and Default Exclusions Analysis

**Research Date:** 2026-01-18
**Researcher:** Claude (Research Agent)
**Project:** mcp-vector-search
**Focus:** CLI interface analysis, default exclusion patterns, and gap identification

---

## Executive Summary

Analyzed the `mcp-vector-search init` CLI flags, default exclusion patterns, and supported languages to identify gaps in current defaults and provide accurate CLI usage documentation.

**Key Findings:**
- ✅ CLI flag for file extensions: `--extensions` or `-e` (comma-separated)
- ✅ Currently excludes: 13 directories, 26 file patterns
- ⚠️ **Gap:** Missing common patterns like `.tox/`, `coverage/`, `.cache/`, `htmlcov/`
- ✅ Supports: 8 languages with full parsers + fallback for all others

---

## 1. CLI Flags for `mcp-vector-search init`

### File Extension Configuration

**Flag:** `--extensions` or `-e`
**Format:** Comma-separated list of extensions
**Auto-correction:** Automatically adds leading dot if missing

**Examples:**
```bash
# Correct syntax
mcp-vector-search init --extensions .py,.js,.ts,.txt,.md

# Also works (auto-corrects to add dots)
mcp-vector-search init -e py,js,ts,txt,md

# Custom subset for Python-only projects
mcp-vector-search init -e .py,.pyi,.pyx

# TypeScript/JavaScript projects
mcp-vector-search init -e .ts,.tsx,.js,.jsx,.mjs
```

**Default Extensions (when not specified):**
```python
[".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".java", ".cpp", ".c",
 ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt",
 ".scala", ".sh", ".bash", ".zsh", ".json", ".md", ".txt"]
# Total: 25 extensions
```

### Other Init Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config` | `-c` | None | Configuration file to use |
| `--embedding-model` | `-m` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `--similarity-threshold` | `-s` | 0.5 | Search result threshold (0.0-1.0) |
| `--force` | `-f` | False | Force re-initialization |
| `--auto-index` | - | True | Auto-start indexing after init |
| `--no-auto-index` | - | False | Skip automatic indexing |
| `--mcp` | - | True | Install Claude Code MCP integration |
| `--no-mcp` | - | False | Skip MCP integration |
| `--auto-indexing` | - | True | Set up file change watching |
| `--no-auto-indexing` | - | False | Disable file watching |

---

## 2. Current Default Exclusions

### Directory Patterns (DEFAULT_IGNORE_PATTERNS)

**Location:** `src/mcp_vector_search/config/defaults.py:112-134`

```python
DEFAULT_IGNORE_PATTERNS = [
    ".git",                # Git repository metadata
    ".svn",                # SVN metadata
    ".hg",                 # Mercurial metadata
    "__pycache__",         # Python bytecode cache
    ".pytest_cache",       # Pytest cache
    ".mypy_cache",         # Mypy type checking cache
    ".ruff_cache",         # Ruff linter cache
    "node_modules",        # Node.js dependencies
    ".venv",               # Python virtual environment (common name)
    "venv",                # Python virtual environment (alternate)
    ".env",                # Environment files (security sensitive)
    "build",               # Build artifacts
    "dist",                # Distribution artifacts
    "target",              # Rust/Java build artifacts
    ".idea",               # JetBrains IDE
    ".vscode",             # VS Code settings
    "*.egg-info",          # Python package metadata
    ".DS_Store",           # macOS metadata
    "Thumbs.db",           # Windows metadata
    ".claude-mpm",         # Claude MPM directory
    ".mcp-vector-search",  # Own index directory
]
# Total: 21 patterns (13 directories + 8 file patterns)
```

### File Patterns (DEFAULT_IGNORE_FILES)

**Location:** `src/mcp_vector_search/config/defaults.py:137-168`

```python
DEFAULT_IGNORE_FILES = [
    # Compiled Python
    "*.pyc", "*.pyo", "*.pyd",

    # Native libraries
    "*.so", "*.dll", "*.dylib",

    # Executables
    "*.exe", "*.bin",

    # Object files
    "*.obj", "*.o", "*.a", "*.lib",

    # Java archives
    "*.jar", "*.war", "*.ear",

    # Compressed files
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.xz", "*.7z", "*.rar",

    # Disk images
    "*.iso", "*.dmg", "*.img",

    # Temporary files
    "*.log", "*.tmp", "*.temp", "*.cache", "*.lock",
]
# Total: 26 patterns
```

---

## 3. Supported Languages and Parsers

### Languages with Full Tree-Sitter Parsers

**Location:** `src/mcp_vector_search/parsers/registry.py:33-65`

| Language | Parser Class | Extensions | Tree-Sitter Support |
|----------|-------------|-----------|---------------------|
| Python | `PythonParser` | `.py`, `.pyw` | ✅ Full AST parsing |
| JavaScript | `JavaScriptParser` | `.js`, `.jsx`, `.mjs` | ✅ Full AST parsing |
| TypeScript | `TypeScriptParser` | `.ts`, `.tsx` | ✅ Full AST parsing |
| Dart | `DartParser` | `.dart` | ✅ Full AST parsing |
| PHP | `PHPParser` | `.php` | ✅ Full AST parsing |
| Ruby | `RubyParser` | `.rb` | ✅ Full AST parsing |
| Text | `TextParser` | `.txt`, `.md` | 📝 Paragraph-based |
| HTML | `HTMLParser` | `.html` | 📝 Tag-based |

**Note:** Extensions listed in `DEFAULT_FILE_EXTENSIONS` but without full parsers use `FallbackParser` (line-based chunking with heuristics).

### Fallback Support

**Extensions without dedicated parsers:**
- `.java`, `.cpp`, `.c`, `.h`, `.hpp` (C/C++/Java)
- `.cs` (C#)
- `.go` (Go)
- `.rs` (Rust)
- `.swift` (Swift)
- `.kt` (Kotlin)
- `.scala` (Scala)
- `.sh`, `.bash`, `.zsh` (Shell scripts)
- `.json` (JSON)

These use `FallbackParser` which provides basic line-based chunking without AST analysis.

---

## 4. Gap Analysis: Missing Common Exclusions

### Critical Gaps (High Priority)

**Python Testing/Coverage:**
```
.tox/                  # Tox testing environments (common in CI/CD)
.coverage              # Coverage.py data file
htmlcov/               # Coverage HTML reports
.hypothesis/           # Hypothesis test data
.nox/                  # Nox testing sessions
```

**Python Caching:**
```
.cache/                # Generic cache directory (pytest, etc.)
.eggs/                 # Python egg directory
*.egg                  # Python egg files
```

**JavaScript/Node.js:**
```
.npm/                  # npm cache
.yarn/                 # Yarn cache
.pnp.js                # Yarn Plug'n'Play
.pnp/                  # Yarn PnP directory
coverage/              # JavaScript coverage reports
.nyc_output/           # NYC coverage tool
bower_components/      # Bower dependencies (legacy)
jspm_packages/         # jspm packages (legacy)
```

**Build/Distribution:**
```
*.whl                  # Python wheel files
*.egg                  # Python egg files
wheels/                # Python wheel cache
pip-wheel-metadata/    # pip metadata
```

**IDE/Editor:**
```
.vim/                  # Vim swap files
*.swp                  # Vim swap files
*.swo                  # Vim swap files
.vimrc.local           # Local vim config
*.sublime-*            # Sublime Text files
.project               # Eclipse project
.classpath             # Eclipse classpath
.settings/             # Eclipse settings
```

**OS/System:**
```
desktop.ini            # Windows desktop config
$RECYCLE.BIN/          # Windows recycle bin
.Trash-*               # Linux trash
.AppleDouble/          # macOS resource forks
.LSOverride            # macOS folder settings
```

### Moderate Priority Gaps

**Documentation Builds:**
```
_build/                # Sphinx build output
docs/_build/           # Common docs location
.doctrees/             # Sphinx doctrees
site/                  # MkDocs output
```

**Language-Specific:**
```
# Rust
Cargo.lock             # Should be committed but changes frequently
target/                # Already excluded (Rust build)

# Go
vendor/                # Go vendor directory
go.sum                 # Go checksum file (changes frequently)

# Ruby
.bundle/               # Bundler directory
vendor/bundle/         # Ruby gems

# Java
.gradle/               # Gradle cache
.mvn/                  # Maven wrapper
```

### Low Priority (Edge Cases)

```
.terraform/            # Terraform state
.vagrant/              # Vagrant box
.kitchen/              # Test Kitchen
.serverless/           # Serverless framework
.aws-sam/              # AWS SAM build
```

---

## 5. Recommended Additions to DEFAULT_IGNORE_PATTERNS

### Immediate Additions (High Impact)

```python
# Python testing/coverage
".tox",
".coverage",
"htmlcov",
".hypothesis",
".nox",
".cache",

# JavaScript/Node.js
".npm",
".yarn",
"coverage",
".nyc_output",

# Generic build/cache
"wheels",
"pip-wheel-metadata",

# Documentation
"_build",
"site",

# Common vendor directories
"vendor",
```

### Immediate Additions (File Patterns)

```python
# Python wheels and eggs
"*.whl",
"*.egg",

# Vim swap files
"*.swp",
"*.swo",

# Sublime Text
"*.sublime-workspace",
"*.sublime-project",

# Go checksums
"go.sum",

# Coverage data
".coverage.*",
"coverage.xml",
```

---

## 6. Language Extension Mapping Reference

**Location:** `src/mcp_vector_search/config/defaults.py:42-69`

```python
LANGUAGE_MAPPINGS = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp", ".hpp": "cpp",
    ".c": "c", ".h": "c",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".json": "json",
    ".md": "markdown",
    ".txt": "text",
}
```

---

## 7. Recommendations

### For Users

**1. Custom Extension Configuration:**
```bash
# Python-only project
mcp-vector-search init -e .py,.pyi,.pyx,.txt,.md,.rst

# Full-stack JavaScript/TypeScript
mcp-vector-search init -e .ts,.tsx,.js,.jsx,.mjs,.json,.md

# Minimal configuration (documentation only)
mcp-vector-search init -e .md,.txt,.rst
```

**2. Post-Init Cleanup:**
```bash
# If you accidentally indexed with defaults, clean up and re-index
mcp-vector-search reset index --force
mcp-vector-search init --extensions .py,.js,.ts --force
```

### For Maintainers

**1. Expand DEFAULT_IGNORE_PATTERNS:**
```python
DEFAULT_IGNORE_PATTERNS = [
    # Existing patterns...

    # Add Python testing/coverage
    ".tox", ".coverage", "htmlcov", ".hypothesis", ".nox", ".cache",

    # Add JS/Node.js
    ".npm", ".yarn", "coverage", ".nyc_output",

    # Add generic build/cache
    "wheels", "pip-wheel-metadata",

    # Add documentation builds
    "_build", "site",

    # Add vendor directories
    "vendor",
]
```

**2. Expand DEFAULT_IGNORE_FILES:**
```python
DEFAULT_IGNORE_FILES = [
    # Existing patterns...

    # Add Python wheels/eggs
    "*.whl", "*.egg",

    # Add editor swap files
    "*.swp", "*.swo",

    # Add coverage reports
    ".coverage.*", "coverage.xml",
]
```

**3. Documentation Updates:**
- Document `--extensions` flag more prominently
- Add examples for common project types
- Clarify difference between full parsers and fallback parsing
- Document LANGUAGE_MAPPINGS for custom extension use

---

## 8. Related Files for Reference

| File | Purpose |
|------|---------|
| `src/mcp_vector_search/cli/commands/init.py` | Init command implementation |
| `src/mcp_vector_search/config/defaults.py` | Default configurations |
| `src/mcp_vector_search/parsers/registry.py` | Parser registration |
| `src/mcp_vector_search/parsers/base.py` | Parser base classes |

---

## Appendix A: Full CLI Help Text

```
mcp-vector-search init --help

Usage: mcp-vector-search init [OPTIONS]

  🚀 Complete project setup for semantic code search with MCP integration.

Options:
  -c, --config PATH              Configuration file to use
  -e, --extensions TEXT          Comma-separated list of file extensions
  -m, --embedding-model TEXT     Embedding model to use
  -s, --similarity-threshold FLOAT  Similarity threshold (0.0 to 1.0)
  -f, --force                    Force re-initialization
  --auto-index/--no-auto-index   Auto-start indexing [default: auto-index]
  --mcp/--no-mcp                 Install Claude Code MCP [default: mcp]
  --auto-indexing/--no-auto-indexing  Set up file watching [default: auto-indexing]
  --help                         Show this message and exit
```

---

**End of Analysis**

Generated by Claude Research Agent
Repository: https://github.com/bobmatnyc/mcp-vector-search
Version: 1.1.22

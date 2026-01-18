# Ticket #75: Default to Indexing All Supported Code Extensions

**Investigation Date**: 2026-01-18
**Ticket**: [Issue #75](https://github.com/bobmatnyc/mcp-vector-search/issues/75)
**Objective**: Understand where extensions are defined and how to make "all supported extensions" the default

---

## Executive Summary

To implement "default to all supported extensions", you need to:

1. **Update DEFAULT_FILE_EXTENSIONS** in `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py` (lines 13-39)
2. **Generate extensions dynamically** from parser registry instead of hardcoded list
3. **Update ProjectConfig default** in `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py` (line 17)

**Key Insight**: The parser registry dynamically reports all supported extensions. We can query it to get the complete list rather than maintaining a separate hardcoded default list.

---

## 1. Where Are Default Extensions Defined?

### Primary Location: config/defaults.py

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`
**Lines**: 13-39

```python
# Default file extensions to index (prioritize supported languages)
DEFAULT_FILE_EXTENSIONS = [
    ".py",  # Python (fully supported)
    ".js",  # JavaScript (fully supported)
    ".ts",  # TypeScript (fully supported)
    ".jsx",  # React JSX (fully supported)
    ".tsx",  # React TSX (fully supported)
    ".mjs",  # ES6 modules (fully supported)
    ".java",  # Java (fallback parsing)
    ".cpp",  # C++ (fallback parsing)
    ".c",  # C (fallback parsing)
    ".h",  # C/C++ headers (fallback parsing)
    ".hpp",  # C++ headers (fallback parsing)
    ".cs",  # C# (fallback parsing)
    ".go",  # Go (fallback parsing)
    ".rs",  # Rust (fallback parsing)
    ".php",  # PHP (fallback parsing)
    ".rb",  # Ruby (fallback parsing)
    ".swift",  # Swift (fallback parsing)
    ".kt",  # Kotlin (fallback parsing)
    ".scala",  # Scala (fallback parsing)
    ".sh",  # Shell scripts (fallback parsing)
    ".bash",  # Bash scripts (fallback parsing)
    ".zsh",  # Zsh scripts (fallback parsing)
    ".json",  # JSON configuration files
    ".md",  # Markdown documentation
    ".txt",  # Plain text files
]
```

**Current Count**: 27 extensions

### Secondary Location: config/settings.py

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/settings.py`
**Lines**: 16-19

```python
file_extensions: list[str] = Field(
    default=[".py", ".js", ".ts", ".jsx", ".tsx"],
    description="File extensions to index",
)
```

**Issue**: This Pydantic model has a DIFFERENT default (only 5 extensions) than `DEFAULT_FILE_EXTENSIONS` (27 extensions).

---

## 2. Where Does `init` Command Use Extensions?

### CLI Handler: cli/commands/init.py

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/init.py`

#### Extension Flag Definition (Lines 50-56)

```python
extensions: str | None = typer.Option(
    None,
    "--extensions",
    "-e",
    help="Comma-separated list of file extensions to index (e.g., '.py,.js,.ts,.txt,.md')",
    rich_help_panel="📁 Configuration",
),
```

#### Extension Processing Logic (Lines 157-166)

```python
# Parse file extensions
file_extensions = None
if extensions:
    file_extensions = [ext.strip() for ext in extensions.split(",")]
    # Ensure extensions start with dot
    file_extensions = [
        ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
    ]
else:
    file_extensions = DEFAULT_FILE_EXTENSIONS
```

**Behavior**:
- If user provides `--extensions`, use those
- Otherwise, fall back to `DEFAULT_FILE_EXTENSIONS` from `config/defaults.py`

#### Config Creation (Lines 191-196)

```python
project_manager.initialize(
    file_extensions=file_extensions,
    embedding_model=embedding_model,
    similarity_threshold=similarity_threshold,
    force=force,
)
```

### ProjectManager: core/project.py

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/project.py`

#### initialize() Method (Lines 80-153)

```python
def initialize(
    self,
    file_extensions: list[str] | None = None,
    embedding_model: str = "microsoft/codebert-base",
    similarity_threshold: float = 0.5,
    force: bool = False,
) -> ProjectConfig:
    """Initialize the project for vector search.

    Args:
        file_extensions: File extensions to index
        ...
    """
    # ... initialization logic ...

    # Create configuration
    config = ProjectConfig(
        project_root=self.project_root,
        index_path=index_path,
        file_extensions=file_extensions or DEFAULT_FILE_EXTENSIONS,  # Line 137
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        languages=detected_languages,
    )
```

**Key Line**: Line 137 uses `file_extensions or DEFAULT_FILE_EXTENSIONS`

#### Config Save Logic (Lines 205-214)

```python
# Convert to JSON-serializable format
config_data = config.model_dump()
config_data["project_root"] = str(config.project_root)
config_data["index_path"] = str(config.index_path)

with open(config_path, "w") as f:
    json.dump(config_data, f, indent=2)
```

**Output Location**: `.mcp-vector-search/config.json`

---

## 3. What Extensions Do Parsers Support?

### Parser Registry: parsers/registry.py

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/parsers/registry.py`

#### Parser Registration (Lines 33-66)

```python
def _register_default_parsers(self) -> None:
    """Register default parsers for supported languages."""
    # Register Python parser
    python_parser = PythonParser()
    self.register_parser("python", python_parser)

    # Register JavaScript parser
    javascript_parser = JavaScriptParser()
    self.register_parser("javascript", javascript_parser)

    # Register TypeScript parser
    typescript_parser = TypeScriptParser()
    self.register_parser("typescript", typescript_parser)

    # Register Dart parser
    dart_parser = DartParser()
    self.register_parser("dart", dart_parser)

    # Register PHP parser
    php_parser = PHPParser()
    self.register_parser("php", php_parser)

    # Register Ruby parser
    ruby_parser = RubyParser()
    self.register_parser("ruby", ruby_parser)

    # Register Text parser for .txt files
    text_parser = TextParser()
    self.register_parser("text", text_parser)

    # Register HTML parser for .html files
    html_parser = HTMLParser()
    self.register_parser("html", html_parser)
```

#### Get Supported Extensions (Lines 120-127)

```python
def get_supported_extensions(self) -> list[str]:
    """Get list of supported file extensions.

    Returns:
        List of file extensions
    """
    self._ensure_initialized()
    return list(self._extension_map.keys())
```

**Important**: This method dynamically returns ALL registered extensions from all parsers.

### Individual Parser Extensions

Based on grep results from each parser's `get_supported_extensions()` method:

| Parser | File | Extensions Returned |
|--------|------|---------------------|
| **PythonParser** | `parsers/python.py:782` | `[".py", ".pyw"]` |
| **JavaScriptParser** | `parsers/javascript.py:618` | `[".js", ".jsx", ".mjs"]` |
| **TypeScriptParser** | `parsers/javascript.py:616` | `[".ts", ".tsx"]` |
| **DartParser** | `parsers/dart.py:605` | `[".dart"]` |
| **PHPParser** | `parsers/php.py:694` | `[".php", ".phtml"]` |
| **RubyParser** | `parsers/ruby.py:678` | `[".rb", ".rake", ".gemspec"]` |
| **TextParser** | `parsers/text.py:186` | `[".txt", ".md", ".markdown"]` |
| **HTMLParser** | `parsers/html.py:413` | `[".html", ".htm"]` |
| **FallbackParser** | `parsers/base.py:296` | `["*"]` (special marker) |

**Total Supported Extensions (from parsers)**: 19 extensions

```python
[
    ".py", ".pyw",           # Python (2)
    ".js", ".jsx", ".mjs",   # JavaScript (3)
    ".ts", ".tsx",           # TypeScript (2)
    ".dart",                 # Dart (1)
    ".php", ".phtml",        # PHP (2)
    ".rb", ".rake", ".gemspec",  # Ruby (3)
    ".txt", ".md", ".markdown",  # Text (3)
    ".html", ".htm",         # HTML (2)
    # Plus FallbackParser handles everything else
]
```

### Comparison: Defaults vs. Parser Support

**DEFAULT_FILE_EXTENSIONS (27 extensions)**:
- Includes: `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.go`, `.rs`, `.swift`, `.kt`, `.scala`, `.sh`, `.bash`, `.zsh`, `.json`
- These are listed as "fallback parsing" languages

**Parser Registry (19 extensions)**:
- Only includes extensions with dedicated parsers
- Does NOT include the fallback languages in its list

**FallbackParser Behavior**:
- Returns `["*"]` as special marker meaning "supports all"
- Is used for ANY extension not in the registry
- Handles: `.java`, `.cpp`, `.go`, `.rs`, `.json`, etc. automatically

---

## 4. Where Is Config Written?

### Config File Path

**Function**: `config/defaults.py:209-211`

```python
def get_default_config_path(project_root: Path) -> Path:
    """Get the default configuration file path for a project."""
    return project_root / ".mcp-vector-search" / "config.json"
```

**Location**: `<project_root>/.mcp-vector-search/config.json`

### Config Write Logic

**File**: `core/project.py:205-214`

```python
try:
    # Convert to JSON-serializable format
    config_data = config.model_dump()
    config_data["project_root"] = str(config.project_root)
    config_data["index_path"] = str(config.index_path)

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.debug(f"Saved configuration to {config_path}")
```

**Process**:
1. Convert `ProjectConfig` Pydantic model to dict via `model_dump()`
2. Convert Path objects to strings
3. Write JSON to `.mcp-vector-search/config.json`

### Example config.json

```json
{
  "project_root": "/path/to/project",
  "index_path": "/path/to/project/.mcp-vector-search",
  "file_extensions": [
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".json",
    ".md",
    ".txt"
  ],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "similarity_threshold": 0.5,
  "languages": ["python", "javascript", "typescript"],
  "watch_files": false,
  "cache_embeddings": true,
  "max_cache_size": 1000,
  "auto_reindex_on_upgrade": true,
  "skip_dotfiles": true,
  "respect_gitignore": true
}
```

---

## 5. Changes Needed for "All Extensions by Default"

### Option A: Dynamic Extension List (Recommended)

**Goal**: Query the parser registry for all supported extensions instead of hardcoding.

#### Change 1: Update defaults.py

**File**: `src/mcp_vector_search/config/defaults.py`

**Add new function** (after line 39):

```python
def get_all_supported_extensions() -> list[str]:
    """Get all extensions supported by registered parsers.

    Returns:
        List of all supported file extensions from parser registry
    """
    from ..parsers.registry import get_parser_registry

    registry = get_parser_registry()
    extensions = registry.get_supported_extensions()

    # Filter out the fallback marker "*"
    return [ext for ext in extensions if ext != "*"]
```

**Replace DEFAULT_FILE_EXTENSIONS** (lines 13-39):

```python
# Default file extensions to index - dynamically generated from parser registry
def _get_default_extensions() -> list[str]:
    """Get default extensions to index.

    Returns all parser-supported extensions plus common fallback languages.
    """
    # Get all extensions with dedicated parsers
    from ..parsers.registry import get_parser_registry

    registry = get_parser_registry()
    parser_extensions = [ext for ext in registry.get_supported_extensions() if ext != "*"]

    # Add common fallback extensions (languages without dedicated parsers but useful)
    fallback_extensions = [
        ".java",   # Java
        ".cpp", ".c", ".h", ".hpp",  # C/C++
        ".cs",     # C#
        ".go",     # Go
        ".rs",     # Rust
        ".swift",  # Swift
        ".kt",     # Kotlin
        ".scala",  # Scala
        ".sh", ".bash", ".zsh",  # Shell
        ".json",   # JSON
    ]

    # Combine and deduplicate
    all_extensions = list(set(parser_extensions + fallback_extensions))
    return sorted(all_extensions)

DEFAULT_FILE_EXTENSIONS = _get_default_extensions()
```

**Why this works**:
- Dynamically includes all parser-supported extensions
- Adds useful fallback extensions (handled by FallbackParser)
- Self-updating when new parsers are added
- Maintains backward compatibility

#### Change 2: Update settings.py default

**File**: `src/mcp_vector_search/config/settings.py`
**Lines**: 16-19

**Before**:
```python
file_extensions: list[str] = Field(
    default=[".py", ".js", ".ts", ".jsx", ".tsx"],
    description="File extensions to index",
)
```

**After**:
```python
file_extensions: list[str] = Field(
    default_factory=lambda: DEFAULT_FILE_EXTENSIONS.copy(),
    description="File extensions to index (defaults to all supported extensions)",
)
```

**Import required**:
```python
from ..config.defaults import DEFAULT_FILE_EXTENSIONS
```

**Why this works**:
- Uses `default_factory` to call function at runtime
- Ensures Pydantic model matches `defaults.py`
- No hardcoded duplication

---

### Option B: Static Extension List (Simpler but requires maintenance)

If you prefer to avoid dynamic imports in defaults.py:

#### Update DEFAULT_FILE_EXTENSIONS manually

**File**: `src/mcp_vector_search/config/defaults.py`
**Lines**: 13-39

**Replace with complete list**:

```python
# Default file extensions to index (all supported languages)
DEFAULT_FILE_EXTENSIONS = [
    # Python (parser: PythonParser)
    ".py", ".pyw",

    # JavaScript (parser: JavaScriptParser)
    ".js", ".jsx", ".mjs",

    # TypeScript (parser: TypeScriptParser)
    ".ts", ".tsx",

    # Dart/Flutter (parser: DartParser)
    ".dart",

    # PHP (parser: PHPParser)
    ".php", ".phtml",

    # Ruby (parser: RubyParser)
    ".rb", ".rake", ".gemspec",

    # Text/Markdown (parser: TextParser)
    ".txt", ".md", ".markdown",

    # HTML (parser: HTMLParser)
    ".html", ".htm",

    # Fallback languages (parser: FallbackParser)
    ".java",   # Java
    ".cpp", ".c", ".h", ".hpp",  # C/C++
    ".cs",     # C#
    ".go",     # Go
    ".rs",     # Rust
    ".swift",  # Swift
    ".kt",     # Kotlin
    ".scala",  # Scala
    ".sh", ".bash", ".zsh",  # Shell scripts
    ".json",   # JSON
]
```

**Total**: 33 extensions (19 with parsers + 14 fallback)

**Downside**: Requires manual updates when new parsers are added.

---

## 6. Testing the Changes

### Test 1: Verify Extension Count

```python
from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS
from mcp_vector_search.parsers.registry import get_parser_registry

print(f"DEFAULT_FILE_EXTENSIONS count: {len(DEFAULT_FILE_EXTENSIONS)}")
print(f"Parser registry extensions: {get_parser_registry().get_supported_extensions()}")

# Should include all parser extensions plus fallback extensions
assert ".dart" in DEFAULT_FILE_EXTENSIONS  # From DartParser
assert ".gemspec" in DEFAULT_FILE_EXTENSIONS  # From RubyParser
assert ".java" in DEFAULT_FILE_EXTENSIONS  # Fallback
assert ".go" in DEFAULT_FILE_EXTENSIONS  # Fallback
```

### Test 2: Init Command Uses All Extensions

```bash
# Initialize project without --extensions flag
cd /tmp/test-project
mcp-vector-search init --no-auto-index --no-mcp

# Check config.json
cat .mcp-vector-search/config.json | jq '.file_extensions | length'
# Should be 30+ extensions (not just 5)

cat .mcp-vector-search/config.json | jq '.file_extensions' | grep -c "dart"
# Should return 1 (dart is included)
```

### Test 3: Custom Extensions Still Work

```bash
# User can still override with --extensions
mcp-vector-search init --extensions .py,.js,.ts --force

# Check config.json
cat .mcp-vector-search/config.json | jq '.file_extensions'
# Should only show [".py", ".js", ".ts"]
```

---

## 7. Summary of Changes

### Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `config/defaults.py` | Add `get_all_supported_extensions()` or update `DEFAULT_FILE_EXTENSIONS` | 13-39 (replace) + new function |
| `config/settings.py` | Change `file_extensions` default to use `DEFAULT_FILE_EXTENSIONS` | 16-19 |

### Behavior Change

**Before**:
- `mcp-vector-search init` (no flags) → indexes 27 extensions
- ProjectConfig Pydantic default → 5 extensions (INCONSISTENT!)

**After (Option A - Dynamic)**:
- `mcp-vector-search init` (no flags) → indexes 30+ extensions (all parsers + fallback)
- ProjectConfig Pydantic default → same 30+ extensions (CONSISTENT!)
- Auto-updates when new parsers added

**After (Option B - Static)**:
- `mcp-vector-search init` (no flags) → indexes 33 extensions (hardcoded list)
- ProjectConfig Pydantic default → same 33 extensions (CONSISTENT!)
- Requires manual updates for new parsers

### User Impact

**Positive**:
- Users get full language support by default without needing to know all extensions
- Consistent behavior across CLI and config
- No surprises from missing languages

**Potential Concerns**:
- Larger initial index (more files indexed)
- Could index unwanted file types (e.g., `.json`, `.txt`)

**Mitigation**:
- Users can still use `--extensions` to restrict
- `.gitignore` and `skip_dotfiles` settings filter most noise
- Performance impact minimal (FallbackParser is lightweight)

---

## 8. Recommendations

**Recommendation**: Use **Option A (Dynamic Extension List)** because:

1. **Self-maintaining**: Automatically includes new parsers when added
2. **Single source of truth**: Parser registry controls what's supported
3. **Extensible**: Easy to add new languages by registering parsers
4. **Consistent**: No risk of hardcoded list getting out of sync

**Implementation Priority**:

1. Update `config/defaults.py` with dynamic extension function
2. Update `config/settings.py` to use `default_factory`
3. Add unit tests to verify extension count matches registry
4. Update documentation to mention "all supported languages" instead of listing specific extensions

**Migration Concern**: Existing projects with `config.json` containing only 5 extensions will NOT be affected (config is saved, not regenerated). New projects will get the full list by default.

---

## Appendix: Complete Extension Reference

### Extensions with Dedicated Parsers (19)

```python
PARSER_SUPPORTED = [
    ".py", ".pyw",                    # Python
    ".js", ".jsx", ".mjs",            # JavaScript
    ".ts", ".tsx",                    # TypeScript
    ".dart",                          # Dart/Flutter
    ".php", ".phtml",                 # PHP
    ".rb", ".rake", ".gemspec",       # Ruby
    ".txt", ".md", ".markdown",       # Text/Markdown
    ".html", ".htm",                  # HTML
]
```

### Extensions Using FallbackParser (14+)

```python
FALLBACK_SUPPORTED = [
    ".java",                          # Java
    ".cpp", ".c", ".h", ".hpp",       # C/C++
    ".cs",                            # C#
    ".go",                            # Go
    ".rs",                            # Rust
    ".swift",                         # Swift
    ".kt",                            # Kotlin
    ".scala",                         # Scala
    ".sh", ".bash", ".zsh",           # Shell
    ".json",                          # JSON
    # Plus any other extension (FallbackParser accepts all)
]
```

### Current DEFAULT_FILE_EXTENSIONS (27)

```python
DEFAULT_FILE_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",  # Fully supported (parsers)
    ".java", ".cpp", ".c", ".h", ".hpp", ".cs",   # Fallback parsing
    ".go", ".rs", ".php", ".rb", ".swift", ".kt", # Fallback parsing
    ".scala", ".sh", ".bash", ".zsh",             # Fallback parsing
    ".json", ".md", ".txt"                        # Text/config
]
```

**Missing from Defaults**:
- `.pyw` (Python)
- `.dart` (Dart/Flutter)
- `.phtml` (PHP)
- `.rake`, `.gemspec` (Ruby)
- `.markdown` (Text)
- `.html`, `.htm` (HTML)

**Total Missing**: 8 extensions that have dedicated parsers!

---

## Conclusion

To implement ticket #75 ("Default to indexing all supported code extensions"):

1. **Modify** `config/defaults.py` to dynamically query parser registry
2. **Update** `config/settings.py` to use consistent defaults
3. **Result**: `mcp-vector-search init` will index all 30+ supported extensions by default

This ensures users get full language support out-of-the-box while maintaining the ability to customize via `--extensions` flag.

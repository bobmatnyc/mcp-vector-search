"""Default configurations for MCP Vector Search."""

from pathlib import Path

# Dotfiles that should NEVER be skipped (CI/CD configurations)
ALLOWED_DOTFILES = {
    ".github",  # GitHub workflows/actions
    ".gitlab-ci",  # GitLab CI
    ".circleci",  # CircleCI config
}

# Default file extensions to index - ALL supported code files
# This comprehensive list includes Tree-Sitter supported languages and fallback parsing
# Users can filter to specific extensions using the --extensions CLI flag
DEFAULT_FILE_EXTENSIONS = [
    # Python - Tree-Sitter fully supported
    ".py",
    ".pyw",
    ".pyi",
    # JavaScript - Tree-Sitter fully supported
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    # TypeScript - Tree-Sitter fully supported
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    # Web - Tree-Sitter supported
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    # Data/Config formats
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    # Documentation
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    # Shell scripts - Tree-Sitter supported
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    # JVM languages - fallback parsing
    ".java",
    ".kt",
    ".scala",
    ".groovy",
    # C/C++ - Tree-Sitter supported
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    # C# - fallback parsing
    ".cs",
    # Go - Tree-Sitter supported
    ".go",
    # Rust - Tree-Sitter supported
    ".rs",
    # Ruby - Tree-Sitter supported
    ".rb",
    ".rake",
    ".gemspec",
    # PHP - Tree-Sitter supported
    ".php",
    ".phtml",
    # Swift - fallback parsing
    ".swift",
    # Dart - fallback parsing
    ".dart",
    # R - fallback parsing
    ".r",
    ".R",
    # SQL - fallback parsing
    ".sql",
    # Lua - fallback parsing
    ".lua",
    # Perl - fallback parsing
    ".pl",
    ".pm",
    # Elixir - fallback parsing
    ".ex",
    ".exs",
    # Clojure - fallback parsing
    ".clj",
    ".cljs",
    ".cljc",
    # Haskell - fallback parsing
    ".hs",
    # OCaml - fallback parsing
    ".ml",
    ".mli",
    # Vim - fallback parsing
    ".vim",
    # Emacs Lisp - fallback parsing
    ".el",
]

# Language mappings for parsers
LANGUAGE_MAPPINGS: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # TypeScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # Web
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    # Data/Config
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    # Documentation
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    # Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    # JVM languages
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    # C/C++
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".hxx": "cpp",
    # C#
    ".cs": "c_sharp",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Ruby
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    # PHP
    ".php": "php",
    ".phtml": "php",
    # Swift
    ".swift": "swift",
    # Dart
    ".dart": "dart",
    # R
    ".r": "r",
    ".R": "r",
    # SQL
    ".sql": "sql",
    # Lua
    ".lua": "lua",
    # Perl
    ".pl": "perl",
    ".pm": "perl",
    # Elixir
    ".ex": "elixir",
    ".exs": "elixir",
    # Clojure
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    # Haskell
    ".hs": "haskell",
    # OCaml
    ".ml": "ocaml",
    ".mli": "ocaml",
    # Vim
    ".vim": "vim",
    # Emacs Lisp
    ".el": "elisp",
}

# Default embedding models by use case
DEFAULT_EMBEDDING_MODELS = {
    "code": "sentence-transformers/all-MiniLM-L6-v2",  # Changed from microsoft/codebert-base which doesn't exist
    "multilingual": "sentence-transformers/all-MiniLM-L6-v2",
    "fast": "sentence-transformers/all-MiniLM-L12-v2",
    "precise": "sentence-transformers/all-mpnet-base-v2",  # Changed from microsoft/unixcoder-base
}

# Default similarity thresholds by language
DEFAULT_SIMILARITY_THRESHOLDS = {
    "python": 0.3,
    "javascript": 0.3,
    "typescript": 0.3,
    "java": 0.3,
    "cpp": 0.3,
    "c": 0.3,
    "go": 0.3,
    "rust": 0.3,
    "json": 0.4,  # JSON files may have more structural similarity
    "markdown": 0.3,  # Markdown documentation
    "text": 0.3,  # Plain text files
    "default": 0.3,
}

# Default chunk sizes by language (in tokens)
DEFAULT_CHUNK_SIZES = {
    "python": 512,
    "javascript": 384,
    "typescript": 384,
    "java": 512,
    "cpp": 384,
    "c": 384,
    "go": 512,
    "rust": 512,
    "json": 256,  # JSON files are often smaller and more structured
    "markdown": 512,  # Markdown documentation can be chunked normally
    "text": 384,  # Plain text files with paragraph-based chunking
    "default": 512,
}

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


def get_default_config_path(project_root: Path) -> Path:
    """Get the default configuration file path for a project."""
    return project_root / ".mcp-vector-search" / "config.json"


def get_default_index_path(project_root: Path) -> Path:
    """Get the default index directory path for a project."""
    return project_root / ".mcp-vector-search"


def get_default_cache_path(project_root: Path) -> Path:
    """Get the default cache directory path for a project."""
    return project_root / ".mcp-vector-search" / "cache"


def get_language_from_extension(extension: str) -> str:
    """Get the language name from file extension."""
    return LANGUAGE_MAPPINGS.get(extension.lower(), "text")


def get_similarity_threshold(language: str) -> float:
    """Get the default similarity threshold for a language."""
    return DEFAULT_SIMILARITY_THRESHOLDS.get(
        language.lower(), DEFAULT_SIMILARITY_THRESHOLDS["default"]
    )


def get_chunk_size(language: str) -> int:
    """Get the default chunk size for a language."""
    return DEFAULT_CHUNK_SIZES.get(language.lower(), DEFAULT_CHUNK_SIZES["default"])

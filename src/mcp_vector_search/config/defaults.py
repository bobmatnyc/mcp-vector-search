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
# MiniLM-L6-v2 is the default: fast, reliable, good enough for code search
# CodeXEmbed integration pending (see issue #81 for status)
DEFAULT_EMBEDDING_MODELS = {
    # MiniLM is the default - fast, reliable, good enough for code search
    # CodeXEmbed integration pending (see issue #81)
    "code": "sentence-transformers/all-MiniLM-L6-v2",  # Default: fast and reliable
    "multilingual": "sentence-transformers/all-MiniLM-L6-v2",  # Default for now
    "fast": "sentence-transformers/all-MiniLM-L6-v2",  # Fastest option
    "precise": "sentence-transformers/all-mpnet-base-v2",  # Higher quality general model
    "legacy": "sentence-transformers/all-MiniLM-L6-v2",  # Backward compatibility (384 dims)
}

# Model specifications for dimension auto-detection and validation
MODEL_SPECIFICATIONS = {
    # CodeXEmbed models (code-specific, state-of-the-art)
    "Salesforce/SFR-Embedding-Code-400M_R": {
        "dimensions": 1024,  # Actual output dimensions from HuggingFace model
        "context_length": 2048,
        "type": "code",
        "description": "CodeXEmbed-400M: State-of-the-art code embeddings (12 languages)",
    },
    "Salesforce/SFR-Embedding-Code-2B_R": {
        "dimensions": 1024,  # Actual output dimensions from HuggingFace model
        "context_length": 2048,
        "type": "code",
        "description": "CodeXEmbed-2B: Highest quality code embeddings (large model)",
    },
    # Microsoft code models (work without trust_remote_code)
    "microsoft/graphcodebert-base": {
        "dimensions": 768,
        "context_length": 512,
        "type": "code",
        "description": "GraphCodeBERT: Code embeddings with data flow understanding (6 languages)",
    },
    "microsoft/codebert-base": {
        "dimensions": 768,
        "context_length": 512,
        "type": "code",
        "description": "CodeBERT: Bimodal code/text embeddings (6 languages)",
    },
    # Legacy sentence-transformers models
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "context_length": 256,
        "type": "general",
        "description": "Legacy: Fast general-purpose embeddings (not code-optimized)",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dimensions": 768,
        "context_length": 512,
        "type": "general",
        "description": "General-purpose embeddings with higher quality",
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "dimensions": 384,
        "context_length": 256,
        "type": "general",
        "description": "Balanced speed and quality for general text",
    },
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
    # Exclude all hidden/dot directories by default
    # This wildcard pattern catches ALL directories starting with a dot
    # Users can override this in their project config if needed
    # Note: ALLOWED_DOTFILES (like .github) are whitelisted separately
    ".*",
    # Specific patterns below are kept for documentation and fallback
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


def get_model_dimensions(model_name: str) -> int:
    """Get embedding dimensions for a model.

    Args:
        model_name: Model identifier (e.g., "Salesforce/SFR-Embedding-Code-400M_R")

    Returns:
        Number of embedding dimensions (768 for CodeXEmbed, 384 for legacy)

    Raises:
        ValueError: If model is unknown and dimensions cannot be inferred
    """
    if model_name in MODEL_SPECIFICATIONS:
        return MODEL_SPECIFICATIONS[model_name]["dimensions"]

    # Fallback: Try to infer from model name patterns
    if "MiniLM" in model_name and "L6" in model_name:
        return 384  # all-MiniLM-L6-v2 pattern
    elif "mpnet" in model_name:
        return 768  # all-mpnet-base-v2 pattern
    elif "SFR-Embedding-Code" in model_name or "CodeXEmbed" in model_name:
        return 768  # CodeXEmbed models

    # Unknown model - raise error to force explicit configuration
    raise ValueError(
        f"Unknown embedding model: {model_name}. "
        f"Please add model specifications to MODEL_SPECIFICATIONS in defaults.py"
    )


def get_model_context_length(model_name: str) -> int:
    """Get maximum context length for a model.

    Args:
        model_name: Model identifier

    Returns:
        Maximum number of tokens (2048 for CodeXEmbed, 256 for legacy)
    """
    if model_name in MODEL_SPECIFICATIONS:
        return MODEL_SPECIFICATIONS[model_name]["context_length"]

    # Fallback: Conservative default
    return 512


def is_code_specific_model(model_name: str) -> bool:
    """Check if model is optimized for code understanding.

    Args:
        model_name: Model identifier

    Returns:
        True if model is code-specific, False otherwise
    """
    if model_name in MODEL_SPECIFICATIONS:
        return MODEL_SPECIFICATIONS[model_name]["type"] == "code"

    # Pattern matching for known code models
    code_model_patterns = [
        "SFR-Embedding-Code",
        "CodeXEmbed",
        "CodeT5",
        "codebert",
        "unixcoder",
    ]
    return any(pattern in model_name for pattern in code_model_patterns)

"""Default configurations for MCP Vector Search."""

from pathlib import Path

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

# Language mappings for parsers
LANGUAGE_MAPPINGS: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".json": "json",
    ".md": "markdown",
    ".txt": "text",
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
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "build",
    "dist",
    "target",
    ".idea",
    ".vscode",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
]

# File patterns to ignore
DEFAULT_IGNORE_FILES = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.bin",
    "*.obj",
    "*.o",
    "*.a",
    "*.lib",
    "*.jar",
    "*.war",
    "*.ear",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.7z",
    "*.rar",
    "*.iso",
    "*.dmg",
    "*.img",
    "*.log",
    "*.tmp",
    "*.temp",
    "*.cache",
    "*.lock",
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

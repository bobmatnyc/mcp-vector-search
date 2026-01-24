"""Pydantic configuration schemas for MCP Vector Search."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS


class ProjectConfig(BaseSettings):
    """Type-safe project configuration with validation."""

    project_root: Path = Field(..., description="Project root directory")
    index_path: Path = Field(
        default=".mcp-vector-search", description="Index storage path"
    )
    file_extensions: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FILE_EXTENSIONS),
        description="File extensions to index",
    )
    embedding_model: str = Field(
        default_factory=lambda: DEFAULT_EMBEDDING_MODELS["code"],
        description="Embedding model name (default: CodeXEmbed-400M for code understanding)",
    )
    similarity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Similarity threshold"
    )
    max_chunk_size: int = Field(
        default=512, gt=0, description="Maximum chunk size in tokens"
    )
    languages: list[str] = Field(
        default=[], description="Detected programming languages"
    )
    watch_files: bool = Field(
        default=False, description="Enable file watching for incremental updates"
    )
    cache_embeddings: bool = Field(default=True, description="Enable embedding caching")
    max_cache_size: int = Field(
        default=1000, gt=0, description="Maximum number of cached embeddings"
    )
    auto_reindex_on_upgrade: bool = Field(
        default=True,
        description="Automatically reindex when tool version is upgraded (minor/major versions)",
    )
    skip_dotfiles: bool = Field(
        default=True,
        description="Skip files and directories starting with '.' (except whitelisted ones)",
    )
    respect_gitignore: bool = Field(
        default=True,
        description="Respect .gitignore patterns when indexing files",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key for chat command (optional, can also use env var)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for chat command (optional, can also use env var)",
    )
    preferred_llm_provider: str | None = Field(
        default=None,
        description="Preferred LLM provider: 'openai' or 'openrouter' (auto-detect if not set)",
    )

    @field_validator("project_root", "index_path", mode="before")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are absolute and normalized."""
        if isinstance(v, str):
            v = Path(v)
        return v.resolve() if isinstance(v, Path) else v

    @field_validator("file_extensions", mode="before")
    @classmethod
    def validate_extensions(cls, v: list[str]) -> list[str]:
        """Ensure extensions start with dot."""
        if isinstance(v, list):
            return [ext if ext.startswith(".") else f".{ext}" for ext in v]
        return v

    model_config = {
        "env_prefix": "MCP_VECTOR_SEARCH_",
        "case_sensitive": False,
    }


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    persist_directory: Path | None = Field(
        default=None, description="ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="code_search", description="ChromaDB collection name"
    )
    batch_size: int = Field(
        default=128,
        gt=0,
        description="Batch size for embedding operations (optimized for modern hardware)",
    )
    enable_telemetry: bool = Field(
        default=False, description="Enable ChromaDB telemetry"
    )

    @field_validator("persist_directory", mode="before")
    @classmethod
    def validate_persist_directory(cls, v: Path | None) -> Path | None:
        """Ensure persist directory is absolute if provided."""
        if v and isinstance(v, str):
            v = Path(v)
        return v.resolve() if isinstance(v, Path) else None

    model_config = {
        "env_prefix": "MCP_VECTOR_SEARCH_DB_",
        "case_sensitive": False,
    }


class SearchConfig(BaseSettings):
    """Search configuration settings."""

    default_limit: int = Field(
        default=10, gt=0, description="Default number of search results"
    )
    max_limit: int = Field(
        default=100, gt=0, description="Maximum number of search results"
    )
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    context_lines: int = Field(
        default=3, ge=0, description="Number of context lines to include"
    )

    @field_validator("max_limit", mode="after")
    @classmethod
    def validate_max_limit(cls, v: int, info) -> int:
        """Ensure max_limit is greater than default_limit."""
        if info.data and "default_limit" in info.data:
            default_limit = info.data["default_limit"]
            if v < default_limit:
                raise ValueError(
                    "max_limit must be greater than or equal to default_limit"
                )
        return v

    model_config = {
        "env_prefix": "MCP_VECTOR_SEARCH_SEARCH_",
        "case_sensitive": False,
    }

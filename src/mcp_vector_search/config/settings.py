"""Pydantic configuration schemas for MCP Vector Search."""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class ProjectConfig(BaseSettings):
    """Type-safe project configuration with validation."""

    project_root: Path = Field(..., description="Project root directory")
    index_path: Path = Field(
        default=".mcp-vector-search", description="Index storage path"
    )
    file_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".jsx", ".tsx"],
        description="File extensions to index",
    )
    embedding_model: str = Field(
        default="microsoft/codebert-base", description="Embedding model name"
    )
    similarity_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Similarity threshold"
    )
    max_chunk_size: int = Field(
        default=512, gt=0, description="Maximum chunk size in tokens"
    )
    languages: List[str] = Field(default=[], description="Detected programming languages")
    watch_files: bool = Field(
        default=False, description="Enable file watching for incremental updates"
    )
    cache_embeddings: bool = Field(
        default=True, description="Enable embedding caching"
    )
    max_cache_size: int = Field(
        default=1000, gt=0, description="Maximum number of cached embeddings"
    )

    @validator("project_root", "index_path")
    def validate_paths(cls, v: Path) -> Path:
        """Ensure paths are absolute and normalized."""
        return v.resolve()

    @validator("file_extensions")
    def validate_extensions(cls, v: List[str]) -> List[str]:
        """Ensure extensions start with dot."""
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]

    class Config:
        env_prefix = "MCP_VECTOR_SEARCH_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    persist_directory: Optional[Path] = Field(
        default=None, description="ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="code_search", description="ChromaDB collection name"
    )
    batch_size: int = Field(
        default=32, gt=0, description="Batch size for embedding operations"
    )
    enable_telemetry: bool = Field(
        default=False, description="Enable ChromaDB telemetry"
    )

    @validator("persist_directory")
    def validate_persist_directory(cls, v: Optional[Path]) -> Optional[Path]:
        """Ensure persist directory is absolute if provided."""
        return v.resolve() if v else None

    class Config:
        env_prefix = "MCP_VECTOR_SEARCH_DB_"
        case_sensitive = False


class SearchConfig(BaseSettings):
    """Search configuration settings."""

    default_limit: int = Field(
        default=10, gt=0, description="Default number of search results"
    )
    max_limit: int = Field(
        default=100, gt=0, description="Maximum number of search results"
    )
    enable_reranking: bool = Field(
        default=True, description="Enable result reranking"
    )
    context_lines: int = Field(
        default=3, ge=0, description="Number of context lines to include"
    )

    @validator("max_limit")
    def validate_max_limit(cls, v: int, values: dict) -> int:
        """Ensure max_limit is greater than default_limit."""
        default_limit = values.get("default_limit", 10)
        if v < default_limit:
            raise ValueError("max_limit must be greater than or equal to default_limit")
        return v

    class Config:
        env_prefix = "MCP_VECTOR_SEARCH_SEARCH_"
        case_sensitive = False

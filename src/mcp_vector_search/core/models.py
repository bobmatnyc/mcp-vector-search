"""Data models for MCP Vector Search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str
    chunk_type: str = "code"  # code, function, class, comment, docstring
    function_name: str | None = None
    class_name: str | None = None
    docstring: str | None = None
    imports: list[str] = None
    complexity_score: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.imports is None:
            self.imports = []

    @property
    def id(self) -> str:
        """Generate unique ID for this chunk."""
        return f"{self.file_path}:{self.start_line}:{self.end_line}"

    @property
    def line_count(self) -> int:
        """Get the number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "docstring": self.docstring,
            "imports": self.imports,
            "complexity_score": self.complexity_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            file_path=Path(data["file_path"]),
            start_line=data["start_line"],
            end_line=data["end_line"],
            language=data["language"],
            chunk_type=data.get("chunk_type", "code"),
            function_name=data.get("function_name"),
            class_name=data.get("class_name"),
            docstring=data.get("docstring"),
            imports=data.get("imports", []),
            complexity_score=data.get("complexity_score", 0.0),
        )


class SearchResult(BaseModel):
    """Represents a search result with metadata."""

    content: str = Field(..., description="The matched code content")
    file_path: Path = Field(..., description="Path to the source file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    language: str = Field(..., description="Programming language")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    rank: int = Field(..., description="Result rank in search results")
    chunk_type: str = Field(default="code", description="Type of code chunk")
    function_name: str | None = Field(
        default=None, description="Function name if applicable"
    )
    class_name: str | None = Field(default=None, description="Class name if applicable")
    context_before: list[str] = Field(default=[], description="Lines before the match")
    context_after: list[str] = Field(default=[], description="Lines after the match")
    highlights: list[str] = Field(default=[], description="Highlighted terms")

    class Config:
        arbitrary_types_allowed = True

    @property
    def line_count(self) -> int:
        """Get the number of lines in this result."""
        return self.end_line - self.start_line + 1

    @property
    def location(self) -> str:
        """Get a human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "chunk_type": self.chunk_type,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "highlights": self.highlights,
            "location": self.location,
            "line_count": self.line_count,
        }


class IndexStats(BaseModel):
    """Statistics about the search index."""

    total_files: int = Field(..., description="Total number of indexed files")
    total_chunks: int = Field(..., description="Total number of code chunks")
    languages: dict[str, int] = Field(..., description="Language distribution")
    file_types: dict[str, int] = Field(..., description="File type distribution")
    index_size_mb: float = Field(..., description="Index size in megabytes")
    last_updated: str = Field(..., description="Last update timestamp")
    embedding_model: str = Field(..., description="Embedding model used")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "languages": self.languages,
            "file_types": self.file_types,
            "index_size_mb": self.index_size_mb,
            "last_updated": self.last_updated,
            "embedding_model": self.embedding_model,
        }


class ProjectInfo(BaseModel):
    """Information about a project."""

    name: str = Field(..., description="Project name")
    root_path: Path = Field(..., description="Project root directory")
    config_path: Path = Field(..., description="Configuration file path")
    index_path: Path = Field(..., description="Index directory path")
    is_initialized: bool = Field(..., description="Whether project is initialized")
    languages: list[str] = Field(default=[], description="Detected languages")
    file_count: int = Field(default=0, description="Number of indexable files")

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "root_path": str(self.root_path),
            "config_path": str(self.config_path),
            "index_path": str(self.index_path),
            "is_initialized": self.is_initialized,
            "languages": self.languages,
            "file_count": self.file_count,
        }

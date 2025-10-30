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

    # Enhancement 1: Complexity scoring
    complexity_score: float = 0.0

    # Enhancement 3: Hierarchical relationships
    chunk_id: str | None = None
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = None
    chunk_depth: int = 0

    # Enhancement 4: Enhanced metadata
    decorators: list[str] = None
    parameters: list[dict] = None
    return_type: str | None = None
    type_annotations: dict[str, str] = None

    # Enhancement 5: Monorepo support
    subproject_name: str | None = None  # "ewtn-plus-foundation"
    subproject_path: str | None = None  # Relative path from root

    def __post_init__(self) -> None:
        """Initialize default values and generate chunk ID."""
        if self.imports is None:
            self.imports = []
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []
        if self.decorators is None:
            self.decorators = []
        if self.parameters is None:
            self.parameters = []
        if self.type_annotations is None:
            self.type_annotations = {}

        # Generate chunk ID if not provided
        if self.chunk_id is None:
            import hashlib

            # Include name and first 50 chars of content for uniqueness
            # This ensures deterministic IDs while handling same-location chunks
            name = self.function_name or self.class_name or ""
            content_hash = hashlib.sha256(self.content[:100].encode()).hexdigest()[:8]
            id_string = f"{self.file_path}:{self.chunk_type}:{name}:{self.start_line}:{self.end_line}:{content_hash}"
            self.chunk_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]

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
            "chunk_id": self.chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "chunk_depth": self.chunk_depth,
            "decorators": self.decorators,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "type_annotations": self.type_annotations,
            "subproject_name": self.subproject_name,
            "subproject_path": self.subproject_path,
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
            chunk_id=data.get("chunk_id"),
            parent_chunk_id=data.get("parent_chunk_id"),
            child_chunk_ids=data.get("child_chunk_ids", []),
            chunk_depth=data.get("chunk_depth", 0),
            decorators=data.get("decorators", []),
            parameters=data.get("parameters", []),
            return_type=data.get("return_type"),
            type_annotations=data.get("type_annotations", {}),
            subproject_name=data.get("subproject_name"),
            subproject_path=data.get("subproject_path"),
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


@dataclass
class Directory:
    """Represents a directory in the project structure."""

    path: Path  # Relative path from project root
    name: str  # Directory name
    parent_path: Path | None = None  # Parent directory path (None for root)
    file_count: int = 0  # Number of files directly in this directory
    subdirectory_count: int = 0  # Number of subdirectories
    total_chunks: int = 0  # Total code chunks in this directory (recursive)
    languages: dict[str, int] = None  # Language distribution in this directory
    depth: int = 0  # Depth from project root (0 = root)
    is_package: bool = False  # True if contains __init__.py or package.json
    last_modified: float | None = (
        None  # Most recent file modification time (unix timestamp)
    )

    def __post_init__(self) -> None:
        """Initialize default values and generate directory ID."""
        if self.languages is None:
            self.languages = {}

    @property
    def id(self) -> str:
        """Generate unique ID for this directory."""
        import hashlib

        return hashlib.sha256(str(self.path).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "path": str(self.path),
            "name": self.name,
            "parent_path": str(self.parent_path) if self.parent_path else None,
            "file_count": self.file_count,
            "subdirectory_count": self.subdirectory_count,
            "total_chunks": self.total_chunks,
            "languages": self.languages,
            "depth": self.depth,
            "is_package": self.is_package,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Directory":
        """Create from dictionary."""
        return cls(
            path=Path(data["path"]),
            name=data["name"],
            parent_path=Path(data["parent_path"]) if data.get("parent_path") else None,
            file_count=data.get("file_count", 0),
            subdirectory_count=data.get("subdirectory_count", 0),
            total_chunks=data.get("total_chunks", 0),
            languages=data.get("languages", {}),
            depth=data.get("depth", 0),
            is_package=data.get("is_package", False),
            last_modified=data.get("last_modified"),
        )


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

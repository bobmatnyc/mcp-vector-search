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
    file_missing: bool = Field(
        default=False, description="True if file no longer exists (stale index)"
    )

    # Quality metrics (from structural analysis)
    cognitive_complexity: int | None = Field(
        default=None, description="Cognitive complexity score"
    )
    cyclomatic_complexity: int | None = Field(
        default=None, description="Cyclomatic complexity score"
    )
    max_nesting_depth: int | None = Field(
        default=None, description="Maximum nesting depth"
    )
    parameter_count: int | None = Field(
        default=None, description="Number of function parameters"
    )
    lines_of_code: int | None = Field(
        default=None, description="Lines of code in chunk"
    )
    complexity_grade: str | None = Field(
        default=None, description="Complexity grade (A-F)"
    )
    code_smells: list[str] = Field(default=[], description="Detected code smells")
    smell_count: int | None = Field(
        default=None, description="Number of code smells detected"
    )
    quality_score: int | None = Field(
        default=None, description="Overall quality score (0-100)"
    )

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

    def calculate_quality_score(self) -> int:
        """Calculate quality score based on complexity grade and code smells.

        Formula:
        - Base: complexity_grade (A=100, B=80, C=60, D=40, F=20)
        - Penalty: -10 per code smell
        - Bonus: +20 if no smells (already factored into base if no smells)

        Returns:
            Quality score (0-100), or None if no quality metrics available
        """
        # If no quality metrics, return None (will be stored in quality_score field)
        if self.complexity_grade is None:
            return None

        # Map complexity grade to base score
        grade_scores = {
            "A": 100,
            "B": 80,
            "C": 60,
            "D": 40,
            "F": 20,
        }

        base_score = grade_scores.get(self.complexity_grade, 0)

        # Apply smell penalty
        smell_count = self.smell_count or 0
        penalty = smell_count * 10

        # Calculate final score (with bonus for no smells already in base)
        # Bonus: +20 if no smells (effectively makes A without smells = 100+20 = 120, capped at 100)
        bonus = 20 if smell_count == 0 else 0
        quality_score = base_score - penalty + bonus

        # Clamp to 0-100 range
        return max(0, min(100, quality_score))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
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

        # Add quality metrics if available
        if self.cognitive_complexity is not None:
            result["cognitive_complexity"] = self.cognitive_complexity
        if self.cyclomatic_complexity is not None:
            result["cyclomatic_complexity"] = self.cyclomatic_complexity
        if self.max_nesting_depth is not None:
            result["max_nesting_depth"] = self.max_nesting_depth
        if self.parameter_count is not None:
            result["parameter_count"] = self.parameter_count
        if self.lines_of_code is not None:
            result["lines_of_code"] = self.lines_of_code
        if self.complexity_grade is not None:
            result["complexity_grade"] = self.complexity_grade
        if self.code_smells:
            result["code_smells"] = self.code_smells
        if self.smell_count is not None:
            result["smell_count"] = self.smell_count
        if self.quality_score is not None:
            result["quality_score"] = self.quality_score

        return result


class IndexStats(BaseModel):
    """Statistics about the search index."""

    total_files: int = Field(..., description="Total number of indexed files")
    total_chunks: int | str = Field(
        ..., description="Total number of code chunks (or status message for large DBs)"
    )
    languages: dict[str, int] = Field(..., description="Language distribution")
    file_types: dict[str, int] = Field(..., description="File type distribution")
    index_size_mb: float = Field(..., description="Index size in megabytes")
    last_updated: str = Field(..., description="Last update timestamp")
    embedding_model: str = Field(..., description="Embedding model used")
    database_size_bytes: int = Field(
        default=0, description="Raw database file size in bytes"
    )

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
            "database_size_bytes": self.database_size_bytes,
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

"""Data models for code story narrative generation.

StoryIndex is the central JSON artifact that connects backend extraction/analysis
to frontend rendering. All renderers consume this single schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TimelineEventType(StrEnum):
    COMMIT = "commit"
    ISSUE = "issue"
    PR = "pull_request"
    PIVOT = "pivot"
    MILESTONE = "milestone"


# --- Extraction Models ---


class CommitInfo(BaseModel):
    """A single git commit."""

    hash: str = Field(..., description="Full commit SHA")
    short_hash: str = Field(..., description="Short commit SHA (7 chars)")
    message: str = Field(..., description="Commit message (first line)")
    body: str = Field(default="", description="Full commit message body")
    author_name: str = Field(..., description="Author name")
    author_email: str = Field(default="", description="Author email")
    date: datetime = Field(..., description="Commit date (ISO 8601)")
    files_changed: int = Field(default=0, ge=0)
    insertions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    is_merge: bool = Field(default=False)
    files: list[str] = Field(
        default_factory=list, description="Files changed in this commit"
    )


class ContributorInfo(BaseModel):
    """Contributor statistics."""

    name: str
    email: str = ""
    commit_count: int = Field(default=0, ge=0)
    first_commit: datetime | None = None
    last_commit: datetime | None = None
    top_files: list[str] = Field(
        default_factory=list, description="Most frequently changed files"
    )


class IssueInfo(BaseModel):
    """GitHub issue."""

    number: int
    title: str
    state: str = "open"
    labels: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    closed_at: datetime | None = None
    body: str = ""


class PullRequestInfo(BaseModel):
    """GitHub pull request."""

    number: int
    title: str
    state: str = "open"
    merged_at: datetime | None = None
    files_changed: int = Field(default=0, ge=0)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    labels: list[str] = Field(default_factory=list)


class DocReference(BaseModel):
    """A documentation file found in the project."""

    path: str
    title: str = ""
    word_count: int = 0
    last_modified: datetime | None = None


# --- Analysis Models ---


class SemanticCluster(BaseModel):
    """A group of semantically related code areas."""

    name: str = Field(..., description="Human-readable cluster name")
    description: str = Field(default="", description="What this cluster represents")
    query: str = Field(default="", description="Search query that found this cluster")
    files: list[str] = Field(
        default_factory=list, description="File paths in this cluster"
    )
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    code_snippets: list[str] = Field(
        default_factory=list, description="Representative code snippets"
    )


class TechStackItem(BaseModel):
    """A technology or framework detected in the project."""

    name: str
    category: str = ""  # "language", "framework", "tool", "library"
    evidence: list[str] = Field(
        default_factory=list, description="Files/patterns that indicate this tech"
    )
    version: str | None = None


class ArchitecturalPattern(BaseModel):
    """An architectural pattern detected in the codebase."""

    name: str  # e.g. "Repository Pattern", "MVC", "Event-driven"
    description: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


class EvolutionPhase(BaseModel):
    """A phase in the project's evolution."""

    name: str
    date_start: datetime
    date_end: datetime
    description: str = ""
    key_commits: list[str] = Field(default_factory=list, description="Commit hashes")
    dominant_areas: list[str] = Field(
        default_factory=list, description="Code areas most active"
    )
    commit_count: int = 0


# --- Narrative Models ---


class NarrativeAct(BaseModel):
    """One act in the three-act narrative structure."""

    number: int = Field(..., ge=1, le=5, description="Act number (1-3 typically)")
    title: str = Field(..., description="Act title")
    date_range: str = Field(default="", description="e.g. 'Jan 2024 - Mar 2024'")
    content: str = Field(default="", description="Markdown narrative content")
    evidence: list[str] = Field(
        default_factory=list,
        description="Citations: commit hashes, issue #s, file paths",
    )
    key_commits: list[str] = Field(default_factory=list)


class Theme(BaseModel):
    """A recurring theme in the project's development."""

    name: str
    description: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence_count: int = Field(default=0, ge=0)
    examples: list[str] = Field(
        default_factory=list, description="Specific evidence: commits, issues, etc."
    )


class RoadNotTaken(BaseModel):
    """An alternative approach considered but not pursued."""

    title: str
    description: str = ""
    evidence: list[str] = Field(default_factory=list)


# --- Visualization Models ---


class TimelineEvent(BaseModel):
    """An event on the project timeline."""

    date: datetime
    event_type: TimelineEventType = TimelineEventType.COMMIT
    title: str
    description: str = ""
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContributorGraphNode(BaseModel):
    """Node in contributor collaboration graph."""

    name: str
    commit_count: int = 0
    files_touched: int = 0


class ContributorGraphEdge(BaseModel):
    """Edge connecting contributors who worked on same files."""

    source: str
    target: str
    shared_files: int = 0
    weight: float = 0.0


# --- Top-level Sections ---


class StoryMetadata(BaseModel):
    """Metadata about the story generation."""

    version: str = Field(default="1.0.0", description="StoryIndex schema version")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generator: str = Field(
        default="mcp-vector-search", description="Tool that generated this"
    )
    project_name: str = Field(default="", description="Project name")
    project_root: str = Field(default="", description="Absolute path to project root")
    git_remote: str = Field(default="", description="Git remote URL")
    git_branch: str = Field(default="", description="Current branch")
    git_commit: str = Field(default="", description="HEAD commit hash")
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None
    total_commits: int = 0
    total_contributors: int = 0
    total_files: int = 0
    generation_time_seconds: float = 0.0
    llm_model: str | None = None
    has_semantic_analysis: bool = False
    has_llm_narrative: bool = False


class StoryExtraction(BaseModel):
    """Raw extracted data from git/GitHub."""

    commits: list[CommitInfo] = Field(default_factory=list)
    contributors: list[ContributorInfo] = Field(default_factory=list)
    issues: list[IssueInfo] = Field(default_factory=list)
    pull_requests: list[PullRequestInfo] = Field(default_factory=list)
    docs: list[DocReference] = Field(default_factory=list)


class StoryAnalysis(BaseModel):
    """Semantic analysis results from vector search."""

    clusters: list[SemanticCluster] = Field(default_factory=list)
    tech_stack: list[TechStackItem] = Field(default_factory=list)
    architectural_patterns: list[ArchitecturalPattern] = Field(default_factory=list)
    evolution_phases: list[EvolutionPhase] = Field(default_factory=list)
    language_distribution: dict[str, int] = Field(
        default_factory=dict, description="language -> line count"
    )


class StoryNarrative(BaseModel):
    """LLM-generated narrative."""

    title: str = Field(default="", description="Generated story title")
    subtitle: str = Field(default="", description="One-line summary")
    executive_summary: str = Field(default="", description="2-3 paragraph summary")
    acts: list[NarrativeAct] = Field(
        default_factory=list, description="Three-act narrative"
    )
    themes: list[Theme] = Field(default_factory=list)
    roads_not_taken: list[RoadNotTaken] = Field(default_factory=list)
    conclusion: str = Field(default="", description="Closing reflection")
    confidence_assessment: str = Field(
        default="", description="Self-assessment of narrative quality"
    )


class StoryVisualization(BaseModel):
    """Pre-computed visualization data."""

    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    contributor_nodes: list[ContributorGraphNode] = Field(default_factory=list)
    contributor_edges: list[ContributorGraphEdge] = Field(default_factory=list)
    commit_density: dict[str, int] = Field(
        default_factory=dict, description="date_str -> commit_count"
    )
    activity_heatmap: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="contributor -> {date: count}"
    )


class StoryIndex(BaseModel):
    """Root schema: the complete story artifact.

    This is the contract between backend (extraction/analysis/synthesis)
    and frontend (renderers). Renderers ONLY consume this schema.
    """

    metadata: StoryMetadata = Field(default_factory=StoryMetadata)
    extraction: StoryExtraction = Field(default_factory=StoryExtraction)
    analysis: StoryAnalysis = Field(default_factory=StoryAnalysis)
    narrative: StoryNarrative = Field(default_factory=StoryNarrative)
    visualization: StoryVisualization = Field(default_factory=StoryVisualization)

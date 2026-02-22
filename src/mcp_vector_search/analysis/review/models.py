"""Data structures for AI-powered code review system."""

from dataclasses import dataclass, field
from enum import Enum


class ReviewType(str, Enum):
    """Type of code review to perform."""

    SECURITY = "security"
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"


class Severity(str, Enum):
    """Severity level of a finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReviewFinding:
    """A single finding from a code review."""

    title: str
    description: str
    severity: Severity
    file_path: str
    start_line: int
    end_line: int
    category: str  # e.g., "SQL Injection", "God Class", "N+1 Query"
    recommendation: str
    confidence: float  # 0.0-1.0
    cwe_id: str | None = None  # For security findings
    code_snippet: str | None = None
    related_files: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    """Complete result of a code review."""

    review_type: ReviewType
    findings: list[ReviewFinding]
    summary: str
    scope: str  # what was reviewed (path, module, etc.)
    context_chunks_used: int
    kg_relationships_used: int
    model_used: str
    duration_seconds: float

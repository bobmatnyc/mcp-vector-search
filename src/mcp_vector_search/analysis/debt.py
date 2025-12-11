"""Technical debt estimation based on code smells and metrics.

This module provides technical debt estimation functionality using the SQALE
methodology (similar to SonarQube). Technical debt is estimated as the time
required to fix all detected code smells and quality issues.

The estimation formula:
    Total Debt = Σ (smell_count × base_remediation_time × severity_multiplier)

Remediation times are based on industry research and conservative estimates
for how long it takes to properly fix each type of code smell.

Example:
    from mcp_vector_search.analysis.debt import TechnicalDebtEstimator
    from mcp_vector_search.analysis.collectors.smells import SmellDetector

    # Detect smells
    detector = SmellDetector()
    smells = detector.detect_all(file_metrics, file_path)

    # Estimate debt
    estimator = TechnicalDebtEstimator()
    summary = estimator.estimate_from_smells(smells)

    print(f"Total debt: {summary.total_hours:.1f} hours")
    print(f"By category: {summary.minutes_by_category}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collectors.smells import CodeSmell
    from .metrics import ProjectMetrics


class DebtCategory(str, Enum):
    """Categories of technical debt.

    These categories align with standard software quality models and help
    organize debt by the type of impact it has on the codebase.

    Attributes:
        COMPLEXITY: Issues related to code complexity and understandability
        MAINTAINABILITY: Issues affecting ease of modification and extension
        RELIABILITY: Issues that could lead to bugs or failures
        SECURITY: Security-related issues (placeholder for future)
        DOCUMENTATION: Missing or inadequate documentation
    """

    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    DOCUMENTATION = "documentation"


@dataclass
class RemediationTime:
    """Time estimate for fixing a specific smell type.

    Attributes:
        smell_type: Type of code smell (e.g., "long_method")
        base_minutes: Base time in minutes to fix this smell
        category: Debt category this smell belongs to
        description: Human-readable description of the remediation work
    """

    smell_type: str
    base_minutes: int
    category: DebtCategory
    description: str


@dataclass
class DebtItem:
    """A single technical debt item representing one code smell instance.

    Attributes:
        smell_type: Type of smell (normalized to snake_case)
        file_path: File where the smell was detected
        line: Line number where the smell occurs
        severity: Severity level ("error", "warning", "info")
        base_minutes: Base remediation time before severity adjustment
        adjusted_minutes: Final remediation time after severity multiplier
        category: Debt category
        message: Detailed message about the smell
    """

    smell_type: str
    file_path: str
    line: int
    severity: str
    base_minutes: int
    adjusted_minutes: float
    category: DebtCategory
    message: str


@dataclass
class DebtSummary:
    """Summary of technical debt for a project.

    Provides comprehensive debt metrics with multiple breakdowns and
    aggregations to help prioritize remediation efforts.

    Attributes:
        total_minutes: Total debt in minutes
        total_hours: Total debt in hours
        total_days: Total debt in work days (8-hour days)
        items_by_category: Debt items grouped by category
        minutes_by_category: Total minutes of debt per category
        items_by_severity: Debt items grouped by severity
        minutes_by_severity: Total minutes of debt per severity
        top_files: Files with most debt (file_path, minutes)
        top_smell_types: Most common smell types (smell_type, minutes)
        item_count: Total number of debt items
    """

    total_minutes: float
    total_hours: float
    total_days: float

    items_by_category: dict[DebtCategory, list[DebtItem]]
    minutes_by_category: dict[DebtCategory, float]

    items_by_severity: dict[str, list[DebtItem]]
    minutes_by_severity: dict[str, float]

    top_files: list[tuple[str, float]]  # (file_path, minutes)
    top_smell_types: list[tuple[str, float]]  # (smell_type, minutes)

    item_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "total_minutes": self.total_minutes,
            "total_hours": self.total_hours,
            "total_days": self.total_days,
            "item_count": self.item_count,
            "by_category": {
                category.value: {
                    "minutes": minutes,
                    "hours": round(minutes / 60, 2),
                    "item_count": len(self.items_by_category.get(category, [])),
                }
                for category, minutes in self.minutes_by_category.items()
            },
            "by_severity": {
                severity: {
                    "minutes": minutes,
                    "hours": round(minutes / 60, 2),
                    "item_count": len(self.items_by_severity.get(severity, [])),
                }
                for severity, minutes in self.minutes_by_severity.items()
            },
            "top_files": [
                {"file_path": path, "minutes": mins, "hours": round(mins / 60, 2)}
                for path, mins in self.top_files
            ],
            "top_smell_types": [
                {
                    "smell_type": smell,
                    "minutes": mins,
                    "hours": round(mins / 60, 2),
                }
                for smell, mins in self.top_smell_types
            ],
        }


class TechnicalDebtEstimator:
    """Estimates technical debt based on code smells.

    Uses the SQALE methodology to estimate the time required to fix all
    detected code smells. Remediation times are configurable and based on
    industry research and conservative estimates.

    Severity multipliers adjust base times:
    - error (critical): 1.5× (urgent, blocks development)
    - warning (major): 1.0× (standard remediation time)
    - info (minor): 0.5× (nice to have, low priority)

    Example:
        estimator = TechnicalDebtEstimator()
        summary = estimator.estimate_from_smells(smells)
        print(f"Total debt: {summary.total_hours:.1f} hours")
    """

    # Default remediation times based on SonarQube research
    # Times are conservative estimates including testing and review
    DEFAULT_REMEDIATION_TIMES: dict[str, RemediationTime] = {
        "long_method": RemediationTime(
            smell_type="long_method",
            base_minutes=20,
            category=DebtCategory.MAINTAINABILITY,
            description="Extract method refactoring, update tests",
        ),
        "deep_nesting": RemediationTime(
            smell_type="deep_nesting",
            base_minutes=15,
            category=DebtCategory.COMPLEXITY,
            description="Flatten control flow, extract guard clauses",
        ),
        "god_class": RemediationTime(
            smell_type="god_class",
            base_minutes=120,
            category=DebtCategory.MAINTAINABILITY,
            description="Split into smaller classes, refactor dependencies",
        ),
        "high_cognitive_complexity": RemediationTime(
            smell_type="high_cognitive_complexity",
            base_minutes=30,
            category=DebtCategory.COMPLEXITY,
            description="Simplify logic flow, extract helper functions",
        ),
        "high_cyclomatic_complexity": RemediationTime(
            smell_type="high_cyclomatic_complexity",
            base_minutes=25,
            category=DebtCategory.COMPLEXITY,
            description="Break into smaller functions, reduce branches",
        ),
        "complex_method": RemediationTime(
            smell_type="complex_method",
            base_minutes=25,
            category=DebtCategory.COMPLEXITY,
            description="Simplify control flow, reduce cyclomatic complexity",
        ),
        "circular_dependency": RemediationTime(
            smell_type="circular_dependency",
            base_minutes=60,
            category=DebtCategory.MAINTAINABILITY,
            description="Restructure module dependencies, introduce abstractions",
        ),
        "empty_catch": RemediationTime(
            smell_type="empty_catch",
            base_minutes=5,
            category=DebtCategory.RELIABILITY,
            description="Add proper error handling and logging",
        ),
        "magic_number": RemediationTime(
            smell_type="magic_number",
            base_minutes=5,
            category=DebtCategory.MAINTAINABILITY,
            description="Extract named constant with documentation",
        ),
        "long_parameter_list": RemediationTime(
            smell_type="long_parameter_list",
            base_minutes=15,
            category=DebtCategory.MAINTAINABILITY,
            description="Introduce parameter object or builder pattern",
        ),
        "duplicate_code": RemediationTime(
            smell_type="duplicate_code",
            base_minutes=30,
            category=DebtCategory.MAINTAINABILITY,
            description="Extract shared function, update call sites",
        ),
        "dead_code": RemediationTime(
            smell_type="dead_code",
            base_minutes=10,
            category=DebtCategory.MAINTAINABILITY,
            description="Remove dead code, verify no side effects",
        ),
        "missing_docstring": RemediationTime(
            smell_type="missing_docstring",
            base_minutes=10,
            category=DebtCategory.DOCUMENTATION,
            description="Write comprehensive documentation",
        ),
    }

    # Severity multipliers based on urgency and impact
    SEVERITY_MULTIPLIERS: dict[str, float] = {
        "error": 1.5,  # Critical: urgent fix required
        "warning": 1.0,  # Major: standard remediation time
        "info": 0.5,  # Minor: nice to have, lower priority
    }

    def __init__(
        self,
        remediation_times: dict[str, RemediationTime] | None = None,
        severity_multipliers: dict[str, float] | None = None,
    ) -> None:
        """Initialize estimator with optional custom remediation times.

        Args:
            remediation_times: Optional custom remediation time mapping.
                             If None, uses DEFAULT_REMEDIATION_TIMES.
            severity_multipliers: Optional custom severity multipliers.
                                If None, uses SEVERITY_MULTIPLIERS.
        """
        self.remediation_times = (
            remediation_times or self.DEFAULT_REMEDIATION_TIMES.copy()
        )
        self.severity_multipliers = (
            severity_multipliers or self.SEVERITY_MULTIPLIERS.copy()
        )

    def estimate_from_smells(self, smells: list[CodeSmell]) -> DebtSummary:
        """Calculate technical debt from a list of code smells.

        Analyzes each smell, applies remediation times and severity multipliers,
        and generates a comprehensive debt summary with multiple breakdowns.

        Args:
            smells: List of detected code smells

        Returns:
            DebtSummary with total debt and detailed breakdowns
        """
        debt_items: list[DebtItem] = []

        # Convert each smell to a debt item
        for smell in smells:
            # Normalize smell name to snake_case for lookup
            smell_type = self._normalize_smell_name(smell.name)

            # Get base remediation time
            base_minutes = self.get_remediation_time(smell_type)

            # Apply severity multiplier
            severity = smell.severity.value  # Convert enum to string
            adjusted_minutes = self.apply_severity_multiplier(base_minutes, severity)

            # Get category
            category = self._get_category(smell_type)

            # Extract file path and line from location
            file_path, line = self._parse_location(smell.location)

            debt_item = DebtItem(
                smell_type=smell_type,
                file_path=file_path,
                line=line,
                severity=severity,
                base_minutes=base_minutes,
                adjusted_minutes=adjusted_minutes,
                category=category,
                message=smell.description,
            )
            debt_items.append(debt_item)

        # Generate summary
        return self._create_summary(debt_items)

    def estimate_from_project_metrics(
        self, project_metrics: ProjectMetrics
    ) -> DebtSummary:
        """Calculate debt from full project metrics.

        Collects all smells from all files in the project and estimates
        total technical debt.

        Args:
            project_metrics: Project-wide metrics containing all file metrics

        Returns:
            DebtSummary with total debt and detailed breakdowns
        """
        from .collectors.smells import SmellDetector

        detector = SmellDetector()
        all_smells: list[CodeSmell] = []

        # Collect smells from all files
        for file_path, file_metrics in project_metrics.files.items():
            file_smells = detector.detect_all(file_metrics, file_path)
            all_smells.extend(file_smells)

        return self.estimate_from_smells(all_smells)

    def get_remediation_time(self, smell_type: str) -> int:
        """Get base remediation time for a smell type.

        Args:
            smell_type: Type of smell (normalized to snake_case)

        Returns:
            Base remediation time in minutes (default: 15 if unknown)
        """
        if smell_type in self.remediation_times:
            return self.remediation_times[smell_type].base_minutes
        return 15  # Default fallback for unknown smell types

    def apply_severity_multiplier(self, base_minutes: int, severity: str) -> float:
        """Apply severity multiplier to base time.

        Args:
            base_minutes: Base remediation time in minutes
            severity: Severity level ("error", "warning", "info")

        Returns:
            Adjusted time after applying severity multiplier
        """
        multiplier = self.severity_multipliers.get(severity, 1.0)
        return base_minutes * multiplier

    def _normalize_smell_name(self, smell_name: str) -> str:
        """Normalize smell name to snake_case for consistent lookup.

        Converts human-readable names like "Long Method" to "long_method".

        Args:
            smell_name: Human-readable smell name

        Returns:
            Normalized snake_case smell type
        """
        return smell_name.lower().replace(" ", "_")

    def _get_category(self, smell_type: str) -> DebtCategory:
        """Get debt category for a smell type.

        Args:
            smell_type: Type of smell (normalized)

        Returns:
            Appropriate debt category
        """
        if smell_type in self.remediation_times:
            return self.remediation_times[smell_type].category
        return DebtCategory.MAINTAINABILITY  # Default category

    def _parse_location(self, location: str) -> tuple[str, int]:
        """Parse location string to extract file path and line number.

        Location format: "file_path:line" or "file_path:line-range"

        Args:
            location: Location string from code smell

        Returns:
            Tuple of (file_path, line_number)
        """
        if ":" in location:
            parts = location.split(":", 1)
            file_path = parts[0]
            # Extract line number (handle ranges like "10-20")
            line_str = parts[1].split("-")[0] if "-" in parts[1] else parts[1]
            try:
                line = int(line_str)
            except ValueError:
                line = 0
            return file_path, line
        return location, 0

    def _create_summary(self, debt_items: list[DebtItem]) -> DebtSummary:
        """Create debt summary from debt items.

        Aggregates debt items by category, severity, file, and smell type.

        Args:
            debt_items: List of debt items to summarize

        Returns:
            Comprehensive debt summary
        """
        # Calculate totals
        total_minutes = sum(item.adjusted_minutes for item in debt_items)
        total_hours = total_minutes / 60
        total_days = total_hours / 8  # 8-hour work days

        # Group by category
        items_by_category: dict[DebtCategory, list[DebtItem]] = defaultdict(list)
        minutes_by_category: dict[DebtCategory, float] = defaultdict(float)

        for item in debt_items:
            items_by_category[item.category].append(item)
            minutes_by_category[item.category] += item.adjusted_minutes

        # Group by severity
        items_by_severity: dict[str, list[DebtItem]] = defaultdict(list)
        minutes_by_severity: dict[str, float] = defaultdict(float)

        for item in debt_items:
            items_by_severity[item.severity].append(item)
            minutes_by_severity[item.severity] += item.adjusted_minutes

        # Calculate top files
        file_minutes: dict[str, float] = defaultdict(float)
        for item in debt_items:
            file_minutes[item.file_path] += item.adjusted_minutes

        top_files = sorted(file_minutes.items(), key=lambda x: x[1], reverse=True)[:10]

        # Calculate top smell types
        smell_minutes: dict[str, float] = defaultdict(float)
        for item in debt_items:
            smell_minutes[item.smell_type] += item.adjusted_minutes

        top_smell_types = sorted(
            smell_minutes.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return DebtSummary(
            total_minutes=total_minutes,
            total_hours=total_hours,
            total_days=total_days,
            items_by_category=dict(items_by_category),
            minutes_by_category=dict(minutes_by_category),
            items_by_severity=dict(items_by_severity),
            minutes_by_severity=dict(minutes_by_severity),
            top_files=top_files,
            top_smell_types=top_smell_types,
            item_count=len(debt_items),
        )

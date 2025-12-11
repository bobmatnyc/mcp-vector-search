"""Baseline comparison for tracking metric changes over time.

This module implements baseline snapshot storage and comparison capabilities,
enabling developers and CI/CD pipelines to track code quality metrics against
a known-good state.

Key Components:
    - BaselineManager: Store/load/manage baseline snapshots
    - BaselineComparator: Compare current metrics against baseline
    - ComparisonResult: Structured comparison output with classifications

Usage Example:
    >>> from pathlib import Path
    >>> from mcp_vector_search.analysis.baseline import BaselineManager, BaselineComparator
    >>> from mcp_vector_search.analysis.metrics import ProjectMetrics
    >>>
    >>> # Save baseline
    >>> manager = BaselineManager()
    >>> metrics = analyze_project(Path.cwd())  # Your analysis function
    >>> manager.save_baseline("main-branch", metrics)
    >>>
    >>> # Compare against baseline
    >>> current_metrics = analyze_project(Path.cwd())
    >>> comparator = BaselineComparator()
    >>> baseline = manager.load_baseline("main-branch")
    >>> result = comparator.compare(current_metrics, baseline)
    >>> print(f"Regressions: {len(result.regressions)}")
    >>> print(f"Improvements: {len(result.improvements)}")

Design Decisions:
    - JSON storage for Phase 2 (human-readable, simple, no dependencies)
    - Storage location: ~/.mcp-vector-search/baselines/
    - Includes git metadata (commit, branch) for traceability
    - Includes tool version for compatibility checking
    - Graceful handling of incompatible baselines

Performance:
    - Save baseline: O(n) where n is number of files, ~50-100ms typical
    - Load baseline: O(n), ~20-50ms typical
    - Compare metrics: O(n + m) where n=files, m=functions, ~10-20ms typical

Future Enhancements (Phase 3):
    - Migrate to SQLite for better queryability (Issue #24)
    - Trend analysis across multiple baselines
    - Automated baseline creation on CI success
"""

from .comparator import BaselineComparator, ComparisonResult, MetricChange
from .manager import (
    BaselineCorruptedError,
    BaselineExistsError,
    BaselineManager,
    BaselineMetadata,
    BaselineNotFoundError,
)

__all__ = [
    # Manager
    "BaselineManager",
    "BaselineMetadata",
    "BaselineNotFoundError",
    "BaselineExistsError",
    "BaselineCorruptedError",
    # Comparator
    "BaselineComparator",
    "ComparisonResult",
    "MetricChange",
]

"""Storage module for metrics persistence and historical tracking.

This module provides SQLite-based storage for code metrics, enabling:
- Historical tracking of file and project metrics over time
- Trend analysis to identify improving/degrading code quality
- Snapshot comparison for release-to-release analysis
- Code smell tracking and remediation monitoring

Public API:
    - MetricsStore: Main storage interface
    - TrendTracker: Trend analysis and regression detection
    - ProjectSnapshot: Project-wide metrics at a point in time
    - TrendData: Trend analysis results with alerts
    - GitInfo: Git metadata for traceability

Exceptions:
    - MetricsStoreError: Base exception for storage errors
    - DatabaseLockedError: Database locked by another process
    - DuplicateEntryError: Attempted duplicate entry

Example Usage:
    >>> from mcp_vector_search.analysis.storage import MetricsStore, TrendTracker
    >>> from mcp_vector_search.analysis.metrics import ProjectMetrics
    >>>
    >>> # Initialize store (uses default ~/.mcp-vector-search/metrics.db)
    >>> store = MetricsStore()
    >>>
    >>> # Save complete snapshot
    >>> metrics = ProjectMetrics(project_root="/path/to/project")
    >>> # ... populate metrics ...
    >>> snapshot_id = store.save_complete_snapshot(metrics)
    >>>
    >>> # Query history
    >>> history = store.get_project_history("/path/to/project", limit=10)
    >>> for snapshot in history:
    ...     print(f"{snapshot.timestamp}: {snapshot.avg_complexity:.2f}")
    >>>
    >>> # Analyze trends with TrendTracker
    >>> tracker = TrendTracker(store, threshold_percentage=5.0)
    >>> trends = tracker.get_trends("/path/to/project", days=30)
    >>> if trends.improving:
    ...     print("Complexity is improving!")
    >>> if trends.has_regressions:
    ...     print(f"Found {len(trends.critical_regressions)} regressions")
    >>>
    >>> store.close()

Context Manager Usage:
    >>> with MetricsStore() as store:
    ...     snapshot_id = store.save_complete_snapshot(metrics)
    ...     # Connection automatically closed

See Also:
    - schema.py: Database schema definitions
    - metrics_store.py: MetricsStore implementation
    - trend_tracker.py: TrendTracker implementation
"""

from .metrics_store import (
    DatabaseLockedError,
    DuplicateEntryError,
    GitInfo,
    MetricsStore,
    MetricsStoreError,
    ProjectSnapshot,
)
from .schema import SCHEMA_VERSION
from .trend_tracker import FileRegression, TrendData, TrendDirection, TrendTracker

__all__ = [
    # Main storage class
    "MetricsStore",
    # Trend tracking
    "TrendTracker",
    "TrendDirection",
    # Data classes
    "ProjectSnapshot",
    "TrendData",
    "GitInfo",
    "FileRegression",
    # Exceptions
    "MetricsStoreError",
    "DatabaseLockedError",
    "DuplicateEntryError",
    # Schema version
    "SCHEMA_VERSION",
]

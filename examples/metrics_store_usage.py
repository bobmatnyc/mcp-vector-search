"""Example usage of MetricsStore for historical tracking.

This example demonstrates:
- Saving project metrics to SQLite database
- Querying historical data
- Analyzing trends over time
"""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import gettempdir

from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics
from mcp_vector_search.analysis.storage import MetricsStore


def create_sample_metrics(project_path: str, complexity: int) -> ProjectMetrics:
    """Create sample ProjectMetrics for demonstration.

    Args:
        project_path: Path to project root
        complexity: Base cognitive complexity for demonstration

    Returns:
        ProjectMetrics with sample data
    """
    # Create sample chunks with varying complexity
    chunk1 = ChunkMetrics(
        cognitive_complexity=complexity,
        cyclomatic_complexity=complexity // 2,
        max_nesting_depth=2,
        parameter_count=3,
        lines_of_code=30,
        smells=["example_smell"] if complexity > 15 else [],
    )

    chunk2 = ChunkMetrics(
        cognitive_complexity=complexity + 5,
        cyclomatic_complexity=(complexity + 5) // 2,
        max_nesting_depth=3,
        parameter_count=4,
        lines_of_code=50,
        smells=["high_complexity"] if complexity > 20 else [],
    )

    # Create file metrics
    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=80,
        comment_lines=15,
        blank_lines=5,
        function_count=2,
        class_count=1,
        method_count=3,
        chunks=[chunk1, chunk2],
    )
    file_metrics.compute_aggregates()

    # Create project metrics
    metrics = ProjectMetrics(
        project_root=project_path,
        analyzed_at=datetime.now(),
        files={"src/example.py": file_metrics},
    )
    metrics.compute_aggregates()

    return metrics


def main() -> None:
    """Demonstrate MetricsStore usage."""
    # Use temporary database for demonstration
    db_path = Path(gettempdir()) / "demo_metrics.db"

    print(f"Using database: {db_path}")
    print("-" * 60)

    # Initialize metrics store
    with MetricsStore(db_path=db_path) as store:
        project_path = "/demo/project"

        # Simulate 5 snapshots over time with varying complexity
        print("\n1. Saving historical snapshots...")
        for i in range(5):
            # Simulate snapshots from 4 days ago to today
            timestamp = datetime.now() - timedelta(days=4 - i)

            # Create metrics with increasing complexity (degrading code quality)
            base_complexity = 10 + i * 3
            metrics = create_sample_metrics(project_path, base_complexity)
            metrics.analyzed_at = timestamp

            snapshot_id = store.save_complete_snapshot(metrics)
            print(
                f"   Snapshot {snapshot_id}: {timestamp.date()} "
                f"(avg complexity: {metrics.avg_file_complexity:.1f})"
            )

        # Query project history
        print("\n2. Retrieving project history...")
        history = store.get_project_history(project_path, limit=5)

        print(f"   Found {len(history)} snapshots:")
        for snapshot in history:
            print(
                f"   - {snapshot.timestamp.date()}: "
                f"{snapshot.total_files} files, "
                f"avg complexity {snapshot.avg_complexity:.1f}"
            )

        # Query file history
        print("\n3. Retrieving file history for src/example.py...")
        file_history = store.get_file_history("src/example.py", limit=5)

        print(f"   Found {len(file_history)} history entries:")
        for file_metrics in file_history[:3]:  # Show first 3
            print(
                f"   - {file_metrics.function_count} functions, "
                f"avg complexity {file_metrics.avg_complexity:.1f}"
            )

        # Analyze trends
        print("\n4. Analyzing trends over last 30 days...")
        trends = store.get_trends(project_path, days=30)

        print(f"   Total snapshots: {len(trends.snapshots)}")
        print(f"   Change rate: {trends.change_rate:+.4f} complexity/day")

        if trends.improving:
            print("   ✅ Code quality is IMPROVING (complexity decreasing)")
        else:
            print("   ⚠️  Code quality is DEGRADING (complexity increasing)")

        # Show complexity trend
        print("\n   Complexity over time:")
        for timestamp, complexity in trends.complexity_trend:
            print(f"   - {timestamp.date()}: {complexity:.1f}")

    print("\n" + "=" * 60)
    print(f"Demo complete! Database saved at: {db_path}")
    print("You can inspect it with: sqlite3", db_path)
    print("=" * 60)


if __name__ == "__main__":
    main()

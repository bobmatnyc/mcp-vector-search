"""Tests for trend tracking functionality."""

import json

import pytest

from mcp_vector_search.analysis.trends import TrendEntry, TrendTracker


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root directory."""
    return tmp_path


@pytest.fixture
def trend_tracker(temp_project_root):
    """Create a TrendTracker instance."""
    return TrendTracker(temp_project_root)


def test_trend_entry_creation():
    """Test TrendEntry creation and serialization."""
    entry = TrendEntry(
        date="2025-12-15",
        timestamp="2025-12-15T10:00:00Z",
        metrics={"total_files": 100, "health_score": 85},
    )

    assert entry.date == "2025-12-15"
    assert entry.metrics["total_files"] == 100
    assert entry.metrics["health_score"] == 85

    # Test serialization
    data = entry.to_dict()
    assert data["date"] == "2025-12-15"
    assert data["metrics"]["total_files"] == 100


def test_load_trends_empty(trend_tracker):
    """Test loading trends when file doesn't exist."""
    trends = trend_tracker.load_trends()
    assert len(trends.entries) == 0
    assert trends.last_updated is None


def test_save_snapshot_creates_entry(trend_tracker):
    """Test saving a new snapshot."""
    metrics = {
        "total_files": 100,
        "total_chunks": 500,
        "avg_complexity": 5.2,
        "health_score": 85,
    }

    trend_tracker.save_snapshot(metrics)

    # Load and verify
    trends = trend_tracker.load_trends()
    assert len(trends.entries) == 1
    assert trends.entries[0].metrics["total_files"] == 100
    assert trends.entries[0].metrics["health_score"] == 85


def test_save_snapshot_updates_same_day(trend_tracker):
    """Test that reindexing on same day updates existing entry."""
    # First snapshot
    metrics1 = {"total_files": 100, "health_score": 85}
    trend_tracker.save_snapshot(metrics1)

    # Second snapshot same day (should update, not create new)
    metrics2 = {"total_files": 105, "health_score": 87}
    trend_tracker.save_snapshot(metrics2)

    # Should have only 1 entry (updated)
    trends = trend_tracker.load_trends()
    assert len(trends.entries) == 1
    assert trends.entries[0].metrics["total_files"] == 105
    assert trends.entries[0].metrics["health_score"] == 87


def test_get_history(trend_tracker):
    """Test retrieving trend history."""
    # Add multiple entries
    for i in range(5):
        metrics = {"total_files": 100 + i, "health_score": 80 + i}
        trend_tracker.save_snapshot(metrics)

    # Get last 3 entries
    history = trend_tracker.get_history(days=3)
    assert len(history) == 1  # Only 1 because we're on same day

    # Get all entries
    history_all = trend_tracker.get_history(days=0)
    assert len(history_all) == 1


def test_compute_metrics_from_stats(trend_tracker):
    """Test computing metrics from database stats."""

    class MockChunk:
        def __init__(self, complexity, smells, lines):
            self.cognitive_complexity = complexity
            self.smell_count = smells
            self.lines_of_code = lines
            self.start_line = 1
            self.end_line = 1 + lines

    stats = {"total_files": 50, "total_chunks": 200}

    chunks = [
        MockChunk(5, 0, 20),
        MockChunk(10, 1, 30),
        MockChunk(25, 2, 50),  # High complexity
    ]

    metrics = trend_tracker.compute_metrics_from_stats(stats, chunks)

    assert metrics["total_files"] == 50
    assert metrics["total_chunks"] == 200
    assert metrics["total_lines"] == 100  # Sum of lines
    assert metrics["avg_complexity"] > 0
    assert metrics["max_complexity"] == 25
    assert metrics["code_smells_count"] == 3
    assert metrics["high_complexity_files"] == 1  # One chunk > 20


def test_get_trend_summary(trend_tracker):
    """Test getting trend summary for visualization."""
    # Add snapshot
    metrics = {"total_files": 100, "health_score": 85}
    trend_tracker.save_snapshot(metrics)

    # Get summary
    summary = trend_tracker.get_trend_summary(days=30)

    assert summary["days"] == 30
    assert summary["entries_count"] == 1
    assert len(summary["entries"]) == 1
    assert summary["date_range"]["start"] is not None


def test_trends_file_persistence(trend_tracker, temp_project_root):
    """Test that trends file is created and persisted."""
    metrics = {"total_files": 100, "health_score": 85}
    trend_tracker.save_snapshot(metrics)

    # Check file exists
    trends_file = temp_project_root / ".mcp-vector-search" / "trends.json"
    assert trends_file.exists()

    # Load and verify raw JSON
    with open(trends_file) as f:
        data = json.load(f)

    assert "entries" in data
    assert len(data["entries"]) == 1
    assert data["last_updated"] is not None


def test_health_score_calculation(trend_tracker):
    """Test health score computation logic."""

    class MockChunk:
        def __init__(self, complexity, smells):
            self.cognitive_complexity = complexity
            self.smell_count = smells
            self.lines_of_code = 50
            self.start_line = 1
            self.end_line = 50

    stats = {"total_files": 10, "total_chunks": 10}

    # Test with low complexity, no smells (should be ~100)
    chunks_good = [MockChunk(3, 0) for _ in range(5)]
    metrics_good = trend_tracker.compute_metrics_from_stats(stats, chunks_good)
    assert metrics_good["health_score"] >= 90

    # Test with high complexity, many smells (should be lower)
    chunks_bad = [MockChunk(35, 5) for _ in range(5)]
    metrics_bad = trend_tracker.compute_metrics_from_stats(stats, chunks_bad)
    assert metrics_bad["health_score"] < 50

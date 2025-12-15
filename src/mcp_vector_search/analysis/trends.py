"""Historical trend tracking for code metrics over time.

Stores daily snapshots of key metrics to track codebase evolution.
At most one entry per day - updates existing entry if reindexed same day.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TrendEntry:
    """Single trend snapshot for a specific date.

    Attributes:
        date: ISO date string (YYYY-MM-DD)
        timestamp: Full ISO timestamp when captured
        metrics: Dictionary of metrics captured at this point
    """

    date: str
    timestamp: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendEntry:
        """Create from dictionary."""
        return cls(
            date=data["date"],
            timestamp=data["timestamp"],
            metrics=data.get("metrics", {}),
        )


@dataclass
class TrendData:
    """Container for all trend entries.

    Attributes:
        entries: List of trend snapshots (one per day)
        last_updated: ISO timestamp of most recent update
    """

    entries: list[TrendEntry] = field(default_factory=list)
    last_updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendData:
        """Create from dictionary."""
        return cls(
            entries=[TrendEntry.from_dict(e) for e in data.get("entries", [])],
            last_updated=data.get("last_updated"),
        )


class TrendTracker:
    """Track code metrics over time with daily snapshots.

    Features:
    - One entry per day (updates existing if reindexed same day)
    - Stores key metrics: files, chunks, lines, complexity, health
    - JSON file storage in .mcp-vector-search/trends.json
    - History retrieval for time series analysis
    """

    def __init__(self, project_root: Path) -> None:
        """Initialize trend tracker.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.trends_file = project_root / ".mcp-vector-search" / "trends.json"
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure .mcp-vector-search directory exists."""
        self.trends_file.parent.mkdir(parents=True, exist_ok=True)

    def load_trends(self) -> TrendData:
        """Load existing trends from file.

        Returns:
            TrendData with existing entries, or empty if file doesn't exist
        """
        if not self.trends_file.exists():
            logger.debug("No trends file found, returning empty TrendData")
            return TrendData()

        try:
            with open(self.trends_file, encoding="utf-8") as f:
                data = json.load(f)
                return TrendData.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load trends file: {e}")
            return TrendData()

    def save_trends(self, trends: TrendData) -> None:
        """Save trends to file.

        Args:
            trends: TrendData to save
        """
        try:
            self._ensure_directory()
            with open(self.trends_file, "w", encoding="utf-8") as f:
                json.dump(trends.to_dict(), f, indent=2)
            logger.debug(f"Saved trends to {self.trends_file}")
        except Exception as e:
            logger.error(f"Failed to save trends file: {e}")

    def save_snapshot(self, metrics: dict[str, Any]) -> None:
        """Save or update today's metrics snapshot.

        If an entry already exists for today, it's replaced (reindex case).
        Otherwise, a new entry is appended.

        Args:
            metrics: Dictionary of metrics to store
        """
        trends = self.load_trends()

        # Get current date and timestamp
        now = datetime.now(UTC)
        today_date = now.date().isoformat()  # YYYY-MM-DD
        timestamp = now.isoformat()

        # Check if entry already exists for today
        existing_index = None
        for i, entry in enumerate(trends.entries):
            if entry.date == today_date:
                existing_index = i
                break

        # Create new entry
        new_entry = TrendEntry(
            date=today_date,
            timestamp=timestamp,
            metrics=metrics,
        )

        # Replace existing or append new
        if existing_index is not None:
            logger.info(f"Updating existing trend entry for {today_date}")
            trends.entries[existing_index] = new_entry
        else:
            logger.info(f"Creating new trend entry for {today_date}")
            trends.entries.append(new_entry)

        # Sort entries by date (oldest first)
        trends.entries.sort(key=lambda e: e.date)

        # Update last_updated timestamp
        trends.last_updated = timestamp

        # Save to file
        self.save_trends(trends)
        logger.info(
            f"Saved trend snapshot with {len(metrics)} metrics for {today_date}"
        )

    def get_history(self, days: int = 30) -> list[TrendEntry]:
        """Get recent trend history.

        Args:
            days: Number of days to retrieve (default: 30)

        Returns:
            List of TrendEntry objects for the last N days
        """
        trends = self.load_trends()

        # Return last N entries
        return trends.entries[-days:] if days > 0 else trends.entries

    def get_trend_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of trends for visualization.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with trend summary data
        """
        history = self.get_history(days)

        if not history:
            return {
                "days": days,
                "entries_count": 0,
                "date_range": None,
                "entries": [],
            }

        return {
            "days": days,
            "entries_count": len(history),
            "date_range": {
                "start": history[0].date,
                "end": history[-1].date,
            },
            "entries": [entry.to_dict() for entry in history],
        }

    def compute_metrics_from_stats(
        self, stats: dict[str, Any], chunks: list[Any] | None = None
    ) -> dict[str, Any]:
        """Compute metrics dictionary from database stats and chunks.

        Args:
            stats: Database statistics (from database.get_stats())
            chunks: Optional list of chunks for detailed metrics

        Returns:
            Dictionary of metrics suitable for save_snapshot()
        """
        metrics = {
            "total_files": stats.get("total_files", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "total_lines": 0,  # Computed from chunks if available
            "avg_complexity": 0.0,
            "max_complexity": 0,
            "health_score": 0,
            "code_smells_count": 0,
            "high_complexity_files": 0,
        }

        # Compute detailed metrics from chunks if provided
        if chunks:
            total_lines = 0
            complexities = []
            smell_counts = 0
            high_complexity_count = 0

            for chunk in chunks:
                # Lines of code
                if hasattr(chunk, "lines_of_code") and chunk.lines_of_code:
                    total_lines += chunk.lines_of_code
                else:
                    # Fallback: estimate from line range
                    total_lines += chunk.end_line - chunk.start_line + 1

                # Complexity
                if (
                    hasattr(chunk, "cognitive_complexity")
                    and chunk.cognitive_complexity
                ):
                    complexities.append(chunk.cognitive_complexity)
                    # High complexity = cognitive > 20
                    if chunk.cognitive_complexity > 20:
                        high_complexity_count += 1

                # Code smells
                if hasattr(chunk, "smell_count") and chunk.smell_count:
                    smell_counts += chunk.smell_count

            metrics["total_lines"] = total_lines

            if complexities:
                metrics["avg_complexity"] = sum(complexities) / len(complexities)
                metrics["max_complexity"] = max(complexities)

            metrics["code_smells_count"] = smell_counts
            metrics["high_complexity_files"] = high_complexity_count

            # Compute health score (0-100)
            # Formula: Base 100, penalty for complexity and smells
            health = 100.0

            # Penalty for average complexity
            if metrics["avg_complexity"] > 30:
                health -= 50
            elif metrics["avg_complexity"] > 20:
                health -= 30
            elif metrics["avg_complexity"] > 10:
                health -= 20
            elif metrics["avg_complexity"] > 5:
                health -= 10

            # Penalty for code smells (5 points per smell, max 30 points)
            smell_penalty = min(30, smell_counts * 5)
            health -= smell_penalty

            metrics["health_score"] = max(0, int(health))

        return metrics

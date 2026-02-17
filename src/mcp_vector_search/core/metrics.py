"""Performance metrics tracking for indexing pipeline.

Tracks timing and throughput for all phases:
- Phase 1: Parsing and chunking
- Phase 2: Embedding generation
- Phase 3: Knowledge graph building
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PhaseMetrics:
    """Metrics for a single indexing phase."""

    phase_name: str
    item_count: int = 0  # Files for parsing, chunks for embedding, entities for KG
    duration_seconds: float = 0.0
    throughput: float = 0.0  # Items per second

    def calculate_throughput(self) -> None:
        """Calculate throughput from count and duration."""
        if self.duration_seconds > 0:
            self.throughput = self.item_count / self.duration_seconds
        else:
            self.throughput = 0.0


@dataclass
class IndexingMetrics:
    """Complete indexing metrics for all phases."""

    # Phase metrics
    parsing: PhaseMetrics = field(default_factory=lambda: PhaseMetrics("Parsing"))
    chunking: PhaseMetrics = field(default_factory=lambda: PhaseMetrics("Chunking"))
    embedding: PhaseMetrics = field(default_factory=lambda: PhaseMetrics("Embedding"))
    kg_build: PhaseMetrics = field(default_factory=lambda: PhaseMetrics("KG Build"))

    # Overall timing
    total_duration_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)

    # Counts
    total_files: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_entities: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "parsing": asdict(self.parsing),
            "chunking": asdict(self.chunking),
            "embedding": asdict(self.embedding),
            "kg_build": asdict(self.kg_build),
            "total_duration_seconds": self.total_duration_seconds,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "total_entities": self.total_entities,
        }

    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsTracker:
    """Singleton metrics tracker for indexing performance.

    Usage:
        tracker = MetricsTracker()

        with tracker.phase("parsing") as metrics:
            # Parse files
            metrics.item_count = len(files)

        tracker.log_summary()  # Print formatted summary table
        print(tracker.metrics.to_json())  # JSON output
    """

    _instance: "MetricsTracker | None" = None

    def __new__(cls) -> "MetricsTracker":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        if self._initialized:
            return

        self.metrics = IndexingMetrics()
        self._phase_start_times: dict[str, float] = {}
        self._initialized = True

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics = IndexingMetrics()
        self._phase_start_times.clear()

    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for tracking a phase.

        Args:
            phase_name: Phase name ("parsing", "chunking", "embedding", "kg_build")

        Yields:
            PhaseMetrics object to update during phase

        Example:
            with tracker.phase("parsing") as metrics:
                # Do parsing work
                metrics.item_count = 100
        """
        # Map phase name to metrics attribute
        phase_map = {
            "parsing": self.metrics.parsing,
            "chunking": self.metrics.chunking,
            "embedding": self.metrics.embedding,
            "kg_build": self.metrics.kg_build,
        }

        if phase_name not in phase_map:
            logger.warning(f"Unknown phase: {phase_name}")
            yield PhaseMetrics(phase_name)
            return

        phase_metrics = phase_map[phase_name]
        start_time = time.perf_counter()

        try:
            yield phase_metrics
        finally:
            # Calculate duration and throughput
            duration = time.perf_counter() - start_time
            phase_metrics.duration_seconds = duration
            phase_metrics.calculate_throughput()

            # Log phase completion
            logger.info(
                f"📊 {phase_metrics.phase_name}: {phase_metrics.item_count:,} items "
                f"in {duration:.1f}s ({phase_metrics.throughput:.1f} items/sec)"
            )

    def finalize(self) -> None:
        """Finalize metrics by calculating total duration."""
        self.metrics.total_duration_seconds = time.time() - self.metrics.start_time

        # Update counts from phases
        self.metrics.total_files = self.metrics.parsing.item_count
        self.metrics.total_chunks = self.metrics.chunking.item_count
        self.metrics.total_embeddings = self.metrics.embedding.item_count
        self.metrics.total_entities = self.metrics.kg_build.item_count

    def log_summary(self) -> None:
        """Log formatted summary table to console."""
        self.finalize()

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(
            title="📊 Performance Summary",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Phase", style="cyan", width=15)
        table.add_column("Count", justify="right", style="green", width=12)
        table.add_column("Time", justify="right", style="yellow", width=10)
        table.add_column("Throughput", justify="right", style="magenta", width=15)

        # Add phase rows
        for phase_name, phase_metrics in [
            ("Parsing", self.metrics.parsing),
            ("Chunking", self.metrics.chunking),
            ("Embedding", self.metrics.embedding),
            ("KG Build", self.metrics.kg_build),
        ]:
            if phase_metrics.item_count > 0:
                table.add_row(
                    phase_name,
                    f"{phase_metrics.item_count:,}",
                    f"{phase_metrics.duration_seconds:.1f}s",
                    f"{phase_metrics.throughput:.1f}/s",
                )

        # Add separator
        table.add_section()

        # Add total row
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{self.metrics.total_duration_seconds:.1f}s[/bold]",
            "",
        )

        console.print()
        console.print(table)
        console.print()

    def save_json(self, output_path: Path) -> None:
        """Save metrics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        self.finalize()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.metrics.to_json())

        logger.info(f"Metrics saved to {output_path}")


# Global singleton instance
_tracker: MetricsTracker | None = None


def get_metrics_tracker() -> MetricsTracker:
    """Get global metrics tracker instance.

    Returns:
        MetricsTracker singleton
    """
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker

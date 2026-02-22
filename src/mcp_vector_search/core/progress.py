"""Thread-safe progress tracking for indexing and KG build operations.

This module provides a simple, print-based progress tracker that is safe
to use with Kuzu (which is not thread-safe). It avoids Rich background
threads (Progress, Live) and uses simple console.print() statements.
"""

import sys
import time

from rich.console import Console


class ProgressTracker:
    """Thread-safe progress tracker using simple print statements.

    This tracker is designed to work with Kuzu, which is not thread-safe
    and cannot use Rich's background rendering threads (Progress, Live).

    Features:
    - Simple print-based output (no background threads)
    - Phase-based progress with clear markers
    - Optional verbose mode for debug output
    - Time tracking for each phase

    Example:
        tracker = ProgressTracker(console, verbose=False)
        tracker.start("Building Knowledge Graph", total_phases=3)

        tracker.phase("Scanning chunks")
        tracker.item(f"Processed {count} chunks")
        tracker.item(f"Found {entities} code entities", done=True)

        tracker.phase("Inserting entities")
        tracker.item(f"{count} code entities", done=True)

        tracker.complete("Knowledge graph built successfully", time_taken=6.2)
    """

    def __init__(self, console: Console, verbose: bool = False):
        """Initialize progress tracker.

        Args:
            console: Rich Console instance for formatted output
            verbose: Enable verbose debug output
        """
        self.console = console
        self.verbose = verbose
        self.current_phase = 0
        self.total_phases = 0
        self._phase_start_time: float | None = None
        self._start_time: float | None = None

    def start(self, title: str, total_phases: int) -> None:
        """Start tracking with title and phase count.

        Args:
            title: Title for progress tracking
            total_phases: Total number of phases to track
        """
        self.console.print(f"\n[bold]{title}[/bold]")
        self.console.print("━" * 50)
        self.total_phases = total_phases
        self.current_phase = 0
        self._start_time = time.time()

    def phase(self, name: str) -> None:
        """Start a new phase.

        Args:
            name: Name of the phase
        """
        # Print elapsed time for previous phase
        if self._phase_start_time is not None:
            elapsed = time.time() - self._phase_start_time
            self.console.print(f"[dim]  (completed in {elapsed:.1f}s)[/dim]")

        self.current_phase += 1
        self._phase_start_time = time.time()

        self.console.print(
            f"\n[cyan]Phase {self.current_phase}/{self.total_phases}: {name}[/cyan]"
        )

    def item(self, message: str, done: bool = False) -> None:
        """Log an item within current phase.

        Args:
            message: Message to display
            done: If True, use checkmark (✓); otherwise use arrow (→)
        """
        marker = "✓" if done else "→"
        style = "green" if done else "dim"
        self.console.print(f"  [{style}]{marker}[/{style}] {message}")

    def debug(self, message: str) -> None:
        """Log debug message (only if verbose enabled).

        Args:
            message: Debug message to display
        """
        if self.verbose:
            self.console.print(f"  [dim]{message}[/dim]")

    def warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Warning message to display
        """
        self.console.print(f"  [yellow]⚠[/yellow] {message}")

    def complete(self, summary: str, time_taken: float | None = None) -> None:
        """Mark tracking complete.

        Args:
            summary: Summary message
            time_taken: Optional total time taken (uses internal timer if not provided)
        """
        # Print elapsed time for final phase
        if self._phase_start_time is not None:
            elapsed = time.time() - self._phase_start_time
            self.console.print(f"[dim]  (completed in {elapsed:.1f}s)[/dim]")

        if time_taken is None and self._start_time is not None:
            time_taken = time.time() - self._start_time

        self.console.print(f"\n[green]✓ {summary}[/green]")
        if time_taken is not None:
            self.console.print(f"  Time: {time_taken:.1f}s")

    def progress_bar(
        self, current: int, total: int, prefix: str = "", width: int = 40
    ) -> None:
        """Display an inline progress bar that updates in place.

        This progress bar updates on the same line without spawning background threads,
        making it safe to use with Kuzu and other non-thread-safe libraries.

        Args:
            current: Current progress value
            total: Total value (100% completion)
            prefix: Text to display before the progress bar
            width: Width of the progress bar in characters (default: 40)

        Example:
            tracker.progress_bar(328, 730, prefix="Parsing files")
            # Output: Parsing files... ━━━━━━━━━╸          45% 328/730
        """
        if total == 0:
            return

        # Calculate percentage and filled width
        percentage = min(100, int((current / total) * 100))
        filled_width = int((current / total) * width)

        # Build progress bar with blocks
        filled = "━" * filled_width
        empty = " " * (width - filled_width)
        bar = f"{filled}{empty}"

        # Format output with percentage and counts
        output = f"\r  {prefix}... {bar} {percentage}% {current:,}/{total:,}"

        # Write directly to stderr (avoids buffering issues)
        sys.stderr.write(output)
        sys.stderr.flush()

        # Print newline when complete
        if current >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def progress_bar_with_eta(
        self,
        current: int,
        total: int,
        prefix: str = "",
        width: int = 40,
        start_time: float | None = None,
    ) -> None:
        """Display a progress bar with ETA estimate.

        Args:
            current: Current progress value
            total: Total value (100% completion)
            prefix: Text to display before the progress bar
            width: Width of the progress bar in characters (default: 40)
            start_time: Start time for ETA calculation (uses time.time())

        Example:
            start = time.time()
            tracker.progress_bar_with_eta(54234, 81238, "Embedding chunks", start_time=start)
            # Output: Embedding chunks... ━━━━━━━━━━━━━╸       67% 54,234/81,238 [01:23 remaining]
        """
        if total == 0:
            return

        # Calculate percentage and filled width
        percentage = min(100, int((current / total) * 100))
        filled_width = int((current / total) * width)

        # Build progress bar
        filled = "━" * filled_width
        empty = " " * (width - filled_width)
        bar = f"{filled}{empty}"

        # Calculate ETA if start_time provided and some progress made
        eta_str = ""
        if start_time and current > 0 and current < total:
            elapsed = time.time() - start_time
            # Only show ETA if we have at least 0.1 seconds of data (avoid division noise)
            if elapsed >= 0.1:
                rate = current / elapsed
                if rate > 0:
                    remaining = (total - current) / rate
                    # Format based on duration
                    if remaining >= 3600:  # >= 1 hour
                        hours = int(remaining // 3600)
                        minutes = int((remaining % 3600) // 60)
                        seconds = int(remaining % 60)
                        eta_str = (
                            f" [{hours:02d}:{minutes:02d}:{seconds:02d} remaining]"
                        )
                    else:
                        minutes = int(remaining // 60)
                        seconds = int(remaining % 60)
                        eta_str = f" [{minutes:02d}:{seconds:02d} remaining]"

        # Format output
        output = f"\r  {prefix}... {bar} {percentage}% {current:,}/{total:,}{eta_str}"

        # Write directly to stderr
        sys.stderr.write(output)
        sys.stderr.flush()

        # Print newline when complete
        if current >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()


class SimpleProgressTracker:
    """Ultra-minimal progress tracker for subprocess usage.

    This is even simpler than ProgressTracker - just tracks counters
    without any console output. Useful for subprocess KG builds where
    we want to track progress but minimize output clutter.
    """

    def __init__(self):
        """Initialize simple progress tracker."""
        self.entities_count = 0
        self.relationships_count = 0
        self.doc_sections_count = 0
        self.tags_count = 0
        self._start_time = time.time()

    def increment_entities(self, count: int = 1) -> None:
        """Increment entity count."""
        self.entities_count += count

    def increment_relationships(self, count: int = 1) -> None:
        """Increment relationship count."""
        self.relationships_count += count

    def increment_doc_sections(self, count: int = 1) -> None:
        """Increment doc section count."""
        self.doc_sections_count += count

    def increment_tags(self, count: int = 1) -> None:
        """Increment tag count."""
        self.tags_count += count

    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self._start_time

    def get_stats(self) -> dict:
        """Get current stats as dictionary."""
        return {
            "entities": self.entities_count,
            "relationships": self.relationships_count,
            "doc_sections": self.doc_sections_count,
            "tags": self.tags_count,
            "elapsed_time": self.elapsed_time(),
        }

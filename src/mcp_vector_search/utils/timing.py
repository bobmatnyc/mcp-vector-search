"""Timing utilities for performance measurement and optimization."""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    operation: str
    duration: float  # in seconds
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return self.duration * 1_000_000


class PerformanceProfiler:
    """Performance profiler for measuring and analyzing operation timings."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.results: list[TimingResult] = []
        self._active_timers: dict[str, float] = {}
        self._nested_level = 0

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        if operation in self._active_timers:
            logger.warning(f"Timer '{operation}' already active, overwriting")
        self._active_timers[operation] = time.perf_counter()

    def stop_timer(
        self, operation: str, metadata: dict[str, Any] | None = None
    ) -> TimingResult:
        """Stop timing an operation and record the result."""
        if operation not in self._active_timers:
            raise ValueError(f"Timer '{operation}' not found or not started")

        start_time = self._active_timers.pop(operation)
        duration = time.perf_counter() - start_time

        result = TimingResult(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    @contextmanager
    def time_operation(self, operation: str, metadata: dict[str, Any] | None = None):
        """Context manager for timing an operation."""
        indent = "  " * self._nested_level
        logger.debug(f"{indent}⏱️  Starting: {operation}")

        self._nested_level += 1
        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._nested_level -= 1

            result = TimingResult(
                operation=operation,
                duration=duration,
                timestamp=time.time(),
                metadata=metadata or {},
            )

            self.results.append(result)

            indent = "  " * self._nested_level
            logger.debug(f"{indent}✅ Completed: {operation} ({duration * 1000:.2f}ms)")

    @asynccontextmanager
    async def time_async_operation(
        self, operation: str, metadata: dict[str, Any] | None = None
    ):
        """Async context manager for timing an operation."""
        indent = "  " * self._nested_level
        logger.debug(f"{indent}⏱️  Starting: {operation}")

        self._nested_level += 1
        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._nested_level -= 1

            result = TimingResult(
                operation=operation,
                duration=duration,
                timestamp=time.time(),
                metadata=metadata or {},
            )

            self.results.append(result)

            indent = "  " * self._nested_level
            logger.debug(f"{indent}✅ Completed: {operation} ({duration * 1000:.2f}ms)")

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get timing statistics for operations."""
        if operation:
            durations = [r.duration for r in self.results if r.operation == operation]
        else:
            durations = [r.duration for r in self.results]

        if not durations:
            return {}

        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "p95": (
                statistics.quantiles(durations, n=20)[18]
                if len(durations) >= 20
                else max(durations)
            ),
            "p99": (
                statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else max(durations)
            ),
        }

    def get_operation_breakdown(self) -> dict[str, dict[str, Any]]:
        """Get breakdown of all operations."""
        operations = {r.operation for r in self.results}
        return {op: self.get_stats(op) for op in operations}

    def print_report(self, show_individual: bool = False, min_duration_ms: float = 0.0):
        """Print a detailed performance report."""
        if not self.results:
            print("No timing results recorded.")
            return

        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE REPORT: {self.name}")
        print(f"{'=' * 60}")

        # Overall stats
        overall_stats = self.get_stats()
        print("\nOVERALL STATISTICS:")
        print(f"  Total operations: {overall_stats['count']}")
        print(f"  Total time: {overall_stats['total'] * 1000:.2f}ms")
        print(f"  Average: {overall_stats['mean'] * 1000:.2f}ms")
        print(f"  Median: {overall_stats['median'] * 1000:.2f}ms")
        print(f"  Min: {overall_stats['min'] * 1000:.2f}ms")
        print(f"  Max: {overall_stats['max'] * 1000:.2f}ms")

        # Per-operation breakdown
        breakdown = self.get_operation_breakdown()
        print("\nPER-OPERATION BREAKDOWN:")

        for operation, stats in sorted(
            breakdown.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            print(f"\n  {operation}:")
            print(f"    Count: {stats['count']}")
            print(
                f"    Total: {stats['total'] * 1000:.2f}ms ({stats['total'] / overall_stats['total'] * 100:.1f}%)"
            )
            print(f"    Average: {stats['mean'] * 1000:.2f}ms")
            print(
                f"    Min/Max: {stats['min'] * 1000:.2f}ms / {stats['max'] * 1000:.2f}ms"
            )
            if stats["count"] > 1:
                print(f"    StdDev: {stats['std_dev'] * 1000:.2f}ms")

        # Individual results if requested
        if show_individual:
            print("\nINDIVIDUAL RESULTS:")
            for result in self.results:
                if result.duration_ms >= min_duration_ms:
                    print(f"  {result.operation}: {result.duration_ms:.2f}ms")
                    if result.metadata:
                        print(f"    Metadata: {result.metadata}")

    def save_results(self, file_path: Path):
        """Save timing results to a JSON file."""
        data = {
            "profiler_name": self.name,
            "timestamp": time.time(),
            "results": [
                {
                    "operation": r.operation,
                    "duration": r.duration,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "stats": self.get_operation_breakdown(),
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def clear(self):
        """Clear all timing results."""
        self.results.clear()
        self._active_timers.clear()
        self._nested_level = 0


# Global profiler instance
_global_profiler = PerformanceProfiler("global")


def time_function(
    operation_name: str | None = None, metadata: dict[str, Any] | None = None
):
    """Decorator for timing function execution."""

    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                async with _global_profiler.time_async_operation(name, metadata):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with _global_profiler.time_operation(name, metadata):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


@contextmanager
def time_block(operation: str, metadata: dict[str, Any] | None = None):
    """Context manager for timing a block of code using the global profiler."""
    with _global_profiler.time_operation(operation, metadata):
        yield


@asynccontextmanager
async def time_async_block(operation: str, metadata: dict[str, Any] | None = None):
    """Async context manager for timing a block of code using the global profiler."""
    async with _global_profiler.time_async_operation(operation, metadata):
        yield


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def print_global_report(**kwargs):
    """Print report from the global profiler."""
    _global_profiler.print_report(**kwargs)


def clear_global_profiler():
    """Clear the global profiler."""
    _global_profiler.clear()


class SearchProfiler(PerformanceProfiler):
    """Specialized profiler for search operations."""

    def __init__(self):
        super().__init__("search_profiler")

    async def profile_search(
        self, search_func: Callable, query: str, **search_kwargs
    ) -> tuple[Any, dict[str, float]]:
        """Profile a complete search operation with detailed breakdown."""

        async with self.time_async_operation(
            "total_search", {"query": query, "kwargs": search_kwargs}
        ):
            # Time the actual search
            async with self.time_async_operation("search_execution", {"query": query}):
                result = await search_func(query, **search_kwargs)

            # Time result processing if we can measure it
            async with self.time_async_operation(
                "result_processing",
                {"result_count": len(result) if hasattr(result, "__len__") else 0},
            ):
                # Simulate any post-processing that might happen
                await asyncio.sleep(0)  # Placeholder for actual processing

        # Return results and timing breakdown
        timing_breakdown = {
            op: self.get_stats(op)["mean"] * 1000  # Convert to ms
            for op in ["total_search", "search_execution", "result_processing"]
            if self.get_stats(op)
        }

        return result, timing_breakdown


# Convenience function for quick search profiling
async def profile_search_operation(
    search_func: Callable, query: str, **kwargs
) -> tuple[Any, dict[str, float]]:
    """Quick function to profile a search operation."""
    profiler = SearchProfiler()
    return await profiler.profile_search(search_func, query, **kwargs)

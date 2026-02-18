#!/usr/bin/env python3
"""Benchmark script to measure pattern matching performance improvements."""

import time
from pathlib import Path

from src.mcp_vector_search.core.file_discovery import FileDiscovery


def benchmark_file_discovery(num_iterations: int = 3) -> dict[str, float]:
    """Benchmark file discovery with compiled patterns.

    Args:
        num_iterations: Number of iterations to average

    Returns:
        Dict with timing statistics
    """
    project_root = Path(".")
    file_extensions = {".py", ".js", ".ts", ".java", ".kt"}

    # Clear any caches
    fd = FileDiscovery(project_root, file_extensions)
    fd.clear_cache()

    timings = []
    file_counts = []

    print(f"\nBenchmarking file discovery with {len(fd._ignore_patterns)} patterns...")
    print(f"Pattern buckets: {len(fd._compiled_patterns)}")
    print(
        f"Total compiled patterns: {sum(len(v) for v in fd._compiled_patterns.values())}"
    )

    for i in range(num_iterations):
        print(f"\nIteration {i + 1}/{num_iterations}...")
        fd.clear_cache()  # Clear cache for fair comparison

        start = time.time()
        files = fd.scan_files_sync()
        elapsed = time.time() - start

        timings.append(elapsed)
        file_counts.append(len(files))

        files_per_sec = len(files) / elapsed if elapsed > 0 else 0
        print(
            f"  Found {len(files)} files in {elapsed:.3f}s ({files_per_sec:.1f} files/sec)"
        )

    avg_time = sum(timings) / len(timings)
    avg_files = sum(file_counts) / len(file_counts)
    avg_files_per_sec = avg_files / avg_time if avg_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("Average Results:")
    print(f"  Time: {avg_time:.3f}s")
    print(f"  Files: {avg_files:.0f}")
    print(f"  Throughput: {avg_files_per_sec:.1f} files/sec")
    print(f"{'=' * 60}")

    return {
        "avg_time": avg_time,
        "avg_files": avg_files,
        "avg_files_per_sec": avg_files_per_sec,
        "timings": timings,
    }


if __name__ == "__main__":
    results = benchmark_file_discovery(num_iterations=3)

    print("\n✅ Benchmark complete!")
    print("\nTarget: >100 files/sec")
    print(f"Actual: {results['avg_files_per_sec']:.1f} files/sec")

    if results["avg_files_per_sec"] >= 100:
        print("✅ Performance target MET!")
    else:
        print("❌ Performance target NOT met - needs further optimization")

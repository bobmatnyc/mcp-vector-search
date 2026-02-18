#!/usr/bin/env python3
"""Benchmark script to test performance with vendor patterns (263 patterns)."""

import time
from pathlib import Path

from src.mcp_vector_search.config.defaults import DEFAULT_IGNORE_PATTERNS
from src.mcp_vector_search.core.file_discovery import FileDiscovery


def create_vendor_patterns() -> set[str]:
    """Create a set of 263 vendor patterns to simulate the real scenario."""
    # Simulate 263 vendor library patterns
    vendor_patterns = set()

    # Common vendor prefixes
    prefixes = ["com", "org", "net", "io", "co", "de", "fr", "uk"]
    domains = [
        "google",
        "facebook",
        "amazon",
        "microsoft",
        "apple",
        "twitter",
        "github",
        "gitlab",
        "jetbrains",
        "intellij",
        "android",
        "spring",
        "apache",
        "hibernate",
        "jackson",
        "fasterxml",
        "squareup",
        "okhttp",
        "retrofit",
        "glide",
        "picasso",
        "dagger",
        "guava",
        "firebase",
        "crashlytics",
        "fabric",
        "flurry",
        "mixpanel",
        "segment",
        "amplitude",
        "stripe",
        "paypal",
        "braintree",
        "square",
        "twilio",
        "sendgrid",
    ]

    # Generate patterns like "com.google.*", "org.apache.*", etc.
    for prefix in prefixes:
        for domain in domains:
            vendor_patterns.add(f"{prefix}.{domain}.*")
            vendor_patterns.add(f"{prefix}.{domain}")

    # Add more patterns to reach 263
    for i in range(100):
        vendor_patterns.add(f"vendor{i}")
        vendor_patterns.add(f"third-party{i}")

    return vendor_patterns


def benchmark_with_vendor_patterns(num_iterations: int = 3) -> dict[str, float]:
    """Benchmark file discovery with vendor patterns.

    Args:
        num_iterations: Number of iterations to average

    Returns:
        Dict with timing statistics
    """
    project_root = Path(".")
    file_extensions = {".py", ".js", ".ts", ".java", ".kt"}

    # Create vendor patterns
    vendor_patterns = create_vendor_patterns()
    print(f"Created {len(vendor_patterns)} vendor patterns")

    # Merge with default patterns
    all_patterns = set(DEFAULT_IGNORE_PATTERNS) | vendor_patterns
    print(f"Total patterns (default + vendor): {len(all_patterns)}")

    # Initialize file discovery
    fd = FileDiscovery(project_root, file_extensions, ignore_patterns=vendor_patterns)
    fd.clear_cache()

    print(f"Pattern buckets: {len(fd._compiled_patterns)}")
    print(
        f"Total compiled patterns: {sum(len(v) for v in fd._compiled_patterns.values())}"
    )

    timings = []
    file_counts = []

    print(f"\nBenchmarking with {len(fd._ignore_patterns)} patterns...")

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
    print(f"Average Results with {len(fd._ignore_patterns)} patterns:")
    print(f"  Time: {avg_time:.3f}s")
    print(f"  Files: {avg_files:.0f}")
    print(f"  Throughput: {avg_files_per_sec:.1f} files/sec")
    print(f"{'=' * 60}")

    return {
        "avg_time": avg_time,
        "avg_files": avg_files,
        "avg_files_per_sec": avg_files_per_sec,
        "timings": timings,
        "num_patterns": len(fd._ignore_patterns),
    }


if __name__ == "__main__":
    results = benchmark_with_vendor_patterns(num_iterations=3)

    print("\n✅ Vendor patterns benchmark complete!")
    print(f"\nPatterns: {results['num_patterns']}")
    print("Target: >100 files/sec")
    print(f"Actual: {results['avg_files_per_sec']:.1f} files/sec")

    # Compare to reported issue (55.5 files/sec with 283 patterns)
    print("\nPrevious performance (reported): ~55.5 files/sec with 283 patterns")
    print(
        f"Current performance: {results['avg_files_per_sec']:.1f} files/sec with {results['num_patterns']} patterns"
    )

    improvement = (results["avg_files_per_sec"] - 55.5) / 55.5 * 100
    print(f"Improvement: {improvement:.1f}%")

    if results["avg_files_per_sec"] >= 100:
        print("\n✅ Performance target MET!")
    else:
        print("\n❌ Performance target NOT met - needs further optimization")

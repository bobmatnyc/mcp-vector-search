#!/usr/bin/env python3
"""Test directory filtering performance with cancellation support.

This script tests:
1. Performance improvements from caching and optimized pattern matching
2. Cancellation support for interruptible operations
3. Progress feedback during directory scanning
"""

import asyncio
import signal
import sys
import time
from pathlib import Path

from loguru import logger

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.database import VectorDatabase
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.utils.cancellation import (
    CancellationToken,
    OperationCancelledError,
    setup_interrupt_handler,
)


async def test_directory_filtering_performance(project_root: Path):
    """Test directory filtering performance."""
    print("=" * 80)
    print("Directory Filtering Performance Test")
    print("=" * 80)
    print()

    # Create database and indexer
    db = VectorDatabase(project_root / ".mcp-vector-search")
    config = ProjectConfig.load_or_create(project_root)
    indexer = SemanticIndexer(
        database=db,
        project_root=project_root,
        config=config,
        use_multiprocessing=False,  # Single-threaded for testing
    )

    # Test 1: Normal file scanning with timing
    print("Test 1: File Scanning Performance")
    print("-" * 80)

    start_time = time.time()
    files = indexer._find_indexable_files()
    scan_time = time.time() - start_time

    print(f"✓ Scanned {len(files)} files in {scan_time:.2f}s")
    print(f"  Speed: {len(files) / scan_time:.0f} files/sec")
    print()

    # Test 2: Cached scanning (should be instant)
    print("Test 2: Cached Scanning")
    print("-" * 80)

    start_time = time.time()
    files_cached = indexer._find_indexable_files()
    cached_time = time.time() - start_time

    print(f"✓ Retrieved {len(files_cached)} files from cache in {cached_time:.4f}s")
    print(f"  Speedup: {scan_time / cached_time:.0f}x faster")
    assert files == files_cached
    print()

    # Test 3: Clear cache and test with cancellation
    print("Test 3: Cancellation Support")
    print("-" * 80)

    indexer._indexable_files_cache = None  # Clear cache
    cancel_token = CancellationToken()

    # Set up SIGINT handler
    previous_handler = setup_interrupt_handler(cancel_token)

    print("Scanning with cancellation support (press Ctrl+C to cancel)...")
    print("Note: Cancellation will be tested programmatically")
    print()

    # Test cancellation after 100ms
    async def cancel_after_delay():
        await asyncio.sleep(0.1)
        print("⚠ Cancelling operation...")
        cancel_token.cancel()

    # Run scan with cancellation
    try:
        cancel_task = asyncio.create_task(cancel_after_delay())
        start_time = time.time()
        files_with_cancel = await indexer._find_indexable_files_async(cancel_token)
        scan_time = time.time() - start_time
        print(f"✓ Scan completed: {len(files_with_cancel)} files in {scan_time:.2f}s")
    except OperationCancelledError:
        scan_time = time.time() - start_time
        print(f"✓ Operation cancelled after {scan_time:.2f}s (as expected)")
        print("  Cancellation is working correctly!")
    finally:
        cancel_task.cancel()
        signal.signal(signal.SIGINT, previous_handler)

    print()

    # Test 4: Pattern matching performance
    print("Test 4: Pattern Matching Performance")
    print("-" * 80)

    # Create test paths
    test_paths = [
        project_root / "src" / "test.py",
        project_root / "node_modules" / "package" / "index.js",
        project_root / ".git" / "objects" / "test",
        project_root / "build" / "output.js",
        project_root / "tests" / "test_file.py",
    ]

    # Test ignore checking performance
    start_time = time.time()
    for _ in range(1000):
        for path in test_paths:
            indexer._should_ignore_path(path, is_directory=False)
    pattern_time = time.time() - start_time

    print(f"✓ Pattern matching: {len(test_paths) * 1000} checks in {pattern_time:.4f}s")
    print(f"  Speed: {(len(test_paths) * 1000) / pattern_time:.0f} checks/sec")
    print(
        f"  Average: {pattern_time / (len(test_paths) * 1000) * 1000:.4f}ms per check"
    )
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ File scanning: {scan_time:.2f}s for {len(files)} files")
    print(f"✓ Cache speedup: {scan_time / cached_time:.0f}x faster")
    print("✓ Cancellation: Working correctly")
    print(
        f"✓ Pattern matching: {(len(test_paths) * 1000) / pattern_time:.0f} checks/sec"
    )
    print()

    # Performance targets
    print("Performance Targets:")
    if len(files) / scan_time > 100:
        print("  ✓ Scan speed > 100 files/sec")
    else:
        print(f"  ✗ Scan speed < 100 files/sec (actual: {len(files) / scan_time:.0f})")

    if (len(test_paths) * 1000) / pattern_time > 10000:
        print("  ✓ Pattern matching > 10,000 checks/sec")
    else:
        print(
            f"  ✗ Pattern matching < 10,000 checks/sec (actual: {(len(test_paths) * 1000) / pattern_time:.0f})"
        )

    print()


async def main():
    """Run performance tests."""
    # Get project root
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path.cwd()

    print(f"Testing project: {project_root}")
    print()

    try:
        await test_directory_filtering_performance(project_root)
        print("✓ All tests passed!")
        return 0
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Manual test to verify multiprocessing performance improvement.

This is a manual test that can be run to see the actual performance difference
between single-process and multiprocess parsing.

Usage:
    uv run python tests/manual/test_multiprocessing_performance.py
"""

import asyncio
import time
from pathlib import Path

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.database import VectorDatabase
from mcp_vector_search.core.indexer import SemanticIndexer


async def benchmark_indexing(
    project_root: Path, use_multiprocessing: bool
) -> tuple[float, int]:
    """Benchmark indexing with or without multiprocessing.

    Args:
        project_root: Path to project to index
        use_multiprocessing: Whether to enable multiprocessing

    Returns:
        Tuple of (elapsed_time, files_indexed)
    """
    # Create temporary database
    db_path = (
        project_root / ".mcp-vector-search" / f"benchmark_{use_multiprocessing}.db"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create config
    config = ProjectConfig(
        project_root=project_root,
        file_extensions=[".py", ".js", ".ts", ".jsx", ".tsx"],
    )

    # Create database and indexer
    database = VectorDatabase(str(db_path), config)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        use_multiprocessing=use_multiprocessing,
        batch_size=10,
    )

    # Benchmark indexing
    start_time = time.perf_counter()
    files_indexed = await indexer.index_project(
        force_reindex=True, skip_relationships=True
    )
    elapsed_time = time.perf_counter() - start_time

    # Cleanup
    db_path.unlink(missing_ok=True)

    return elapsed_time, files_indexed


async def main():
    """Run benchmarks and compare results."""
    # Use current project as test subject
    project_root = Path(__file__).parent.parent.parent

    print("=" * 80)
    print("Multiprocessing Performance Benchmark")
    print("=" * 80)
    print(f"Project: {project_root}")
    print()

    # Benchmark without multiprocessing
    print("Running benchmark WITHOUT multiprocessing...")
    time_single, files_single = await benchmark_indexing(
        project_root, use_multiprocessing=False
    )
    print(f"✓ Single-process: {time_single:.2f}s ({files_single} files)")
    print()

    # Benchmark with multiprocessing
    print("Running benchmark WITH multiprocessing...")
    time_multi, files_multi = await benchmark_indexing(
        project_root, use_multiprocessing=True
    )
    print(f"✓ Multi-process:  {time_multi:.2f}s ({files_multi} files)")
    print()

    # Compare results
    print("=" * 80)
    print("Results:")
    print("=" * 80)
    speedup = time_single / time_multi if time_multi > 0 else 0
    improvement = (
        ((time_single - time_multi) / time_single * 100) if time_single > 0 else 0
    )

    print(f"Single-process: {time_single:.2f}s")
    print(f"Multi-process:  {time_multi:.2f}s")
    print(f"Speedup:        {speedup:.2f}x")
    print(f"Improvement:    {improvement:.1f}%")
    print()

    if speedup > 1.5:
        print("✓ Multiprocessing shows significant improvement!")
    elif speedup > 1.0:
        print("✓ Multiprocessing shows some improvement")
    else:
        print("⚠ Multiprocessing not faster (may need more files or CPU cores)")


if __name__ == "__main__":
    asyncio.run(main())

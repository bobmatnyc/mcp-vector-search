#!/usr/bin/env python3
"""Demo script showing pipeline parallelism in index_project().

This script demonstrates the new pipeline parameter that overlaps
Phase 1 (parsing/chunking) with Phase 2 (embedding) for 30-50% speedup.
"""

import asyncio
import time
from pathlib import Path

from mcp_vector_search.core.indexer import SemanticIndexer


async def demo_pipeline_vs_sequential():
    """Compare pipeline mode vs sequential mode performance."""
    project_root = Path(".")  # Current project

    print("=" * 80)
    print("Pipeline Parallelism Demo")
    print("=" * 80)
    print()
    print("This demo shows the difference between:")
    print("  1. Pipeline mode (default): Phase 1 and Phase 2 overlap")
    print("  2. Sequential mode: Phase 1 completes before Phase 2 starts")
    print()

    # Initialize indexer
    indexer = SemanticIndexer(project_root)

    # Demo 1: Pipeline mode (default)
    print("-" * 80)
    print("Demo 1: Pipeline Mode (pipeline=True, default)")
    print("-" * 80)
    print()
    print("Timeline:")
    print("[====== Phase 1: Parsing ======]")
    print("  [====== Phase 2: Embedding ======]")
    print()

    start_time = time.time()
    try:
        files_indexed = await indexer.index_project(
            force_reindex=False,
            pipeline=True,  # Explicit (same as default)
        )
        pipeline_time = time.time() - start_time

        print()
        print(f"✓ Pipeline mode completed in {pipeline_time:.2f}s")
        print(f"  Files indexed: {files_indexed}")
    except Exception as e:
        print(f"✗ Pipeline mode failed: {e}")
        return

    # Demo 2: Sequential mode (fallback)
    print()
    print("-" * 80)
    print("Demo 2: Sequential Mode (pipeline=False)")
    print("-" * 80)
    print()
    print("Timeline:")
    print("[====== Phase 1: Parsing ======][====== Phase 2: Embedding ======]")
    print()

    # Reinitialize to clear any caching
    indexer = SemanticIndexer(project_root)

    start_time = time.time()
    try:
        files_indexed = await indexer.index_project(
            force_reindex=False,
            pipeline=False,  # Disable pipeline
        )
        sequential_time = time.time() - start_time

        print()
        print(f"✓ Sequential mode completed in {sequential_time:.2f}s")
        print(f"  Files indexed: {files_indexed}")
    except Exception as e:
        print(f"✗ Sequential mode failed: {e}")
        return

    # Compare results
    print()
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"  Pipeline mode:   {pipeline_time:.2f}s")
    print(f"  Sequential mode: {sequential_time:.2f}s")

    if sequential_time > pipeline_time:
        speedup_pct = ((sequential_time - pipeline_time) / sequential_time) * 100
        print(f"  Speedup:         {speedup_pct:.1f}% faster with pipeline")
    elif pipeline_time > sequential_time:
        slowdown_pct = ((pipeline_time - sequential_time) / pipeline_time) * 100
        print(f"  Note:            {slowdown_pct:.1f}% slower with pipeline (overhead)")
    else:
        print("  Result:          Same performance (likely cached)")

    print()
    print("Note: On incremental indexes (no changes), both modes may be fast")
    print("      due to change detection. Use force_reindex=True for full benchmark.")


async def demo_phase_specific():
    """Demo phase-specific indexing (no pipeline)."""
    print()
    print("=" * 80)
    print("Demo 3: Phase-Specific Indexing")
    print("=" * 80)
    print()
    print("Pipeline is only used when phase='all' (both parsing and embedding).")
    print("For phase-specific indexing, sequential execution is used:")
    print()

    project_root = Path(".")
    indexer = SemanticIndexer(project_root)

    # Phase 1 only (no pipeline, no embedding)
    print("-" * 40)
    print("Phase 1 Only (phase='chunk')")
    print("-" * 40)
    try:
        files_indexed = await indexer.index_project(phase="chunk")
        print(f"✓ Phase 1 complete: {files_indexed} files chunked")
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")

    print()

    # Phase 2 only (no pipeline, only embedding)
    print("-" * 40)
    print("Phase 2 Only (phase='embed')")
    print("-" * 40)
    try:
        files_indexed = await indexer.index_project(phase="embed")
        print("✓ Phase 2 complete: pending chunks embedded")
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")


if __name__ == "__main__":
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║         Pipeline Parallelism Demo for index_project()        ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    try:
        asyncio.run(demo_pipeline_vs_sequential())
        asyncio.run(demo_phase_specific())

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        print("The pipeline parameter enables overlapping Phase 1 (parsing) and")
        print("Phase 2 (embedding) for improved throughput:")
        print()
        print("  • pipeline=True (default):  Phase 1 and Phase 2 overlap")
        print("  • pipeline=False:           Sequential execution (fallback)")
        print("  • phase='chunk' or 'embed': No pipeline (phase-specific)")
        print()
        print("Expected speedup: 30-50% on full reindex with many files.")
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback

        traceback.print_exc()

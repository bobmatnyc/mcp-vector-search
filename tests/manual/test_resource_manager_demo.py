"""Manual test to demonstrate memory-aware worker spawning.

Run this to see the resource manager in action:
    python tests/manual/test_resource_manager_demo.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_vector_search.core.resource_manager import (
    calculate_optimal_workers,
    get_batch_size_for_memory,
    get_configured_workers,
    get_system_memory,
)


def main():
    """Demonstrate resource manager functionality."""
    print("=" * 60)
    print("Memory-Aware Worker Spawning Demo")
    print("=" * 60)
    print()

    # Show system memory
    total_mb, available_mb = get_system_memory()
    print("System Memory:")
    print(f"  Total:     {total_mb:,} MB ({total_mb / 1024:.1f} GB)")
    print(f"  Available: {available_mb:,} MB ({available_mb / 1024:.1f} GB)")
    print()

    # Calculate optimal workers (default settings)
    print("Optimal Workers (Default Settings):")
    print("  Memory per worker: 500 MB")
    print("  Max workers: 8")
    limits = calculate_optimal_workers()
    print(f"  → Calculated workers: {limits.max_workers}")
    print()

    # Calculate for embedding workers (higher memory)
    print("Optimal Workers (Embedding Tasks):")
    print("  Memory per worker: 800 MB")
    print("  Max workers: 4")
    limits_embedding = calculate_optimal_workers(
        memory_per_worker_mb=800, max_workers=4
    )
    print(f"  → Calculated workers: {limits_embedding.max_workers}")
    print()

    # Calculate for batch processing
    print("Optimal Workers (Batch Processing):")
    print("  Memory per worker: 300 MB")
    print("  Max workers: 16")
    limits_batch = calculate_optimal_workers(memory_per_worker_mb=300, max_workers=16)
    print(f"  → Calculated workers: {limits_batch.max_workers}")
    print()

    # Get configured workers (respects environment)
    print("Configured Workers:")
    workers = get_configured_workers()
    env_override = os.environ.get("MCP_VECTOR_SEARCH_WORKERS")
    if env_override:
        print(f"  From environment: {env_override}")
    else:
        print(f"  Auto-configured: {workers}")
    print()

    # Batch size calculation
    print("Batch Size Calculations:")
    batch_default = get_batch_size_for_memory()
    print(f"  Default (10KB items, 100MB target): {batch_default}")

    batch_small = get_batch_size_for_memory(item_size_kb=5, target_batch_mb=50)
    print(f"  Small items (5KB items, 50MB target): {batch_small}")

    batch_large = get_batch_size_for_memory(item_size_kb=100, target_batch_mb=200)
    print(f"  Large items (100KB items, 200MB target): {batch_large}")
    print()

    # Environment variable overrides
    print("Environment Variable Overrides:")
    print("  MCP_VECTOR_SEARCH_WORKERS: Force specific worker count")
    print("  MCP_VECTOR_SEARCH_MEMORY_PER_WORKER: Adjust memory per worker")
    print()
    print("Example usage:")
    print("  export MCP_VECTOR_SEARCH_WORKERS=4")
    print("  export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=800")
    print()

    # Recommendations based on memory
    print("Recommendations:")
    if available_mb < 4000:
        print("  ⚠️  Low memory system (<4GB available)")
        print("     Consider: MCP_VECTOR_SEARCH_WORKERS=1")
        print("     Consider: MCP_VECTOR_SEARCH_BATCH_SIZE=16")
    elif available_mb < 8000:
        print("  ℹ️  Medium memory system (4-8GB available)")
        print("     Auto-configuration should work well")
        print("     Workers: 2-4 recommended")
    else:
        print("  ✓ High memory system (>8GB available)")
        print("     Auto-configuration will maximize throughput")
        print(f"     Workers: {limits.max_workers} recommended")
    print()

    print("=" * 60)


if __name__ == "__main__":
    main()

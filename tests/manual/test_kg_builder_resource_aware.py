"""Manual test to demonstrate resource-aware KG builder.

This shows how KGBuilder now auto-configures workers and batch sizes
based on available system memory.

Run this to see the integration:
    python tests/manual/test_kg_builder_resource_aware.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_vector_search.core.resource_manager import get_system_memory


def main():
    """Demonstrate KGBuilder resource awareness."""
    print("=" * 60)
    print("KGBuilder Resource-Aware Configuration")
    print("=" * 60)
    print()

    # Show system memory
    total_mb, available_mb = get_system_memory()
    print("System Memory:")
    print(f"  Total:     {total_mb:,} MB ({total_mb / 1024:.1f} GB)")
    print(f"  Available: {available_mb:,} MB ({available_mb / 1024:.1f} GB)")
    print()

    # Import after path setup
    from mcp_vector_search.core.kg_builder import KGBuilder
    from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

    # Create temporary KG (in-memory)
    temp_dir = Path("/tmp/test_kg")
    temp_dir.mkdir(exist_ok=True)

    print("Creating KnowledgeGraph and KGBuilder...")
    kg = KnowledgeGraph(temp_dir / "test.db")
    builder = KGBuilder(kg, temp_dir)

    print()
    print("KGBuilder Configuration:")
    print(f"  Workers:    {builder._workers}")
    print(f"  Batch size: {builder._batch_size}")
    print()

    print("Configuration Details:")
    print("  - Workers are auto-configured based on available memory")
    print("  - Batch size optimized for ~5KB entities in 50MB batches")
    print("  - Configuration can be overridden via environment variables:")
    print("      export MCP_VECTOR_SEARCH_WORKERS=4")
    print("      export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=500")
    print()

    # Cleanup
    kg.close()
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)

    print("✓ KGBuilder successfully configured with memory-aware settings")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

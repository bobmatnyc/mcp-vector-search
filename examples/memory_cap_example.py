"""Example demonstrating memory cap configuration for large codebase indexing.

This example shows how to configure and use the memory monitor to prevent
out-of-memory issues when indexing large codebases.
"""

import asyncio
import os
from pathlib import Path

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.factory import create_database
from mcp_vector_search.core.indexer import SemanticIndexer


async def index_with_memory_cap():
    """Index a project with memory cap enabled."""

    # Configure memory cap via environment variable
    # Set to 25GB (default) or adjust based on available system memory
    os.environ["MCP_VECTOR_SEARCH_MAX_MEMORY_GB"] = "25"

    # Optionally configure other memory-related settings
    os.environ["MCP_VECTOR_SEARCH_WORKERS"] = "4"  # Limit parallel workers
    os.environ["MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"] = "256"  # Files per batch (Phase 1)

    # Setup project paths
    project_root = Path.cwd()
    index_path = project_root / ".mcp-vector-search"

    # Create configuration
    config = ProjectConfig(
        project_root=project_root,
        index_path=index_path,
        embedding_model="jinaai/jina-embeddings-v3",  # Default embedding model
    )

    # Create database with embedding function
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=index_path / "code_search.lance",
        embedding_function=embedding_function,
    )

    # Create indexer with memory monitoring enabled (default)
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        use_multiprocessing=True,  # Enable parallel processing
        auto_optimize=True,  # Enable automatic optimization
    )

    # Access the memory monitor
    print(f"Memory cap: {indexer.memory_monitor.max_memory_gb:.1f}GB")
    print(
        f"Warning threshold: {indexer.memory_monitor.warn_threshold * 100:.0f}% ({indexer.memory_monitor.max_memory_gb * indexer.memory_monitor.warn_threshold:.1f}GB)"
    )
    print(
        f"Critical threshold: {indexer.memory_monitor.critical_threshold * 100:.0f}% ({indexer.memory_monitor.max_memory_gb * indexer.memory_monitor.critical_threshold:.1f}GB)"
    )

    # Index the project
    # The memory monitor will automatically:
    # 1. Check memory every 100 files during Phase 1 (chunking)
    # 2. Check memory before each batch during Phase 2 (embedding)
    # 3. Adjust batch sizes when memory pressure is detected
    # 4. Apply backpressure if memory limit is exceeded
    print("\nStarting indexing with memory monitoring...")
    indexed_count = await indexer.index_project(
        force_reindex=False,  # Incremental update
        show_progress=True,  # Show progress bars
        skip_relationships=False,  # Compute relationships
        phase="all",  # Both Phase 1 (chunk) and Phase 2 (embed)
    )

    print(f"\n✓ Indexed {indexed_count} files successfully")

    # Check final memory usage
    indexer.memory_monitor.log_memory_summary()


async def index_with_custom_memory_settings():
    """Example with custom memory settings for different scenarios."""

    # Scenario 1: Large codebase (100k+ files) on high-memory machine
    # Increase memory cap and use more workers
    os.environ["MCP_VECTOR_SEARCH_MAX_MEMORY_GB"] = "40"
    os.environ["MCP_VECTOR_SEARCH_WORKERS"] = "8"
    os.environ["MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"] = "512"

    # Scenario 2: Medium codebase on low-memory machine
    # Reduce memory cap and limit parallelism
    os.environ["MCP_VECTOR_SEARCH_MAX_MEMORY_GB"] = "10"
    os.environ["MCP_VECTOR_SEARCH_WORKERS"] = "2"
    os.environ["MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"] = "128"

    # Scenario 3: Memory-constrained environment (CI/CD, containers)
    # Strict memory limit with conservative settings
    os.environ["MCP_VECTOR_SEARCH_MAX_MEMORY_GB"] = "8"
    os.environ["MCP_VECTOR_SEARCH_WORKERS"] = "1"
    os.environ["MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"] = "64"

    # ... rest of indexing code


async def monitor_memory_during_indexing():
    """Example showing how to monitor memory usage programmatically."""

    # Setup (same as index_with_memory_cap)
    project_root = Path.cwd()
    config = ProjectConfig(project_root=project_root)
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=project_root / ".mcp-vector-search" / "code_search.lance",
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database, project_root=project_root, config=config
    )

    # Check memory before indexing
    print("Before indexing:")
    indexer.memory_monitor.log_memory_summary()

    # Check if we should reduce batch size
    if indexer.memory_monitor.should_reduce_batch_size():
        print("⚠️  High memory pressure detected before starting")

    # Start indexing
    await indexer.index_project()

    # Check memory after indexing
    print("\nAfter indexing:")
    indexer.memory_monitor.log_memory_summary()

    # Get detailed stats
    current_gb = indexer.memory_monitor.get_current_memory_gb()
    usage_pct = indexer.memory_monitor.get_memory_usage_pct()
    print(f"\nFinal memory: {current_gb:.2f}GB ({usage_pct * 100:.1f}%)")


if __name__ == "__main__":
    # Run the example
    asyncio.run(index_with_memory_cap())

    # Or try custom settings
    # asyncio.run(index_with_custom_memory_settings())

    # Or monitor memory programmatically
    # asyncio.run(monitor_memory_during_indexing())

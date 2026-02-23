#!/usr/bin/env python3
"""Comprehensive SIGILL diagnosis matching exact production sequence."""

import asyncio
import sys
from pathlib import Path


async def test_actual_indexing_flow():
    """Reproduce the EXACT flow from reindex command."""

    print("=" * 80)
    print("DIAGNOSING SIGILL CRASH - PRODUCTION FLOW")
    print("=" * 80)

    # Step 1: Initialize project manager
    print("\n[1/7] Loading project configuration...")
    from mcp_vector_search.core.project import ProjectManager

    project_root = Path.cwd()
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        print("ERROR: Project not initialized")
        return

    config = project_manager.load_config()
    print(f"✓ Loaded config: {config.embedding_model}")

    # Step 2: Create embedding function (loads PyTorch)
    print("\n[2/7] Loading embedding model (PyTorch)...")
    from mcp_vector_search.config.defaults import get_default_cache_path
    from mcp_vector_search.core.embeddings import create_embedding_function

    cache_dir = (
        get_default_cache_path(project_root) if config.cache_embeddings else None
    )
    embedding_function, cache = create_embedding_function(
        model_name=config.embedding_model,
        cache_dir=cache_dir,
        cache_size=config.max_cache_size,
    )
    print("✓ Embedding model ready")

    # Step 3: Create database
    print("\n[3/7] Creating database...")
    from mcp_vector_search.core.factory import create_database

    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )
    print("✓ Database ready")

    # Step 4: Create indexer (THIS creates ChunkProcessor with persistent pool)
    print(
        "\n[4/7] Creating SemanticIndexer (ChunkProcessor with ProcessPoolExecutor)..."
    )
    from mcp_vector_search.cli.output import console
    from mcp_vector_search.core.indexer import SemanticIndexer
    from mcp_vector_search.core.progress import ProgressTracker

    progress_tracker = ProgressTracker(console, verbose=True)

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        batch_size=512,
        progress_tracker=progress_tracker,
    )
    print("✓ SemanticIndexer created")
    print(f"   - ChunkProcessor: {indexer.chunk_processor}")
    print(f"   - Max workers: {indexer.chunk_processor.max_workers}")
    print(f"   - Use multiprocessing: {indexer.chunk_processor.use_multiprocessing}")
    print(f"   - Persistent pool: {indexer.chunk_processor._persistent_pool}")

    # Step 5: Check multiprocessing context
    print("\n[5/7] Checking multiprocessing configuration...")
    import multiprocessing as mp

    print(
        f"   - Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"   - Platform: {sys.platform}")
    print(f"   - Default start method: {mp.get_start_method()}")

    # Step 6: Run indexing (THIS is where SIGILL occurs)
    print("\n[6/7] Starting indexing...")
    print("   WARNING: If SIGILL occurs, it will happen during chunk processing")

    try:
        async with database:
            # This calls _index_with_pipeline which uses chunk_processor
            result = await indexer.chunk_and_embed(fresh=False, batch_size=512)

        print("✓ Indexing completed!")
        print(f"   - Files: {result.get('files_processed', 0)}")
        print(f"   - Chunks: {result.get('chunks_created', 0)}")
        print(f"   - Embeddings: {result.get('chunks_embedded', 0)}")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        raise
    except Exception as e:
        print(f"\n✗ Indexing failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 7: Clean up
    print("\n[7/7] Cleaning up...")
    indexer.chunk_processor.close()
    print("✓ Cleanup complete")

    print("\n" + "=" * 80)
    print("SUCCESS! No SIGILL crash")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(test_actual_indexing_flow())
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

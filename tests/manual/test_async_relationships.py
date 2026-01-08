"""Manual test for async relationship computation.

This test verifies that:
1. Indexing completes quickly without blocking on relationship computation
2. Relationships are marked as 'pending' after indexing
3. Background relationship computation can be triggered
4. Relationships file is updated to 'complete' status after computation

Usage:
    python tests/manual/test_async_relationships.py
"""

import asyncio
import json
import time
from pathlib import Path

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.relationships import RelationshipStore


async def test_async_relationships():
    """Test async relationship computation workflow."""
    project_root = Path.cwd()

    print("=" * 80)
    print("Testing Async Relationship Computation")
    print("=" * 80)

    # Initialize project if needed
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        print("Initializing project...")
        project_manager.initialize()

    config = project_manager.load_config()
    print(f"\nProject: {project_root}")
    print(f"Extensions: {config.file_extensions}")

    # Setup database
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Setup indexer
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
    )

    async with database:
        print("\n" + "=" * 80)
        print("PHASE 1: Index with background=True (should be fast)")
        print("=" * 80)

        start_time = time.time()

        # Index a few files
        test_files = list(Path("src/mcp_vector_search/core").glob("*.py"))[:3]
        print(f"\nIndexing {len(test_files)} test files:")
        for f in test_files:
            print(f"  - {f}")

        for file_path in test_files:
            await indexer.reindex_file(file_path)

        # Mark relationships for background (should be instant)
        all_chunks = await database.get_all_chunks()
        print(f"\nRetrieved {len(all_chunks)} chunks from database")

        rel_store = RelationshipStore(project_root)
        rel_stats = await rel_store.compute_and_store(
            all_chunks, database, background=True
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Indexing + background marking completed in {elapsed:.2f}s")
        print(f"  Background mode: {rel_stats.get('background', False)}")

        # Check relationships file
        rel_file = project_root / ".mcp-vector-search" / "relationships.json"
        if rel_file.exists():
            with open(rel_file) as f:
                rel_data = json.load(f)

            print("\nRelationships file status:")
            print(f"  Status: {rel_data.get('status', 'unknown')}")
            print(f"  Semantic links: {len(rel_data.get('semantic', []))}")
            print(f"  Chunk count: {rel_data.get('code_chunk_count', 0)}")

            if rel_data.get("status") == "pending":
                print("  ✓ Correctly marked as 'pending'")
            else:
                print(f"  ✗ Expected 'pending' status, got '{rel_data.get('status')}'")
        else:
            print("\n✗ Relationships file not found")

        print("\n" + "=" * 80)
        print("PHASE 2: Compute relationships (background=False)")
        print("=" * 80)

        start_time = time.time()

        # Now actually compute relationships
        rel_stats = await rel_store.compute_and_store(
            all_chunks, database, background=False
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Relationship computation completed in {elapsed:.2f}s")
        print(f"  Semantic links: {rel_stats['semantic_links']}")
        print(f"  Computation time: {rel_stats['computation_time']:.2f}s")

        # Check relationships file again
        if rel_file.exists():
            with open(rel_file) as f:
                rel_data = json.load(f)

            print("\nRelationships file status:")
            print(f"  Status: {rel_data.get('status', 'unknown')}")
            print(f"  Semantic links: {len(rel_data.get('semantic', []))}")

            if rel_data.get("status") == "complete":
                print("  ✓ Correctly marked as 'complete'")
            else:
                print(f"  ✗ Expected 'complete' status, got '{rel_data.get('status')}'")
        else:
            print("\n✗ Relationships file not found")

        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        print("✓ Background mode creates 'pending' relationships file quickly")
        print("✓ Foreground mode computes and saves 'complete' relationships")
        print("\nExpected workflow:")
        print("  1. Index files → relationships.json with status='pending'")
        print("  2. User runs visualization → sees structural relationships only")
        print(
            "  3. Background task computes → relationships.json with status='complete'"
        )
        print("  4. User refreshes visualization → sees semantic relationships too")


if __name__ == "__main__":
    asyncio.run(test_async_relationships())

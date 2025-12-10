#!/usr/bin/env python3
"""Test full indexing flow on EWTN project."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_full_index():
    """Test the complete indexing flow."""
    from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS
    from mcp_vector_search.core.database import ChromaVectorDatabase
    from mcp_vector_search.core.indexer import SemanticIndexer

    project_root = Path("/Users/masa/Clients/EWTN/projects")

    print(f"Testing full index flow for: {project_root}")
    print()

    # Step 1: Initialize database
    print("[1/5] Initializing database...")
    start = time.time()
    from mcp_vector_search.core.embeddings import create_embedding_function

    db_path = project_root / ".mcp-vector-search" / "chroma_db"
    embedding_function = create_embedding_function(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    database = ChromaVectorDatabase(
        persist_directory=db_path, embedding_function=embedding_function
    )
    await database.initialize()
    elapsed = time.time() - start
    print(f"  ✓ Database initialized in {elapsed:.2f}s")
    print()

    # Step 2: Create indexer
    print("[2/5] Creating indexer...")
    start = time.time()
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        file_extensions=DEFAULT_FILE_EXTENSIONS,
    )
    elapsed = time.time() - start
    print(f"  ✓ Indexer created in {elapsed:.2f}s")
    print(
        f"  Gitignore patterns: {len(indexer.gitignore_parser.patterns) if indexer.gitignore_parser else 0}"
    )
    print()

    # Step 3: Get files to index
    print("[3/5] Finding indexable files...")
    start = time.time()
    all_files, files_to_index = await indexer.get_files_to_index(force_reindex=False)
    elapsed = time.time() - start
    print(f"  ✓ Found files in {elapsed:.2f}s")
    print(f"  Total indexable: {len(all_files):,}")
    print(f"  Need indexing: {len(files_to_index):,}")
    print()

    # Step 4: Get stats (this is what status command does)
    print("[4/5] Getting indexing stats...")
    start = time.time()
    stats = await indexer.get_indexing_stats()
    elapsed = time.time() - start
    print(f"  ✓ Got stats in {elapsed:.2f}s")
    print(f"  Stats: {stats}")
    print()

    # Step 5: Get database stats
    print("[5/5] Getting database stats...")
    start = time.time()
    db_stats = await database.get_stats()
    elapsed = time.time() - start
    print(f"  ✓ Got DB stats in {elapsed:.2f}s")
    print(f"  Total files: {db_stats.total_files}")
    print(f"  Total chunks: {db_stats.total_chunks}")
    print()

    await database.close()

    print("✓ All steps completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_full_index())

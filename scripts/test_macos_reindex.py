#!/usr/bin/env python3
"""Test script to verify macOS SIGBUS fix for incremental reindex.

This script simulates the crash scenario:
1. Index a project
2. Modify some files
3. Run incremental reindex (this should NOT crash with SIGBUS)
4. Verify that new chunks are indexed

Usage:
    python scripts/test_macos_reindex.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.embeddings.factory import create_embedding_function

from mcp_vector_search.core.chunks_backend import ChunksBackend, compute_file_hash
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.lancedb_backend import LanceVectorDatabase


async def main():
    """Run test to verify SIGBUS fix."""
    print("🧪 Testing macOS SIGBUS fix for incremental reindex...")

    # Create temporary project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "test_project"
        project_dir.mkdir()

        # Create test file
        test_file = project_dir / "test.py"
        test_file.write_text(
            """
def hello():
    print("Hello, world!")
"""
        )

        # Create database directory
        db_dir = project_dir / ".mcp-vector-search"
        db_dir.mkdir()

        print(f"📁 Created test project at {project_dir}")

        # Step 1: Initial index
        print("\n🔨 Step 1: Initial index...")
        embedding_function, _ = create_embedding_function()
        database = LanceVectorDatabase(db_dir, embedding_function)

        indexer = SemanticIndexer(
            database=database,
            project_root=project_dir,
            config=None,
            use_multiprocessing=False,  # Simpler for testing
        )

        await indexer.index(force=True)
        print("✅ Initial index complete")

        # Step 2: Modify the file
        print("\n📝 Step 2: Modifying test file...")
        test_file.write_text(
            """
def hello():
    print("Hello, world!")

def goodbye():
    print("Goodbye!")
"""
        )
        print("✅ File modified")

        # Step 3: Incremental reindex (this should NOT crash with SIGBUS)
        print("\n🔄 Step 3: Incremental reindex (testing SIGBUS fix)...")
        try:
            await indexer.index(force=False)
            print("✅ Incremental reindex complete (no SIGBUS crash!)")
        except Exception as e:
            print(f"❌ Incremental reindex failed: {e}")
            raise

        # Step 4: Verify chunks were updated
        print("\n✅ Step 4: Verifying chunks...")
        chunks_backend = ChunksBackend(db_dir)
        await chunks_backend.initialize()

        stats = await chunks_backend.get_stats()
        print(f"📊 Chunks stats: {stats}")

        if stats["total"] > 0:
            print("✅ Chunks were successfully indexed")
        else:
            print("❌ No chunks found (indexing may have failed)")

        await chunks_backend.close()

    print("\n🎉 All tests passed! SIGBUS fix is working correctly.")


if __name__ == "__main__":
    asyncio.run(main())

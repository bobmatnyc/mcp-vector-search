"""Manual test to demonstrate glob pattern filtering for --files option.

This test shows the before/after behavior of the glob pattern fix.

Run this test manually after indexing the project:
    mcp-vector-search index
    python tests/manual/test_glob_pattern_filtering.py
"""

import asyncio
from pathlib import Path

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


async def test_glob_filtering():
    """Test glob pattern filtering with different patterns."""
    project_root = Path.cwd()
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        print("❌ Project not initialized. Run 'mcp-vector-search init' first.")
        return

    config = project_manager.load_config()

    # Setup database and search engine
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=0.3,
    )

    async with database:
        # Test 1: Search all files
        print("\n🔍 Test 1: Search without file filter")
        print("=" * 60)
        results = await search_engine.search(
            query="function", limit=5, include_context=False
        )
        print(f"Found {len(results)} results")
        for r in results[:3]:
            print(f"  - {r.file_path.name}")

        # Test 2: Filter by *.py pattern
        print("\n🔍 Test 2: Search with --files '*.py' (glob pattern)")
        print("=" * 60)
        results = await search_engine.search(
            query="function", limit=20, include_context=False
        )

        # Apply post-filtering (simulating the fix)
        import os
        from fnmatch import fnmatch

        filtered_results = []
        pattern = "*.py"
        for result in results:
            try:
                rel_path = str(result.file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(result.file_path)

            if fnmatch(rel_path, pattern) or fnmatch(
                os.path.basename(rel_path), pattern
            ):
                filtered_results.append(result)

        print(f"Found {len(filtered_results)} results matching '*.py'")
        for r in filtered_results[:3]:
            print(f"  - {r.file_path.name}")

        # Test 3: Filter by src/*.py pattern
        print("\n🔍 Test 3: Search with --files 'src/*.py' (directory pattern)")
        print("=" * 60)
        results = await search_engine.search(
            query="function", limit=20, include_context=False
        )

        filtered_results = []
        pattern = "src/*.py"
        for result in results:
            try:
                rel_path = str(result.file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(result.file_path)

            if fnmatch(rel_path, pattern):
                filtered_results.append(result)

        print(f"Found {len(filtered_results)} results matching 'src/*.py'")
        for r in filtered_results[:3]:
            print(f"  - {r.file_path}")

        # Test 4: Filter by specific filename
        print("\n🔍 Test 4: Search with --files 'search.py' (specific file)")
        print("=" * 60)
        results = await search_engine.search(
            query="function", limit=20, include_context=False
        )

        filtered_results = []
        pattern = "search.py"
        for result in results:
            if fnmatch(result.file_path.name, pattern):
                filtered_results.append(result)

        print(f"Found {len(filtered_results)} results matching 'search.py'")
        for r in filtered_results[:3]:
            print(f"  - {r.file_path}")

    print("\n✅ All tests completed successfully!")
    print("\nYou can now test the CLI with:")
    print("  mcp-vector-search search --files '*.py' 'function'")
    print("  mcp-vector-search search --files 'src/*.py' 'database'")
    print("  mcp-vector-search search --files 'search.py' 'query'")


if __name__ == "__main__":
    asyncio.run(test_glob_filtering())

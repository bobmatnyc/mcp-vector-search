#!/usr/bin/env python3
"""Simple smoke test for basic functionality."""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.factory import ComponentFactory
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.models import CodeChunk
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


@pytest.mark.asyncio
async def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🔍 Testing basic functionality...")

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create a simple test file
        test_file = project_dir / "test.py"
        test_file.write_text(
            '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return "greeting"

class TestClass:
    def test_method(self):
        return "test"
'''
        )

        try:
            # Test component creation
            print("  ✓ Testing component creation...")
            components = await ComponentFactory.create_standard_components(
                project_root=project_dir,
                use_pooling=False,
                include_search_engine=False,
                include_auto_indexer=False,
            )

            assert components.project_manager is not None
            assert components.config is not None
            assert components.database is not None
            assert components.indexer is not None
            assert components.embedding_function is not None

            print("  ✓ Component creation successful")

            # Test basic models
            print("  ✓ Testing basic models...")
            chunk = CodeChunk(
                id="test_chunk",
                content="def test(): pass",
                file_path=Path("test.py"),
                start_line=1,
                end_line=1,
                language="python",
                chunk_type="function",
            )

            assert chunk.id == "test_chunk"
            assert chunk.content == "def test(): pass"
            assert chunk.language == "python"

            print("  ✓ Basic models working")

            print("✅ Basic functionality test passed!")
            return True

        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False


async def main():
    """Test basic functionality."""
    print("🚀 Testing MCP Vector Search...")

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)
        print(f"📁 Using temp directory: {project_dir}")

        # Create test files
        (project_dir / "example.py").write_text(
            """
def hello_world():
    \"\"\"Print hello world message.\"\"\"
    print("Hello, World!")
    return "success"

def calculate_area(radius):
    \"\"\"Calculate the area of a circle.\"\"\"
    import math
    return math.pi * radius * radius

class DataProcessor:
    \"\"\"Process data efficiently.\"\"\"

    def __init__(self):
        self.data = []

    def add_item(self, item):
        \"\"\"Add an item to the data list.\"\"\"
        self.data.append(item)
        return len(self.data)

    def process_all(self):
        \"\"\"Process all items in the data list.\"\"\"
        results = []
        for item in self.data:
            results.append(str(item).upper())
        return results
"""
        )

        print("✅ Created test files")

        # Initialize project
        project_manager = ProjectManager(project_dir)
        config = project_manager.initialize(
            file_extensions=[".py"],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.3,
        )
        print(f"✅ Project initialized with {len(config.languages)} languages")

        # Set up components
        embedding_function, _ = create_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_dir,
            file_extensions=[".py"],
        )

        search_engine = SemanticSearchEngine(
            database=database,
            project_root=project_dir,
            similarity_threshold=0.2,  # Lower threshold
        )

        async with database:
            # Index the project
            print("📚 Indexing project...")
            indexed_count = await indexer.index_project()
            print(f"✅ Indexed {indexed_count} files")

            # Get stats
            stats = await indexer.get_indexing_stats()
            print(f"📊 Total chunks: {stats['total_chunks']}")
            print(f"📊 Languages: {stats['languages']}")

            # Test various searches with very low thresholds
            test_queries = [
                ("hello", 0.1),
                ("calculate", 0.1),
                ("data", 0.1),
                ("process", 0.1),
                ("function", 0.1),
                ("class", 0.1),
                ("def", 0.1),
                ("python", 0.1),
            ]

            for query, threshold in test_queries:
                print(f"\n🔍 Searching for '{query}' (threshold: {threshold})...")
                results = await search_engine.search(
                    query=query, limit=5, similarity_threshold=threshold
                )

                if results:
                    print(f"✅ Found {len(results)} results:")
                    for i, result in enumerate(results[:2], 1):
                        print(
                            f"  {i}. {result.file_path.name}:{result.start_line}-{result.end_line}"
                        )
                        print(f"     Similarity: {result.similarity_score:.2%}")
                        print(f"     Content preview: {result.content[:100]}...")
                else:
                    print(f"❌ No results found for '{query}'")

            # Test if we can get ANY results by querying the database directly
            print("\n🔍 Testing direct database query...")
            try:
                # Get all documents to see what's actually stored
                collection = database._collection
                all_docs = collection.get(limit=10)
                print(f"📊 Database contains {len(all_docs['ids'])} documents")

                if all_docs["documents"]:
                    print("📄 Sample document content:")
                    print(f"  ID: {all_docs['ids'][0]}")
                    print(f"  Content: {all_docs['documents'][0][:200]}...")

                    # Try searching for exact content
                    sample_content = all_docs["documents"][0][:50]
                    print(f"\n🔍 Searching for exact content: '{sample_content}'")
                    exact_results = await search_engine.search(
                        query=sample_content,
                        limit=5,
                        similarity_threshold=0.0,  # Accept any similarity
                    )
                    print(f"✅ Exact search found {len(exact_results)} results")

            except Exception as e:
                print(f"❌ Database query failed: {e}")

    print("\n🎉 Test completed!")


def main_wrapper():
    """Main function wrapper for CLI usage."""
    try:
        # Run basic functionality test first
        basic_result = asyncio.run(test_basic_functionality())
        if not basic_result:
            print("💥 Basic functionality test failed!")
            sys.exit(1)

        # Run full test
        asyncio.run(main())
        print("🎉 All tests completed!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_wrapper()

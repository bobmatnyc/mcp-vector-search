"""Basic functionality tests for MCP Vector Search."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.project import ProjectManager
from mcp_vector_search.core.search import SemanticSearchEngine


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create some sample Python files
        (project_dir / "main.py").write_text(
            """
def main():
    \"\"\"Main function for the application.\"\"\"
    print("Hello, world!")
    return 0

if __name__ == "__main__":
    main()
"""
        )

        (project_dir / "utils.py").write_text(
            """
def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b

def parse_config(config_file):
    \"\"\"Parse configuration from file.\"\"\"
    with open(config_file, 'r') as f:
        return f.read()
"""
        )

        yield project_dir


def test_project_initialization(temp_project_dir):
    """Test project initialization."""
    project_manager = ProjectManager(temp_project_dir)

    # Should not be initialized initially
    assert not project_manager.is_initialized()

    # Initialize project
    config = project_manager.initialize(
        file_extensions=[".py"],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.7,
    )

    # Should be initialized now
    assert project_manager.is_initialized()
    assert config.project_root.resolve() == temp_project_dir.resolve()
    assert ".py" in config.file_extensions

    # Should be able to load config
    loaded_config = project_manager.load_config()
    assert loaded_config.project_root.resolve() == temp_project_dir.resolve()


@pytest.mark.asyncio
async def test_indexing_and_search(temp_project_dir):
    """Test basic indexing and search functionality."""
    # Initialize project
    project_manager = ProjectManager(temp_project_dir)
    config = project_manager.initialize(
        file_extensions=[".py"],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.5,
        force=True,
    )

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
        project_root=temp_project_dir,
        file_extensions=[".py"],
    )

    search_engine = SemanticSearchEngine(
        database=database,
        project_root=temp_project_dir,
        similarity_threshold=0.3,
    )

    async with database:
        # Index the project
        indexed_count = await indexer.index_project()
        assert indexed_count > 0

        # Get stats
        stats = await indexer.get_indexing_stats()
        assert stats["total_chunks"] > 0
        assert "python" in stats["languages"]

        # Test search with very low threshold to ensure we get results
        results = await search_engine.search(
            "function", limit=10, similarity_threshold=0.1
        )
        assert len(results) > 0, (
            f"No results found. Total chunks: {stats['total_chunks']}"
        )

        # Test search for any content
        all_results = await search_engine.search(
            "def", limit=10, similarity_threshold=0.1
        )
        assert len(all_results) > 0, "Should find some Python function definitions"


def test_project_info(temp_project_dir):
    """Test project info functionality."""
    project_manager = ProjectManager(temp_project_dir)

    # Get info before initialization
    info = project_manager.get_project_info()
    assert not info.is_initialized
    assert info.name == temp_project_dir.name

    # Initialize and get info again
    project_manager.initialize(file_extensions=[".py"])
    info = project_manager.get_project_info()
    assert info.is_initialized
    assert info.file_count > 0
    assert "python" in info.languages


if __name__ == "__main__":
    # Run a simple test
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create test file
        (project_dir / "test.py").write_text(
            """
def hello_world():
    \"\"\"A simple hello world function.\"\"\"
    return 'Hello, World!'

def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.\"\"\"
    return a + b

class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def add(self, x, y):
        \"\"\"Add two numbers.\"\"\"
        return x + y

    def multiply(self, x, y):
        \"\"\"Multiply two numbers.\"\"\"
        return x * y
"""
        )

        # Test initialization
        project_manager = ProjectManager(project_dir)
        config = project_manager.initialize(file_extensions=[".py"])

        print(f"✓ Project initialized at {project_dir}")
        print(f"✓ Config: {config.embedding_model}")
        print(f"✓ Languages detected: {config.languages}")

        # Test basic functionality
        asyncio.run(test_indexing_and_search(project_dir))
        print("✓ All tests passed!")

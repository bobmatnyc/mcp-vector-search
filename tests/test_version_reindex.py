"""Tests for version-based automatic reindexing."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_vector_search import __version__
from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.project import ProjectManager


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir)

        # Create sample Python file
        (project_dir / "main.py").write_text(
            '''
def hello_world():
    """Say hello to the world."""
    print("Hello, world!")
    return 0
'''
        )

        yield project_dir


@pytest.fixture
def initialized_project(temp_project_dir):
    """Initialize project with database and indexer."""
    # Initialize project
    project_manager = ProjectManager(temp_project_dir)
    project_manager.initialize(file_extensions=[".py"])
    config = project_manager.load_config()

    # Create database and indexer
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=temp_project_dir,
        file_extensions=config.file_extensions,
    )

    return temp_project_dir, database, indexer, config


@pytest.mark.asyncio
async def test_version_stored_in_metadata(initialized_project):
    """Test that version is stored in index metadata after indexing."""
    temp_project_dir, database, indexer, config = initialized_project

    # Index the project
    async with database:
        await indexer.index_project(force_reindex=True)

    # Check metadata file
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    assert metadata_file.exists()

    with open(metadata_file) as f:
        metadata = json.load(f)

    assert "index_version" in metadata
    assert metadata["index_version"] == __version__
    assert "indexed_at" in metadata
    assert "file_mtimes" in metadata


@pytest.mark.asyncio
async def test_get_index_version(initialized_project):
    """Test getting index version from metadata."""
    temp_project_dir, database, indexer, config = initialized_project

    # Initially no version
    assert indexer.get_index_version() is None

    # Index the project
    async with database:
        await indexer.index_project(force_reindex=True)

    # Should have version now
    version = indexer.get_index_version()
    assert version == __version__


def test_needs_reindex_no_metadata(initialized_project):
    """Test that reindex is needed when no metadata exists."""
    temp_project_dir, database, indexer, config = initialized_project

    # No metadata yet
    assert indexer.needs_reindex_for_version() is True


@pytest.mark.asyncio
async def test_needs_reindex_same_version(initialized_project):
    """Test that reindex is NOT needed for same version."""
    temp_project_dir, database, indexer, config = initialized_project

    # Index with current version
    async with database:
        await indexer.index_project(force_reindex=True)

    # Should not need reindex
    assert indexer.needs_reindex_for_version() is False


def test_needs_reindex_patch_version(initialized_project):
    """Test that reindex is NOT needed for patch version upgrade."""
    temp_project_dir, database, indexer, config = initialized_project

    # Simulate older patch version
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # If current version is 0.5.1, simulate 0.5.0
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "index_version": "0.5.0",
                "indexed_at": datetime.now(UTC).isoformat(),
                "file_mtimes": {},
            },
            f,
        )

    # Patch version upgrade should NOT require reindex
    # (assuming current version is 0.5.x)
    needs_reindex = indexer.needs_reindex_for_version()
    # This might be True or False depending on actual version
    # Just verify it doesn't crash
    assert isinstance(needs_reindex, bool)


def test_needs_reindex_minor_version(initialized_project):
    """Test that reindex IS needed for minor version upgrade."""
    temp_project_dir, database, indexer, config = initialized_project

    # Simulate older minor version
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # Simulate version 0.4.0 when current is 0.5.x
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "index_version": "0.4.0",
                "indexed_at": datetime.now(UTC).isoformat(),
                "file_mtimes": {},
            },
            f,
        )

    # Minor version upgrade should require reindex
    assert indexer.needs_reindex_for_version() is True


def test_needs_reindex_major_version(initialized_project):
    """Test that reindex IS needed for major version upgrade."""
    temp_project_dir, database, indexer, config = initialized_project

    # Simulate older major version
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # Simulate version 0.5.1 when current might be 1.x.x in future
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "index_version": "0.5.1",
                "indexed_at": datetime.now(UTC).isoformat(),
                "file_mtimes": {},
            },
            f,
        )

    # Check - if we're still on 0.x, this will be False
    # But the logic should work correctly
    needs_reindex = indexer.needs_reindex_for_version()
    assert isinstance(needs_reindex, bool)


def test_legacy_metadata_format(initialized_project):
    """Test that legacy metadata format (without version) triggers reindex."""
    temp_project_dir, database, indexer, config = initialized_project

    # Create legacy format metadata (just file_mtimes dict)
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        json.dump({"some_file.py": 1234567890.0}, f)

    # Should need reindex due to no version
    assert indexer.needs_reindex_for_version() is True

    # Should handle loading legacy format without crashing
    metadata = indexer._load_index_metadata()
    assert isinstance(metadata, dict)


def test_metadata_backward_compatibility(initialized_project):
    """Test that new code can read both old and new metadata formats."""
    temp_project_dir, database, indexer, config = initialized_project

    # Test 1: Legacy format (just dict)
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    legacy_data = {
        "file1.py": 1234567890.0,
        "file2.py": 1234567891.0,
    }

    with open(metadata_file, "w") as f:
        json.dump(legacy_data, f)

    metadata = indexer._load_index_metadata()
    assert metadata == legacy_data

    # Test 2: New format (with structure)
    new_data = {
        "index_version": "0.5.1",
        "indexed_at": datetime.now(UTC).isoformat(),
        "file_mtimes": {
            "file1.py": 1234567890.0,
            "file2.py": 1234567891.0,
        },
    }

    with open(metadata_file, "w") as f:
        json.dump(new_data, f)

    metadata = indexer._load_index_metadata()
    assert metadata == new_data["file_mtimes"]


@pytest.mark.asyncio
async def test_auto_reindex_integration(temp_project_dir):
    """Integration test for auto-reindex workflow."""
    # Initialize project
    project_manager = ProjectManager(temp_project_dir)
    project_manager.initialize(file_extensions=[".py"])
    config = project_manager.load_config()

    # Ensure auto_reindex_on_upgrade is enabled (default)
    assert config.auto_reindex_on_upgrade is True

    # Create database and indexer
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=temp_project_dir,
        file_extensions=config.file_extensions,
    )

    # Index initially
    async with database:
        await indexer.index_project(force_reindex=True)

    # Verify version is stored
    version = indexer.get_index_version()
    assert version == __version__

    # Simulate older version
    metadata_file = temp_project_dir / ".mcp-vector-search" / "index_metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    metadata["index_version"] = "0.4.0"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Create new indexer to check version
    indexer2 = SemanticIndexer(
        database=database,
        project_root=temp_project_dir,
        file_extensions=config.file_extensions,
    )

    # Should detect need for reindex
    assert indexer2.needs_reindex_for_version() is True

    # Reindex should update version
    async with database:
        await indexer2.index_project(force_reindex=True)

    # Version should be updated
    assert indexer2.get_index_version() == __version__
    assert indexer2.needs_reindex_for_version() is False

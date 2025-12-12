"""Tests for relationship pre-computation and storage."""

import json
from pathlib import Path

import pytest

from mcp_vector_search.core.models import CodeChunk
from mcp_vector_search.core.relationships import (
    RelationshipStore,
    extract_chunk_name,
    extract_function_calls,
)


def test_extract_function_calls():
    """Test extracting function calls from Python code."""
    code = """
def foo():
    bar()  # Actual call
    baz.qux()  # Method call

# Comment about bar() - not a call
s = "bar()"  # String literal - not a call
"""

    calls = extract_function_calls(code)

    # Should find actual calls
    assert "bar" in calls
    assert "qux" in calls

    # Should not include keywords
    assert "def" not in calls


def test_extract_chunk_name():
    """Test extracting meaningful names from chunk content."""
    # Function definition
    assert extract_chunk_name("def calculate_total(items): ...") == "calculate_total"

    # Class definition
    assert extract_chunk_name("class UserManager: ...") == "UserManager"

    # Skip keywords (True/False are in skip list, so we get fallback)
    assert extract_chunk_name("if True: return False", fallback="no_name") == "no_name"

    # Fallback
    assert extract_chunk_name("# Just a comment") == "Just"


def test_relationship_store_init(tmp_path: Path):
    """Test relationship store initialization."""
    store = RelationshipStore(tmp_path)

    assert store.project_root == tmp_path
    assert store.store_path == tmp_path / ".mcp-vector-search" / "relationships.json"
    assert not store.exists()


def test_relationship_store_load_nonexistent(tmp_path: Path):
    """Test loading relationships when file doesn't exist."""
    store = RelationshipStore(tmp_path)

    data = store.load()

    assert data == {"semantic": [], "callers": {}}
    assert not store.exists()


def test_relationship_store_manual_save_and_load(tmp_path: Path):
    """Test manually saving and loading relationships."""
    store = RelationshipStore(tmp_path)

    # Manually create relationships file
    relationships = {
        "version": "1.0",
        "computed_at": "2025-01-01T00:00:00Z",
        "chunk_count": 10,
        "code_chunk_count": 5,
        "computation_time_seconds": 1.5,
        "semantic": [
            {
                "source": "chunk1",
                "target": "chunk2",
                "type": "semantic",
                "similarity": 0.85,
            }
        ],
        "callers": {
            "chunk1": [
                {
                    "file": "test.py",
                    "chunk_id": "chunk3",
                    "name": "test_func",
                    "type": "function",
                }
            ]
        },
    }

    # Save
    store.store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(store.store_path, "w") as f:
        json.dump(relationships, f)

    # Load
    loaded = store.load()

    assert loaded["version"] == "1.0"
    assert len(loaded["semantic"]) == 1
    assert loaded["semantic"][0]["source"] == "chunk1"
    assert "chunk1" in loaded["callers"]
    assert store.exists()


def test_relationship_store_invalidate(tmp_path: Path):
    """Test invalidating relationships."""
    store = RelationshipStore(tmp_path)

    # Create file
    store.store_path.parent.mkdir(parents=True, exist_ok=True)
    store.store_path.write_text("{}")

    assert store.exists()

    # Invalidate
    store.invalidate()

    assert not store.exists()


@pytest.mark.asyncio
async def test_compute_caller_relationships():
    """Test computing caller relationships from chunks."""
    from mcp_vector_search.core.relationships import RelationshipStore

    # Create mock chunks
    chunks = [
        CodeChunk(
            chunk_id="chunk1",
            file_path="module1.py",
            start_line=1,
            end_line=5,
            content='def my_function():\n    return "test"',
            chunk_type="function",
            function_name="my_function",
            language="python",
        ),
        CodeChunk(
            chunk_id="chunk2",
            file_path="module2.py",
            start_line=1,
            end_line=5,
            content="def caller():\n    my_function()  # Call the function",
            chunk_type="function",
            function_name="caller",
            language="python",
        ),
    ]

    store = RelationshipStore(Path("/tmp"))
    caller_map = store._compute_caller_relationships(chunks)

    # chunk1 (my_function) should be called by chunk2 (caller)
    # The caller map maps the CALLED function to its CALLERS
    assert "chunk1" in caller_map
    assert len(caller_map["chunk1"]) == 1
    assert caller_map["chunk1"][0]["chunk_id"] == "chunk2"
    assert caller_map["chunk1"][0]["name"] == "caller"

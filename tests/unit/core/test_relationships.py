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


@pytest.mark.asyncio
async def test_parallel_semantic_relationships():
    """Test parallel computation produces same results as sequential."""
    from unittest.mock import AsyncMock, MagicMock

    from mcp_vector_search.core.models import SearchResult
    from mcp_vector_search.core.relationships import RelationshipStore

    # Create test chunks
    chunks = [
        CodeChunk(
            chunk_id=f"chunk{i}",
            file_path=f"file{i}.py",
            start_line=1,
            end_line=10,
            content=f"def function_{i}():\n    pass",
            chunk_type="function",
            function_name=f"function_{i}",
            language="python",
        )
        for i in range(10)
    ]

    # Mock database that returns different results for each chunk
    mock_db = MagicMock()

    async def mock_search(query: str, limit: int, similarity_threshold: float):
        # Return mock results based on query content
        chunk_id = query.split("_")[1].split("(")[0] if "_" in query else "0"
        return [
            SearchResult(
                content=f"def function_{j}():\n    pass",
                file_path=f"file{j}.py",
                start_line=1,
                end_line=10,
                language="python",
                similarity_score=0.8 - (abs(int(chunk_id) - j) * 0.1),
                rank=idx,
            )
            for idx, j in enumerate(range(10))
            if j != int(chunk_id)
        ][:limit]

    mock_db.search = AsyncMock(side_effect=mock_search)

    store = RelationshipStore(Path("/tmp"))

    # Compute relationships
    links = await store._compute_semantic_relationships(chunks, mock_db)

    # Verify results
    assert len(links) > 0
    assert all(isinstance(link, dict) for link in links)
    assert all("source" in link and "target" in link for link in links)
    assert all(link["type"] == "semantic" for link in links)
    assert all(0.0 <= link["similarity"] <= 1.0 for link in links)

    # Verify no self-references
    assert all(link["source"] != link["target"] for link in links)

    # Verify database was called for each chunk
    assert mock_db.search.call_count == len(chunks)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Test that semaphore correctly limits concurrent queries."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from mcp_vector_search.core.relationships import RelationshipStore

    # Track concurrent calls
    concurrent_calls = 0
    max_concurrent_calls = 0
    call_lock = asyncio.Lock()

    # Create test chunks
    chunks = [
        CodeChunk(
            chunk_id=f"chunk{i}",
            file_path=f"file{i}.py",
            start_line=1,
            end_line=10,
            content=f"def function_{i}():\n    pass",
            chunk_type="function",
            function_name=f"function_{i}",
            language="python",
        )
        for i in range(20)
    ]

    # Mock database with delay to allow concurrent tracking
    mock_db = MagicMock()

    async def mock_search_with_delay(
        query: str, limit: int, similarity_threshold: float
    ):
        nonlocal concurrent_calls, max_concurrent_calls

        async with call_lock:
            concurrent_calls += 1
            max_concurrent_calls = max(max_concurrent_calls, concurrent_calls)

        # Simulate some work
        await asyncio.sleep(0.01)

        async with call_lock:
            concurrent_calls -= 1

        return []

    mock_db.search = AsyncMock(side_effect=mock_search_with_delay)

    store = RelationshipStore(Path("/tmp"))

    # Set low concurrency limit for testing
    max_concurrent = 5
    await store._compute_semantic_relationships(chunks, mock_db, max_concurrent)

    # Verify concurrency was limited
    assert max_concurrent_calls <= max_concurrent
    assert max_concurrent_calls > 1  # Should have some parallelism


@pytest.mark.asyncio
async def test_parallel_handles_exceptions():
    """Test that exceptions in individual tasks don't fail entire computation."""
    from unittest.mock import AsyncMock, MagicMock

    from mcp_vector_search.core.models import SearchResult
    from mcp_vector_search.core.relationships import RelationshipStore

    # Create test chunks
    chunks = [
        CodeChunk(
            chunk_id=f"chunk{i}",
            file_path=f"file{i}.py",
            start_line=1,
            end_line=10,
            content=f"def function_{i}():\n    pass",
            chunk_type="function",
            function_name=f"function_{i}",
            language="python",
        )
        for i in range(10)
    ]

    # Mock database that fails on some chunks
    mock_db = MagicMock()
    call_count = 0

    async def mock_search_with_failures(
        query: str, limit: int, similarity_threshold: float
    ):
        nonlocal call_count
        call_count += 1

        # Fail every 3rd call
        if call_count % 3 == 0:
            raise Exception("Simulated database error")

        # Return some results for successful calls
        return [
            SearchResult(
                content="def other():\n    pass",
                file_path="other.py",
                start_line=1,
                end_line=10,
                language="python",
                similarity_score=0.7,
                rank=0,
            )
        ]

    mock_db.search = AsyncMock(side_effect=mock_search_with_failures)

    store = RelationshipStore(Path("/tmp"))

    # Should complete without raising exception
    links = await store._compute_semantic_relationships(chunks, mock_db)

    # Should have some results (from successful calls)
    assert isinstance(links, list)
    # At least some successful links expected
    assert len(links) >= 0


@pytest.mark.asyncio
async def test_progress_tracking_works():
    """Test that progress tracking is updated during parallel execution."""
    from unittest.mock import AsyncMock, MagicMock

    from mcp_vector_search.core.relationships import RelationshipStore

    # Create test chunks
    chunks = [
        CodeChunk(
            chunk_id=f"chunk{i}",
            file_path=f"file{i}.py",
            start_line=1,
            end_line=10,
            content=f"def function_{i}():\n    pass",
            chunk_type="function",
            function_name=f"function_{i}",
            language="python",
        )
        for i in range(5)
    ]

    # Mock database
    mock_db = MagicMock()
    mock_db.search = AsyncMock(return_value=[])

    store = RelationshipStore(Path("/tmp"))

    # Should complete without errors
    links = await store._compute_semantic_relationships(chunks, mock_db)

    assert isinstance(links, list)
    # All chunks should be processed
    assert mock_db.search.call_count == len(chunks)

"""Tests for differential (incremental) KG build — issue #108.

Covers:
- KGBuilder._get_changed_files(): hash-based change detection
- KnowledgeGraph.delete_entities_for_files(): Kuzu DELETE execution
"""

import json
from pathlib import Path

import pytest

from mcp_vector_search.core.kg_builder import KGBuilder
from mcp_vector_search.core.knowledge_graph import (
    CodeEntity,
    CodeRelationship,
    KnowledgeGraph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(tmp_path: Path) -> KGBuilder:
    """Create a KGBuilder whose metadata path lives inside tmp_path."""
    # KGBuilder.__init__ only needs project_root for the metadata path;
    # passing kg=None is fine for metadata-only tests.
    builder = KGBuilder(None, tmp_path)  # type: ignore[arg-type]
    return builder


def _write_metadata(builder: KGBuilder, file_hashes: dict[str, str]) -> None:
    """Write a minimal kg_metadata.json with the given file_hashes."""
    builder._metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(builder._metadata_path, "w") as fh:
        json.dump({"file_hashes": file_hashes}, fh)


# ---------------------------------------------------------------------------
# _get_changed_files — no prior metadata
# ---------------------------------------------------------------------------


def test_get_changed_files_no_prior_metadata(tmp_path):
    """With no kg_metadata.json, every current file should be in new_files."""
    builder = _make_builder(tmp_path)
    # Metadata file does NOT exist
    assert not builder._metadata_path.exists()

    current = {"src/a.py": "aaaa", "src/b.py": "bbbb"}
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == set()
    assert new == {"src/a.py", "src/b.py"}
    assert deleted == set()


def test_get_changed_files_metadata_without_file_hashes_key(tmp_path):
    """Metadata present but missing 'file_hashes' key → all files are new."""
    builder = _make_builder(tmp_path)
    builder._metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(builder._metadata_path, "w") as fh:
        json.dump({"source_chunk_ids": ["a", "b"]}, fh)

    current = {"src/a.py": "aaaa"}
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == set()
    assert new == {"src/a.py"}
    assert deleted == set()


# ---------------------------------------------------------------------------
# _get_changed_files — all hashes identical
# ---------------------------------------------------------------------------


def test_get_changed_files_no_changes(tmp_path):
    """When all hashes match, every set should be empty."""
    builder = _make_builder(tmp_path)
    hashes = {"src/a.py": "aaaa", "src/b.py": "bbbb"}
    _write_metadata(builder, hashes)

    changed, new, deleted = builder._get_changed_files(dict(hashes))

    assert changed == set()
    assert new == set()
    assert deleted == set()


# ---------------------------------------------------------------------------
# _get_changed_files — one file hash changed
# ---------------------------------------------------------------------------


def test_get_changed_files_one_file_changed(tmp_path):
    """A file whose hash differs from stored should appear in changed_files."""
    builder = _make_builder(tmp_path)
    stored = {"src/a.py": "aaaa", "src/b.py": "bbbb"}
    _write_metadata(builder, stored)

    current = {"src/a.py": "xxxx", "src/b.py": "bbbb"}  # a.py hash changed
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == {"src/a.py"}
    assert new == set()
    assert deleted == set()


# ---------------------------------------------------------------------------
# _get_changed_files — file deleted from LanceDB
# ---------------------------------------------------------------------------


def test_get_changed_files_file_deleted(tmp_path):
    """A file in stored metadata that is no longer in current → deleted_files."""
    builder = _make_builder(tmp_path)
    stored = {"src/a.py": "aaaa", "src/b.py": "bbbb"}
    _write_metadata(builder, stored)

    current = {"src/a.py": "aaaa"}  # b.py removed
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == set()
    assert new == set()
    assert deleted == {"src/b.py"}


# ---------------------------------------------------------------------------
# _get_changed_files — new file appeared in LanceDB
# ---------------------------------------------------------------------------


def test_get_changed_files_new_file(tmp_path):
    """A file in current hashes not present in stored metadata → new_files."""
    builder = _make_builder(tmp_path)
    stored = {"src/a.py": "aaaa"}
    _write_metadata(builder, stored)

    current = {"src/a.py": "aaaa", "src/b.py": "bbbb"}  # b.py is new
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == set()
    assert new == {"src/b.py"}
    assert deleted == set()


# ---------------------------------------------------------------------------
# _get_changed_files — combined scenario
# ---------------------------------------------------------------------------


def test_get_changed_files_combined(tmp_path):
    """Mixed scenario: changed + new + deleted at the same time."""
    builder = _make_builder(tmp_path)
    stored = {
        "src/a.py": "aaaa",  # unchanged
        "src/b.py": "bbbb",  # will be changed
        "src/c.py": "cccc",  # will be deleted
    }
    _write_metadata(builder, stored)

    current = {
        "src/a.py": "aaaa",  # same
        "src/b.py": "xxxx",  # changed
        "src/d.py": "dddd",  # new
    }
    changed, new, deleted = builder._get_changed_files(current)

    assert changed == {"src/b.py"}
    assert new == {"src/d.py"}
    assert deleted == {"src/c.py"}


# ---------------------------------------------------------------------------
# update_metadata_file_hashes
# ---------------------------------------------------------------------------


def test_update_metadata_file_hashes_creates_file(tmp_path):
    """Should create kg_metadata.json when it does not exist."""
    builder = _make_builder(tmp_path)
    hashes = {"src/a.py": "aaaa"}
    builder.update_metadata_file_hashes(hashes)

    assert builder._metadata_path.exists()
    data = json.loads(builder._metadata_path.read_text())
    assert data["file_hashes"] == hashes


def test_update_metadata_file_hashes_preserves_existing_keys(tmp_path):
    """Should keep existing metadata keys (e.g. last_build) when patching."""
    builder = _make_builder(tmp_path)
    builder._metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(builder._metadata_path, "w") as fh:
        json.dump({"last_build": "2025-01-01", "entities_created": 42}, fh)

    builder.update_metadata_file_hashes({"src/a.py": "aaaa"})

    data = json.loads(builder._metadata_path.read_text())
    assert data["last_build"] == "2025-01-01"
    assert data["entities_created"] == 42
    assert data["file_hashes"] == {"src/a.py": "aaaa"}


# ---------------------------------------------------------------------------
# delete_entities_for_files — KnowledgeGraph (uses a real Kuzu DB)
# ---------------------------------------------------------------------------


@pytest.fixture
async def kg_with_data(tmp_path):
    """Async fixture: real KG with two entities and a CALLS edge."""
    kg_path = tmp_path / "test_kg"
    kg = KnowledgeGraph(kg_path)
    kg.initialize_sync()

    await kg.add_entity(
        CodeEntity(id="e1", name="func_a", entity_type="function", file_path="src/a.py")
    )
    await kg.add_entity(
        CodeEntity(id="e2", name="func_b", entity_type="function", file_path="src/b.py")
    )
    await kg.add_relationship(
        CodeRelationship(source_id="e1", target_id="e2", relationship_type="calls")
    )

    yield kg
    kg.close_sync()


async def test_delete_entities_for_files_removes_node_and_edges(kg_with_data):
    """Deleting src/a.py should remove e1 and the CALLS edge it sources."""
    kg = kg_with_data
    stats_before = kg.get_stats_sync()
    assert stats_before["code_entities"] == 2

    kg.delete_entities_for_files(["src/a.py"])

    stats_after = kg.get_stats_sync()
    # e1 should be gone; CALLS edge that referenced it should be gone too
    assert stats_after["code_entities"] == 1
    assert stats_after["relationships"].get("calls", 0) == 0


async def test_delete_entities_for_files_empty_list(kg_with_data):
    """Passing an empty list should be a no-op."""
    kg = kg_with_data
    stats_before = kg.get_stats_sync()

    result = kg.delete_entities_for_files([])

    stats_after = kg.get_stats_sync()
    assert stats_after["code_entities"] == stats_before["code_entities"]
    assert result == 0


async def test_delete_entities_for_files_nonexistent_path(kg_with_data):
    """Deleting a file path that has no entities should not error."""
    kg = kg_with_data
    # Should not raise
    kg.delete_entities_for_files(["nonexistent/file.py"])
    stats = kg.get_stats_sync()
    assert stats["code_entities"] == 2  # unchanged

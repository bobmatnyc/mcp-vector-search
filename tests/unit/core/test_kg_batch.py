"""Test batch insert operations for knowledge graph."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.knowledge_graph import (
    CodeEntity,
    CodeRelationship,
    DocSection,
    KnowledgeGraph,
)


@pytest.fixture
async def kg():
    """Create temporary knowledge graph for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kg_path = Path(tmpdir) / "test_kg"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()
        yield kg
        await kg.close()


@pytest.mark.asyncio
async def test_add_entities_batch(kg):
    """Test batch entity insertion."""
    entities = [
        CodeEntity(
            id=f"entity:{i}",
            name=f"function_{i}",
            entity_type="function",
            file_path=f"test_{i}.py",
        )
        for i in range(10)
    ]

    count = await kg.add_entities_batch(entities, batch_size=5)
    assert count == 10

    # Verify entities were inserted
    stats = await kg.get_stats()
    assert stats["code_entities"] == 10


@pytest.mark.asyncio
async def test_add_entities_batch_empty(kg):
    """Test batch entity insertion with empty list."""
    count = await kg.add_entities_batch([], batch_size=5)
    assert count == 0


@pytest.mark.asyncio
async def test_add_entities_batch_single(kg):
    """Test batch entity insertion with single entity."""
    entities = [
        CodeEntity(
            id="entity:1",
            name="function_1",
            entity_type="function",
            file_path="test.py",
        )
    ]

    count = await kg.add_entities_batch(entities, batch_size=5)
    assert count == 1

    stats = await kg.get_stats()
    assert stats["code_entities"] == 1


@pytest.mark.asyncio
async def test_add_doc_sections_batch(kg):
    """Test batch doc section insertion."""
    docs = [
        DocSection(
            id=f"doc:{i}",
            name=f"Section {i}",
            file_path="README.md",
            level=1,
            line_start=i * 10,
            line_end=i * 10 + 5,
        )
        for i in range(10)
    ]

    count = await kg.add_doc_sections_batch(docs, batch_size=5)
    assert count == 10

    stats = await kg.get_stats()
    assert stats["doc_sections"] == 10


@pytest.mark.asyncio
async def test_add_tags_batch(kg):
    """Test batch tag insertion."""
    tags = [f"tag{i}" for i in range(10)]

    count = await kg.add_tags_batch(tags, batch_size=5)
    assert count == 10

    stats = await kg.get_stats()
    assert stats["tags"] == 10


@pytest.mark.asyncio
async def test_add_tags_batch_deduplicate(kg):
    """Test batch tag insertion with duplicates."""
    tags = ["python", "rust", "python", "go", "rust"]

    count = await kg.add_tags_batch(tags, batch_size=5)
    assert count == 3  # Only unique tags

    stats = await kg.get_stats()
    assert stats["tags"] == 3


@pytest.mark.asyncio
async def test_add_relationships_batch(kg):
    """Test batch relationship insertion."""
    # First add entities
    entities = [
        CodeEntity(
            id=f"entity:{i}",
            name=f"function_{i}",
            entity_type="function",
            file_path="test.py",
        )
        for i in range(5)
    ]
    await kg.add_entities_batch(entities)

    # Create CALLS relationships
    relationships = [
        CodeRelationship(
            source_id=f"entity:{i}",
            target_id=f"entity:{i + 1}",
            relationship_type="calls",
        )
        for i in range(4)
    ]

    count = await kg.add_relationships_batch(relationships, batch_size=2)
    assert count == 4

    stats = await kg.get_stats()
    assert stats["relationships"]["calls"] == 4


@pytest.mark.asyncio
async def test_add_relationships_batch_multiple_types(kg):
    """Test batch relationship insertion with multiple types."""
    # Add entities
    entities = [
        CodeEntity(
            id=f"entity:{i}",
            name=f"func_{i}",
            entity_type="function",
            file_path="test.py",
        )
        for i in range(6)
    ]
    await kg.add_entities_batch(entities)

    # Create mixed relationships
    relationships = [
        CodeRelationship(
            source_id="entity:0", target_id="entity:1", relationship_type="calls"
        ),
        CodeRelationship(
            source_id="entity:1", target_id="entity:2", relationship_type="calls"
        ),
        CodeRelationship(
            source_id="entity:2", target_id="entity:3", relationship_type="inherits"
        ),
        CodeRelationship(
            source_id="entity:3", target_id="entity:4", relationship_type="inherits"
        ),
        CodeRelationship(
            source_id="entity:4", target_id="entity:5", relationship_type="contains"
        ),
    ]

    count = await kg.add_relationships_batch(relationships, batch_size=2)
    assert count == 5

    stats = await kg.get_stats()
    assert stats["relationships"]["calls"] == 2
    assert stats["relationships"]["inherits"] == 2
    assert stats["relationships"]["contains"] == 1


@pytest.mark.asyncio
async def test_add_part_of_batch(kg):
    """Test batch PART_OF relationship insertion."""
    from mcp_vector_search.core.knowledge_graph import Project

    # Add project
    project = Project(id="project:test", name="test", description="Test project")
    await kg.add_project(project)

    # Add entities
    entities = [
        CodeEntity(
            id=f"entity:{i}",
            name=f"func_{i}",
            entity_type="function",
            file_path="test.py",
        )
        for i in range(10)
    ]
    await kg.add_entities_batch(entities)

    # Add PART_OF relationships
    entity_ids = [f"entity:{i}" for i in range(10)]
    count = await kg.add_part_of_batch(entity_ids, "project:test", batch_size=5)
    assert count == 10

    stats = await kg.get_stats()
    assert stats["relationships"]["part_of"] == 10


@pytest.mark.asyncio
async def test_batch_operations_large_scale(kg):
    """Test batch operations with larger dataset."""
    # Add 1000 entities
    entities = [
        CodeEntity(
            id=f"entity:{i}",
            name=f"function_{i}",
            entity_type="function",
            file_path=f"test_{i % 10}.py",
        )
        for i in range(1000)
    ]

    count = await kg.add_entities_batch(entities, batch_size=500)
    assert count == 1000

    # Add 500 relationships
    relationships = [
        CodeRelationship(
            source_id=f"entity:{i}",
            target_id=f"entity:{i + 1}",
            relationship_type="calls",
        )
        for i in range(500)
    ]

    rel_count = await kg.add_relationships_batch(relationships, batch_size=500)
    assert rel_count == 500

    stats = await kg.get_stats()
    assert stats["code_entities"] == 1000
    assert stats["relationships"]["calls"] == 500


@pytest.mark.asyncio
async def test_batch_vs_individual_equivalence(kg):
    """Test that batch and individual inserts produce same results."""
    # Batch insert
    entities_batch = [
        CodeEntity(
            id=f"batch:{i}",
            name=f"func_{i}",
            entity_type="function",
            file_path="test.py",
        )
        for i in range(5)
    ]
    batch_count = await kg.add_entities_batch(entities_batch)

    # Individual insert
    entities_individual = [
        CodeEntity(
            id=f"indiv:{i}",
            name=f"func_{i}",
            entity_type="function",
            file_path="test.py",
        )
        for i in range(5)
    ]
    individual_count = 0
    for entity in entities_individual:
        await kg.add_entity(entity)
        individual_count += 1

    assert batch_count == individual_count == 5

    stats = await kg.get_stats()
    assert stats["code_entities"] == 10  # 5 batch + 5 individual


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

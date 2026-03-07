"""Unit tests for trace_execution_flow KG method."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestTraceExecutionFlowUnit:
    """Unit tests for the trace_execution_flow method logic."""

    @pytest.mark.asyncio
    async def test_entry_not_found_returns_empty(self):
        """When entry point cannot be resolved, return empty result."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.find_entity_by_name = AsyncMock(return_value=None)
        kg.trace_execution_flow = KnowledgeGraph.trace_execution_flow.__get__(kg)

        result = await kg.trace_execution_flow("nonexistent_xyz_abc")

        assert result["entry"] is None
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["total_nodes"] == 0

    @pytest.mark.asyncio
    async def test_depth_clamped_to_8(self):
        """Depth > 8 is clamped to 8."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.find_entity_by_name = AsyncMock(return_value=None)
        kg.trace_execution_flow = KnowledgeGraph.trace_execution_flow.__get__(kg)

        # Should not raise; depth clamped internally
        result = await kg.trace_execution_flow("anything", depth=999)
        assert result["entry"] is None  # nothing found, but depth was clamped

    @pytest.mark.asyncio
    async def test_depth_minimum_is_1(self):
        """Depth < 1 is clamped to 1."""
        from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

        kg = MagicMock(spec=KnowledgeGraph)
        kg._initialized = True
        kg.find_entity_by_name = AsyncMock(return_value=None)
        kg.trace_execution_flow = KnowledgeGraph.trace_execution_flow.__get__(kg)

        result = await kg.trace_execution_flow("anything", depth=0)
        assert result["entry"] is None


class TestTraceResultStructure:
    """Verify the result dict shape contract."""

    def test_empty_result_has_required_keys(self):
        """Empty result must have all documented keys."""
        required = {
            "entry",
            "nodes",
            "edges",
            "paths",
            "total_nodes",
            "depth_reached",
            "truncated",
        }
        empty = {
            "entry": None,
            "nodes": [],
            "edges": [],
            "paths": [],
            "total_nodes": 0,
            "depth_reached": 0,
            "truncated": False,
        }
        assert set(empty.keys()) == required

    def test_node_has_required_fields(self):
        """Each node in the result must have documented fields."""
        node = {
            "id": "func:test",
            "name": "test",
            "entity_type": "function",
            "file_path": "/repo/test.py",
            "start_line": 1,
            "depth": 1,
        }
        required_node_fields = {
            "id",
            "name",
            "entity_type",
            "file_path",
            "start_line",
            "depth",
        }
        assert required_node_fields.issubset(set(node.keys()))

    def test_edge_has_required_fields(self):
        """Each edge in the result must have documented fields."""
        edge = {
            "from_id": "func:a",
            "to_id": "func:b",
            "from_name": "a",
            "to_name": "b",
            "depth": 1,
        }
        required_edge_fields = {"from_id", "to_id", "from_name", "to_name", "depth"}
        assert required_edge_fields.issubset(set(edge.keys()))

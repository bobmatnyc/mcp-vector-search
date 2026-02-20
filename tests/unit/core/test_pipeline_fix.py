"""Unit tests for pipeline parallelism fix.

This test verifies that the fix for pipeline parallelism is working correctly:
- parse_file() uses asyncio.to_thread() to avoid blocking event loop
- build_chunk_hierarchy() uses asyncio.to_thread() in indexer
- Both sync and async paths produce identical results
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcp_vector_search.core.chunk_processor import ChunkProcessor
from mcp_vector_search.core.models import CodeChunk
from mcp_vector_search.parsers.registry import get_parser_registry
from mcp_vector_search.utils.monorepo import MonorepoDetector


class TestPipelineFix:
    """Test suite for pipeline parallelism fix."""

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test Python file."""
        test_py = tmp_path / "test.py"
        test_py.write_text(
            """
def function_one():
    '''Test function 1'''
    return 1

class TestClass:
    '''Test class'''

    def method_one(self):
        '''Test method'''
        return 2
"""
        )
        return test_py

    @pytest.fixture
    def chunk_processor(self, tmp_path):
        """Create a ChunkProcessor instance."""
        parser_registry = get_parser_registry()
        monorepo_detector = MonorepoDetector(tmp_path)
        return ChunkProcessor(
            parser_registry=parser_registry,
            monorepo_detector=monorepo_detector,
            use_multiprocessing=False,  # Single-threaded for tests
        )

    def test_parse_file_sync_exists(self, chunk_processor):
        """Verify parse_file_sync method exists."""
        assert hasattr(chunk_processor, "parse_file_sync")
        assert callable(chunk_processor.parse_file_sync)

    @pytest.mark.asyncio
    async def test_parse_file_uses_to_thread(self, chunk_processor, test_file):
        """Verify parse_file() uses asyncio.to_thread() internally."""
        # Mock asyncio.to_thread in the chunk_processor module
        with patch(
            "mcp_vector_search.core.chunk_processor.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_to_thread:
            # Mock the return value with valid CodeChunk
            mock_chunks = [
                CodeChunk(
                    chunk_id="test1",
                    file_path=test_file,
                    content="def foo(): pass",
                    language="python",
                    start_line=1,
                    end_line=1,
                )
            ]
            mock_to_thread.return_value = mock_chunks

            # Call parse_file
            result = await chunk_processor.parse_file(test_file)

            # Verify to_thread was called with parse_file_sync
            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args[0][0] == chunk_processor.parse_file_sync
            assert call_args[0][1] == test_file

            # Verify result matches mock
            assert result == mock_chunks

    @pytest.mark.asyncio
    async def test_sync_and_async_produce_same_results(
        self, chunk_processor, test_file
    ):
        """Verify sync and async parsing paths produce identical results."""
        # Parse with sync method
        chunks_sync = chunk_processor.parse_file_sync(test_file)

        # Parse with async method
        chunks_async = await chunk_processor.parse_file(test_file)

        # Should produce same number of chunks
        assert len(chunks_sync) == len(chunks_async)

        # Should produce same chunk IDs (content hashes)
        sync_ids = {c.chunk_id for c in chunks_sync}
        async_ids = {c.chunk_id for c in chunks_async}
        assert sync_ids == async_ids

    @pytest.mark.asyncio
    async def test_build_hierarchy_is_sync(self, chunk_processor):
        """Verify build_chunk_hierarchy is synchronous (can be wrapped in to_thread)."""
        # Create test chunks
        test_chunks = [
            CodeChunk(
                chunk_id="test1",
                file_path=Path("/test.py"),
                content="class TestClass: pass",
                language="python",
                start_line=1,
                end_line=1,
                chunk_type="class",
                class_name="TestClass",
            )
        ]

        # This should work synchronously without await
        result = chunk_processor.build_chunk_hierarchy(test_chunks)

        # Should return same chunks with hierarchy
        assert len(result) == len(test_chunks)
        assert result[0].chunk_depth == 1  # Top-level class

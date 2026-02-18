"""Tests for ESC key cancellation functionality."""

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_vector_search.cli.commands.index import (
    _reset_cancellation_flag,
    _restore_index_from_backup,
    _start_esc_listener,
    _stop_esc_listener,
)


class TestEscListener:
    """Test ESC key listener functionality."""

    def test_start_esc_listener(self):
        """Test starting the ESC listener thread."""
        _reset_cancellation_flag()
        _start_esc_listener()

        # Verify thread was started (if running in Unix with TTY)
        # This is best-effort - may not work in all test environments
        from mcp_vector_search.cli.commands.index import _esc_listener_thread

        # Thread should be created (may or may not be alive depending on environment)
        assert _esc_listener_thread is not None

        # Clean up
        _stop_esc_listener()

    def test_stop_esc_listener(self):
        """Test stopping the ESC listener thread."""
        _reset_cancellation_flag()
        _start_esc_listener()
        _stop_esc_listener()

        from mcp_vector_search.cli.commands.index import (
            _cancellation_flag,
            _esc_listener_thread,
        )

        # Thread should be cleaned up
        assert _esc_listener_thread is None or not _esc_listener_thread.is_alive()
        # Flag may be set (by stop) or clear
        assert isinstance(_cancellation_flag, threading.Event)

    def test_reset_cancellation_flag(self):
        """Test resetting the cancellation flag."""
        from mcp_vector_search.cli.commands.index import _cancellation_flag

        # Set flag
        _cancellation_flag.set()
        assert _cancellation_flag.is_set()

        # Reset flag
        _reset_cancellation_flag()
        assert not _cancellation_flag.is_set()


class TestIndexBackup:
    """Test index backup and restore functionality."""

    def test_restore_index_from_backup_success(self, tmp_path):
        """Test successful restoration from backup."""
        # Setup
        cache_path = tmp_path / "cache"
        backup_path = tmp_path / "cache.backup"

        # Create backup with test file
        backup_path.mkdir()
        (backup_path / "test.txt").write_text("backup content")

        # Create corrupted cache
        cache_path.mkdir()
        (cache_path / "corrupted.txt").write_text("corrupted")

        # Restore
        _restore_index_from_backup(cache_path, backup_path)

        # Verify
        assert cache_path.exists()
        assert (cache_path / "test.txt").read_text() == "backup content"
        assert not (cache_path / "corrupted.txt").exists()
        assert not backup_path.exists()  # Backup should be removed

    def test_restore_index_from_backup_no_backup(self, tmp_path):
        """Test restore when no backup exists."""
        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        (cache_path / "test.txt").write_text("original")

        # Call with non-existent backup
        _restore_index_from_backup(cache_path, tmp_path / "nonexistent")

        # Original should be unchanged
        assert cache_path.exists()
        assert (cache_path / "test.txt").read_text() == "original"

    def test_restore_index_from_backup_none(self, tmp_path):
        """Test restore when backup_path is None."""
        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        (cache_path / "test.txt").write_text("original")

        # Call with None backup
        _restore_index_from_backup(cache_path, None)

        # Original should be unchanged
        assert cache_path.exists()
        assert (cache_path / "test.txt").read_text() == "original"


class TestIndexerCancellation:
    """Test cancellation in the indexer."""

    @pytest.mark.asyncio
    async def test_indexer_checks_cancellation_flag(self):
        """Test that indexer checks cancellation flag during processing."""
        from mcp_vector_search.core.indexer import SemanticIndexer

        # Create mock database
        mock_db = MagicMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock()

        # Create indexer
        indexer = SemanticIndexer(
            database=mock_db,
            project_root=Path("/tmp"),
            file_extensions=[".py"],
        )

        # Set cancellation flag
        cancellation_flag = threading.Event()
        cancellation_flag.set()
        indexer.cancellation_flag = cancellation_flag

        # Mock methods
        indexer.chunks_backend = MagicMock()
        indexer.chunks_backend._db = MagicMock()
        indexer.chunks_backend.initialize = AsyncMock()
        indexer.vectors_backend = MagicMock()
        indexer.vectors_backend._db = MagicMock()
        indexer.vectors_backend.initialize = AsyncMock()
        indexer.metadata = MagicMock()
        indexer.metadata.load = MagicMock(return_value={})
        indexer.metadata.write_indexing_run_header = MagicMock()

        # Process with cancellation flag set
        result = []
        async for item in indexer.index_files_with_progress(
            [Path("/tmp/test.py")], force_reindex=False
        ):
            result.append(item)

        # Should return early (no items processed)
        assert len(result) == 0

    def test_cancellation_flag_passed_to_indexer(self):
        """Test that cancellation flag is properly set on indexer."""
        from mcp_vector_search.core.indexer import SemanticIndexer

        # Create mock database
        mock_db = MagicMock()

        # Create indexer
        indexer = SemanticIndexer(
            database=mock_db,
            project_root=Path("/tmp"),
            file_extensions=[".py"],
        )

        # Initially no cancellation flag
        assert indexer.cancellation_flag is None

        # Set cancellation flag
        cancellation_flag = threading.Event()
        indexer.cancellation_flag = cancellation_flag

        # Verify it was set
        assert indexer.cancellation_flag is cancellation_flag
        assert not indexer.cancellation_flag.is_set()

        # Set the flag
        cancellation_flag.set()
        assert indexer.cancellation_flag.is_set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

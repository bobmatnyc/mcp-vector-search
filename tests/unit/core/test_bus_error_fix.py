"""Tests for bus error corruption detection and recovery.

This module tests the multi-layered defense against ChromaDB SIGBUS crashes
caused by corrupted HNSW binary index files.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from mcp_vector_search.core.corruption_recovery import CorruptionRecovery
from mcp_vector_search.core.dimension_checker import DimensionChecker


class TestBinaryFileCorruptionDetection:
    """Test detection of corrupted .bin files before they cause bus errors."""

    @pytest.mark.asyncio
    async def test_detect_zero_size_bin_file(self):
        """Test that zero-size .bin files are detected as corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create a zero-size .bin file
            corrupted_bin = index_dir / "hnsw_index.bin"
            corrupted_bin.touch()

            # Run corruption detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            assert is_corrupted, "Zero-size .bin file should be detected as corrupted"

    @pytest.mark.asyncio
    async def test_detect_small_bin_file(self):
        """Test that suspiciously small .bin files are detected as corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create a very small .bin file (< 100 bytes)
            corrupted_bin = index_dir / "hnsw_index.bin"
            corrupted_bin.write_bytes(b"corrupted" * 5)  # 45 bytes

            # Run corruption detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            assert is_corrupted, "Suspiciously small .bin file should be detected"

    @pytest.mark.asyncio
    async def test_detect_truncated_bin_file(self):
        """Test that truncated .bin files are detected as corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create a .bin file that's large enough but truncated
            # (header reads less than expected)
            corrupted_bin = index_dir / "hnsw_index.bin"

            # Write partial data then truncate
            with open(corrupted_bin, "wb") as f:
                f.write(b"header" * 100)  # Write some data
                f.flush()

            # Manually truncate to cause read issues
            # This simulates incomplete writes
            corrupted_bin.write_bytes(b"partial")

            # Run corruption detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            # Should detect the small size
            assert is_corrupted, "Truncated .bin file should be detected"

    @pytest.mark.asyncio
    async def test_detect_all_zero_bin_file(self):
        """Test that all-zero .bin files are detected as corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create a .bin file filled with zeros (corrupted/incomplete write)
            corrupted_bin = index_dir / "hnsw_index.bin"
            corrupted_bin.write_bytes(b"\x00" * 5000)

            # Run corruption detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            assert is_corrupted, "All-zero .bin file should be detected as corrupted"

    @pytest.mark.asyncio
    async def test_valid_bin_file_not_flagged(self):
        """Test that valid .bin files are not flagged as corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create a valid-looking .bin file
            valid_bin = index_dir / "hnsw_index.bin"
            valid_bin.write_bytes(b"HNSW_HEADER" + b"data" * 1000)

            # Run corruption detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            assert not is_corrupted, "Valid .bin file should not be flagged"


# Define pickle-able classes at module level (can't be nested in test functions)
class FakeCollection:
    """Fake collection that can be pickled for subprocess testing."""

    def count(self):
        return 42


class HangingCollection:
    """Collection that hangs during count()."""

    def count(self):
        import time

        time.sleep(10)
        return 42


class ErrorCollection:
    """Collection that raises an error during count()."""

    def count(self):
        raise RuntimeError("Corrupted index")


class TestSafeCollectionCount:
    """Test subprocess-isolated collection.count() to prevent bus errors."""

    @pytest.mark.asyncio
    async def test_safe_count_success(self):
        """Test successful count operation in subprocess."""
        fake_collection = FakeCollection()

        # Run safe count
        count = await DimensionChecker._safe_collection_count(
            fake_collection, timeout=2.0
        )

        assert count == 42, "Should return correct count from subprocess"

    @pytest.mark.asyncio
    async def test_safe_count_timeout(self):
        """Test that timeouts are handled gracefully."""
        hanging_collection = HangingCollection()

        # Run safe count with short timeout
        count = await DimensionChecker._safe_collection_count(
            hanging_collection, timeout=1.0
        )

        assert count is None, "Timeout should return None"

    @pytest.mark.asyncio
    async def test_safe_count_exception(self):
        """Test that exceptions in subprocess are handled."""
        error_collection = ErrorCollection()

        # Run safe count
        count = await DimensionChecker._safe_collection_count(
            error_collection, timeout=2.0
        )

        assert count is None, "Exception should return None"

    @pytest.mark.asyncio
    async def test_check_compatibility_with_corrupted_count(self):
        """Test dimension checker handles corrupted count gracefully."""
        # Mock collection that would cause bus error
        mock_collection = Mock()
        mock_collection.count = Mock(side_effect=Exception("Bus error simulation"))

        # Mock embedding function
        mock_embedding = Mock()
        mock_embedding.model_name = "test-model"

        # Should not crash, just log warning
        await DimensionChecker.check_compatibility(mock_collection, mock_embedding)

        # Test passes if no exception raised


class TestIntegrationProtection:
    """Test the complete protection flow during database initialization."""

    @pytest.mark.asyncio
    async def test_corruption_detected_before_count(self):
        """Test that corruption is detected before dimension checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create corrupted .bin file
            corrupted_bin = index_dir / "hnsw_index.bin"
            corrupted_bin.touch()  # Zero-size

            # Also create SQLite database
            db_path = persist_dir / "chroma.sqlite3"
            db_path.touch()

            # Run detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            # Should detect corruption BEFORE any ChromaDB operations
            assert is_corrupted, "Corruption should be detected early"

    @pytest.mark.asyncio
    async def test_valid_index_passes_all_checks(self):
        """Test that valid index passes all corruption checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir)
            index_dir = persist_dir / "index"
            index_dir.mkdir(parents=True)

            # Create valid-looking files
            valid_bin = index_dir / "hnsw_index.bin"
            valid_bin.write_bytes(b"HNSW_HEADER" + b"data" * 2000)

            # Run detection
            recovery = CorruptionRecovery(persist_dir)
            is_corrupted = await recovery.detect_corruption()

            assert not is_corrupted, "Valid index should pass all checks"


class TestRecoveryFlow:
    """Test the complete recovery flow for corrupted indices."""

    @pytest.mark.asyncio
    async def test_recovery_creates_backup(self):
        """Test that recovery creates a backup before clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma_data"
            persist_dir.mkdir(parents=True)

            # Create some files to backup
            test_file = persist_dir / "test.bin"
            test_file.write_bytes(b"test data")

            # Run recovery
            recovery = CorruptionRecovery(persist_dir)
            await recovery.recover()

            # Check backup was created
            backup_dir = persist_dir.parent / f"{persist_dir.name}_backup"
            assert backup_dir.exists(), "Backup directory should be created"

            backups = list(backup_dir.glob("backup_*"))
            assert len(backups) > 0, "At least one backup should exist"

    @pytest.mark.asyncio
    async def test_recovery_clears_corrupted_index(self):
        """Test that recovery clears the corrupted index directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma_data"
            persist_dir.mkdir(parents=True)

            # Create corrupted files
            corrupted_file = persist_dir / "corrupted.bin"
            corrupted_file.write_bytes(b"corrupted")

            # Run recovery
            recovery = CorruptionRecovery(persist_dir)
            await recovery.recover()

            # Directory should be recreated empty
            assert persist_dir.exists(), "Directory should be recreated"
            assert not corrupted_file.exists(), "Corrupted file should be removed"

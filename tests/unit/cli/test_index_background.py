"""Tests for background indexing functionality."""

import json
import signal
from unittest.mock import MagicMock, Mock, patch

from mcp_vector_search.cli.commands.index import (
    _cancel_background_indexer,
    _is_process_alive,
    _show_background_status,
    _spawn_background_indexer,
)


class TestSpawnBackgroundIndexer:
    """Test background process spawning."""

    @patch("mcp_vector_search.cli.commands.index.subprocess.Popen")
    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_spawn_background_indexer_success_unix(
        self, mock_is_alive, mock_popen, tmp_path
    ):
        """Test spawning background indexer on Unix."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        # Mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_is_alive.return_value = False

        # Test
        with patch("sys.platform", "linux"):
            _spawn_background_indexer(project_root, force=True, extensions=".py,.js")

        # Verify
        assert mock_popen.called
        call_args = mock_popen.call_args

        # Check command
        cmd = call_args[0][0]
        assert "-m" in cmd
        assert "mcp_vector_search.cli.commands.index_background" in cmd
        assert "--project-root" in cmd
        assert str(project_root) in cmd
        assert "--force" in cmd
        assert "--extensions" in cmd
        assert ".py,.js" in cmd

        # Check Unix detachment options
        assert call_args[1]["start_new_session"] is True
        assert call_args[1]["stdout"] is not None
        assert call_args[1]["stderr"] is not None
        assert call_args[1]["stdin"] is not None

    @patch("mcp_vector_search.cli.commands.index.subprocess.Popen")
    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_spawn_background_indexer_success_windows(
        self, mock_is_alive, mock_popen, tmp_path
    ):
        """Test spawning background indexer on Windows."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        # Mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_is_alive.return_value = False

        # Test
        with patch("sys.platform", "win32"):
            _spawn_background_indexer(project_root, force=False, extensions=None)

        # Verify
        assert mock_popen.called
        call_args = mock_popen.call_args

        # Check Windows detachment options
        creationflags = call_args[1]["creationflags"]
        detached_process = 0x00000008
        create_new_process_group = 0x00000200
        assert creationflags & detached_process
        assert creationflags & create_new_process_group

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_spawn_background_indexer_already_running(self, mock_is_alive, tmp_path):
        """Test spawning when process already running."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_file.write_text(json.dumps({"pid": 9999, "status": "running"}))

        mock_is_alive.return_value = True

        # Test - should not spawn new process
        with patch(
            "mcp_vector_search.cli.commands.index.subprocess.Popen"
        ) as mock_popen:
            _spawn_background_indexer(project_root)
            assert not mock_popen.called

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    @patch("mcp_vector_search.cli.commands.index.subprocess.Popen")
    def test_spawn_background_indexer_stale_progress_file(
        self, mock_popen, mock_is_alive, tmp_path
    ):
        """Test spawning with stale progress file (dead process)."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_file.write_text(json.dumps({"pid": 9999, "status": "running"}))

        mock_is_alive.return_value = False  # Process is dead
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # Test
        with patch("sys.platform", "linux"):
            _spawn_background_indexer(project_root)

        # Verify - stale file should be removed and new process spawned
        assert mock_popen.called


class TestIsProcessAlive:
    """Test process alive checking."""

    @patch("os.kill")
    def test_is_process_alive_unix_alive(self, mock_kill):
        """Test process alive check on Unix (alive)."""
        # Setup
        mock_kill.return_value = None  # No exception = process alive

        # Test
        with patch("sys.platform", "linux"):
            result = _is_process_alive(12345)

        # Verify
        assert result is True
        mock_kill.assert_called_once_with(12345, 0)

    @patch("os.kill")
    def test_is_process_alive_unix_dead(self, mock_kill):
        """Test process alive check on Unix (dead)."""
        # Setup
        mock_kill.side_effect = ProcessLookupError()

        # Test
        with patch("sys.platform", "linux"):
            result = _is_process_alive(12345)

        # Verify
        assert result is False

    def test_is_process_alive_windows_alive(self):
        """Test process alive check on Windows (alive)."""
        import ctypes

        # Setup
        mock_kernel32 = MagicMock()
        mock_kernel32.OpenProcess.return_value = 999  # Non-zero = success

        # Create mock windll object
        mock_windll = MagicMock()
        mock_windll.kernel32 = mock_kernel32

        # Test - need to patch at module level and handle ctypes.windll not existing on macOS
        with (
            patch("mcp_vector_search.cli.commands.index.sys.platform", "win32"),
            patch.object(ctypes, "windll", mock_windll, create=True),
        ):
            result = _is_process_alive(12345)

        # Verify
        assert result is True
        mock_kernel32.OpenProcess.assert_called_once()
        mock_kernel32.CloseHandle.assert_called_once_with(999)

    def test_is_process_alive_windows_dead(self):
        """Test process alive check on Windows (dead)."""
        import ctypes

        # Setup
        mock_kernel32 = MagicMock()
        mock_kernel32.OpenProcess.return_value = 0  # Zero = failure

        # Create mock windll object
        mock_windll = MagicMock()
        mock_windll.kernel32 = mock_kernel32

        # Test - need to patch at module level and handle ctypes.windll not existing on macOS
        with (
            patch("mcp_vector_search.cli.commands.index.sys.platform", "win32"),
            patch.object(ctypes, "windll", mock_windll, create=True),
        ):
            result = _is_process_alive(12345)

        # Verify
        assert result is False


class TestShowBackgroundStatus:
    """Test status display."""

    def test_show_status_no_progress_file(self, tmp_path, capsys):
        """Test status when no progress file exists."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        # Test
        _show_background_status(project_root)

        # Verify - should indicate no indexing in progress
        # (output captured by print_info, which uses rich console)

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_show_status_process_running(self, mock_is_alive, tmp_path):
        """Test status when process is running."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_data = {
            "pid": 12345,
            "status": "running",
            "total_files": 100,
            "processed_files": 45,
            "current_file": "src/example.py",
            "chunks_created": 250,
            "errors": 2,
            "eta_seconds": 320,
            "last_updated": "2025-12-20T10:35:15Z",
        }
        progress_file.write_text(json.dumps(progress_data))

        mock_is_alive.return_value = True

        # Test
        _show_background_status(project_root)

        # Verify - should display table with progress info
        # (output goes to rich console, hard to capture in test)

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_show_status_process_dead(self, mock_is_alive, tmp_path):
        """Test status when process is dead."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_data = {"pid": 12345, "status": "running"}
        progress_file.write_text(json.dumps(progress_data))

        mock_is_alive.return_value = False

        # Test
        _show_background_status(project_root)

        # Verify - should indicate process is dead


class TestCancelBackgroundIndexer:
    """Test canceling background indexer."""

    def test_cancel_no_progress_file(self, tmp_path):
        """Test cancel when no progress file exists."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        # Test
        _cancel_background_indexer(project_root, force=True)

        # Verify - should handle gracefully

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_cancel_process_already_dead(self, mock_is_alive, tmp_path):
        """Test cancel when process is already dead."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_data = {"pid": 12345, "status": "running"}
        progress_file.write_text(json.dumps(progress_data))

        mock_is_alive.return_value = False

        # Test
        _cancel_background_indexer(project_root, force=True)

        # Verify - should clean up progress file
        assert not progress_file.exists()

    @patch("mcp_vector_search.cli.commands.index.os.kill")
    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_cancel_unix_success(self, mock_is_alive, mock_kill, tmp_path):
        """Test successful cancellation on Unix."""
        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_data = {"pid": 12345, "status": "running"}
        progress_file.write_text(json.dumps(progress_data))

        mock_is_alive.return_value = True

        # Test
        with (
            patch("mcp_vector_search.cli.commands.index.sys.platform", "linux"),
            patch("time.sleep"),
        ):
            _cancel_background_indexer(project_root, force=True)

        # Verify - os.kill called with SIGTERM (signal 15)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        # Progress file should be cleaned up
        assert not progress_file.exists()

    @patch("mcp_vector_search.cli.commands.index._is_process_alive")
    def test_cancel_windows_success(self, mock_is_alive, tmp_path):
        """Test successful cancellation on Windows."""
        import ctypes

        # Setup
        project_root = tmp_path
        config_dir = project_root / ".mcp-vector-search"
        config_dir.mkdir()

        progress_file = config_dir / "indexing_progress.json"
        progress_data = {"pid": 12345, "status": "running"}
        progress_file.write_text(json.dumps(progress_data))

        mock_is_alive.return_value = True

        # Mock ctypes
        mock_kernel32 = MagicMock()
        mock_kernel32.OpenProcess.return_value = 999

        # Create mock windll object
        mock_windll = MagicMock()
        mock_windll.kernel32 = mock_kernel32

        # Test - need to patch at module level and handle ctypes.windll not existing on macOS
        with (
            patch("mcp_vector_search.cli.commands.index.sys.platform", "win32"),
            patch.object(ctypes, "windll", mock_windll, create=True),
            patch("time.sleep"),
        ):
            _cancel_background_indexer(project_root, force=True)

        # Verify
        mock_kernel32.OpenProcess.assert_called_once()
        mock_kernel32.TerminateProcess.assert_called_once_with(999, 0)
        mock_kernel32.CloseHandle.assert_called_once_with(999)

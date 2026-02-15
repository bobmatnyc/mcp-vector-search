"""Unit tests for project config extension merging on upgrades."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mcp_vector_search.config.defaults import DEFAULT_FILE_EXTENSIONS
from mcp_vector_search.core.project import ProjectManager


class TestProjectConfigExtensionMerge:
    """Test cases for automatic extension merging on config load."""

    def test_merge_new_extensions_on_load(self, tmp_path: Path):
        """Test that new extensions from DEFAULT_FILE_EXTENSIONS are merged on load."""
        # Create a config with a subset of extensions (simulating old version)
        old_extensions = [".py", ".js", ".ts", ".java"]
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        # Create mock config data
        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "file_extensions": old_extensions,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python", "javascript"],
        }

        # Write config to file
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Create index directory to satisfy is_initialized() check
        config_dir.mkdir(exist_ok=True)

        # Create project manager and load config
        pm = ProjectManager(project_root=tmp_path)
        config = pm.load_config()

        # Verify that new extensions are present
        # .dart and .arb should now be in the config (they're in DEFAULT_FILE_EXTENSIONS)
        assert ".dart" in config.file_extensions
        assert ".arb" in config.file_extensions

        # Verify old extensions are still present
        for ext in old_extensions:
            assert ext in config.file_extensions

    def test_no_merge_if_all_extensions_present(self, tmp_path: Path):
        """Test that no merge happens if config already has all current extensions."""
        # Create config with ALL current DEFAULT_FILE_EXTENSIONS
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "file_extensions": DEFAULT_FILE_EXTENSIONS.copy(),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python"],
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config_dir.mkdir(exist_ok=True)

        # Create project manager and load config
        pm = ProjectManager(project_root=tmp_path)

        # Patch logger to verify no INFO message is logged
        with patch("mcp_vector_search.core.project.logger") as mock_logger:
            config = pm.load_config()

            # Verify no merge happened (no info log about adding extensions)
            mock_logger.info.assert_not_called()

        # Verify extensions match defaults
        assert set(config.file_extensions) == set(DEFAULT_FILE_EXTENSIONS)

    def test_user_custom_extensions_preserved(self, tmp_path: Path):
        """Test that user's custom extensions are preserved during merge."""
        # User added custom extensions not in defaults
        custom_extensions = [".py", ".js", ".custom", ".xyz"]
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "file_extensions": custom_extensions,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python"],
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config_dir.mkdir(exist_ok=True)

        pm = ProjectManager(project_root=tmp_path)
        config = pm.load_config()

        # Verify user's custom extensions are still present
        assert ".custom" in config.file_extensions
        assert ".xyz" in config.file_extensions

        # Verify new default extensions are added
        assert ".dart" in config.file_extensions
        assert ".arb" in config.file_extensions

    def test_merge_logs_info_message(self, tmp_path: Path):
        """Test that merge operation logs an INFO message with new extensions."""
        old_extensions = [".py", ".js"]
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "file_extensions": old_extensions,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python"],
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config_dir.mkdir(exist_ok=True)

        pm = ProjectManager(project_root=tmp_path)

        # Patch logger to capture log messages
        with patch("mcp_vector_search.core.project.logger") as mock_logger:
            config = pm.load_config()

            # Verify INFO log was called
            assert mock_logger.info.called

            # Get the log message
            log_call = mock_logger.info.call_args
            log_message = log_call[0][0]

            # Verify message mentions new extensions
            assert "new file extensions" in log_message.lower()
            assert ".dart" in str(log_call)  # Should be in the arguments

    def test_config_not_saved_automatically(self, tmp_path: Path):
        """Test that merged config is NOT automatically saved back to disk."""
        old_extensions = [".py", ".js"]
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "file_extensions": old_extensions,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python"],
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config_dir.mkdir(exist_ok=True)

        # Load config (which triggers merge)
        pm = ProjectManager(project_root=tmp_path)
        config = pm.load_config()

        # Verify merged config in memory has new extensions
        assert ".dart" in config.file_extensions

        # Re-read file from disk and verify it's unchanged
        with open(config_path, "r") as f:
            disk_config = json.load(f)

        # File on disk should still have old extensions only
        assert disk_config["file_extensions"] == old_extensions
        assert ".dart" not in disk_config["file_extensions"]

    def test_missing_file_extensions_key_handled(self, tmp_path: Path):
        """Test that config without file_extensions key is handled gracefully."""
        config_dir = tmp_path / ".mcp-vector-search"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"

        # Config without file_extensions key (edge case)
        config_data = {
            "project_root": str(tmp_path),
            "index_path": str(config_dir),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.5,
            "languages": ["python"],
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config_dir.mkdir(exist_ok=True)

        pm = ProjectManager(project_root=tmp_path)

        # Should not raise an error (merge logic skips if key missing)
        # ProjectConfig will use its default value
        config = pm.load_config()

        # Config should have default extensions from ProjectConfig model
        assert config.file_extensions is not None

"""Unit tests for OpenRouter API key setup functionality.

This module tests the setup_openrouter_api_key function including:
- API key obfuscation
- Interactive prompts with existing keys
- Environment variable precedence
- Config file operations (save, keep, clear)
"""

import os
from unittest.mock import patch

import pytest

from mcp_vector_search.cli.commands.setup import (
    _obfuscate_api_key,
    setup_openrouter_api_key,
)

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def mock_project_root(tmp_path):
    """Create a mock project root with config directory."""
    config_dir = tmp_path / ".mcp-vector-search"
    config_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def clean_env():
    """Ensure OPENROUTER_API_KEY is not in environment."""
    original_value = os.environ.get("OPENROUTER_API_KEY")
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]
    yield
    # Restore original value
    if original_value is not None:
        os.environ["OPENROUTER_API_KEY"] = original_value


# ==============================================================================
# Test API Key Obfuscation
# ==============================================================================


class TestApiKeyObfuscation:
    """Test suite for _obfuscate_api_key function."""

    def test_obfuscate_standard_openrouter_key(self):
        """Test obfuscating standard OpenRouter API key format."""
        api_key = "sk-or-v1-1234567890abcdef"
        result = _obfuscate_api_key(api_key)
        assert result == "sk-or-...cdef"

    def test_obfuscate_short_key(self):
        """Test obfuscating short API key (<10 chars)."""
        api_key = "short"
        result = _obfuscate_api_key(api_key)
        assert result == "****...hort"

    def test_obfuscate_ten_char_key(self):
        """Test obfuscating exactly 10 character key."""
        api_key = "1234567890"
        result = _obfuscate_api_key(api_key)
        assert result == "123456...7890"

    def test_obfuscate_very_short_key(self):
        """Test obfuscating very short key (< 4 chars)."""
        api_key = "abc"
        result = _obfuscate_api_key(api_key)
        assert result == "****...abc"

    def test_obfuscate_long_key(self):
        """Test obfuscating long API key."""
        api_key = "very_long_api_key_with_many_characters"
        result = _obfuscate_api_key(api_key)
        assert result == "very_l...ters"

    def test_obfuscate_empty_string(self):
        """Test obfuscating empty string."""
        result = _obfuscate_api_key("")
        assert result == "****"


# ==============================================================================
# Test Interactive Setup
# ==============================================================================


class TestSetupOpenRouterApiKey:
    """Test suite for setup_openrouter_api_key function."""

    @patch("mcp_vector_search.cli.output.console")
    def test_non_interactive_with_existing_key_from_env(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test non-interactive mode with existing key from environment."""
        # Arrange
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test-key-12345678"

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=False)

        # Assert
        assert result is True
        # Console should not prompt for input in non-interactive mode
        mock_console.input.assert_not_called()

    @patch("mcp_vector_search.cli.output.console")
    def test_non_interactive_without_key(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test non-interactive mode without existing key."""
        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=False)

        # Assert
        assert result is False
        mock_console.input.assert_not_called()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.save_openrouter_api_key")
    def test_interactive_new_key_saved(
        self, mock_save, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user enters new key."""
        # Arrange
        new_key = "sk-or-new-key-12345678"
        mock_console.input.return_value = new_key

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is True
        mock_console.input.assert_called_once()
        mock_save.assert_called_once()
        # Verify the key was passed to save function
        saved_key = mock_save.call_args[0][0]
        assert saved_key == new_key

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.get_openrouter_api_key")
    def test_interactive_keep_existing_key(
        self, mock_get_key, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user presses Enter to keep existing key."""
        # Arrange
        existing_key = "sk-or-existing-key-12345678"
        mock_get_key.return_value = existing_key
        mock_console.input.return_value = ""  # Empty input = keep existing

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is True
        mock_console.input.assert_called_once()
        # Verify prompt shows obfuscated key
        prompt = mock_console.input.call_args[0][0]
        assert "sk-or-...5678" in prompt

    @patch("mcp_vector_search.cli.output.console")
    def test_interactive_skip_no_existing_key(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user presses Enter to skip (no existing key)."""
        # Arrange
        mock_console.input.return_value = ""  # Empty input = skip

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is False
        mock_console.input.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.get_openrouter_api_key")
    @patch("mcp_vector_search.core.config_utils.delete_openrouter_api_key")
    def test_interactive_clear_existing_key(
        self, mock_delete, mock_get_key, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user types 'clear' to remove key."""
        # Arrange
        existing_key = "sk-or-existing-key-12345678"
        mock_get_key.return_value = existing_key
        mock_console.input.return_value = "clear"
        mock_delete.return_value = True

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is False  # No key after deletion
        mock_console.input.assert_called_once()
        mock_delete.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.get_openrouter_api_key")
    @patch("mcp_vector_search.core.config_utils.delete_openrouter_api_key")
    def test_interactive_delete_existing_key(
        self, mock_delete, mock_get_key, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user types 'delete' to remove key."""
        # Arrange
        existing_key = "sk-or-existing-key-12345678"
        mock_get_key.return_value = existing_key
        mock_console.input.return_value = "delete"
        mock_delete.return_value = True

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is False
        mock_delete.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.get_openrouter_api_key")
    def test_interactive_env_var_cannot_be_cleared(
        self, mock_get_key, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - cannot clear key from environment variable."""
        # Arrange
        os.environ["OPENROUTER_API_KEY"] = "sk-or-env-key-12345678"
        mock_get_key.return_value = "sk-or-env-key-12345678"
        mock_console.input.return_value = "clear"

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is True  # Key still exists (in env)
        mock_console.input.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    def test_interactive_keyboard_interrupt(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - user presses Ctrl+C."""
        # Arrange
        mock_console.input.side_effect = KeyboardInterrupt()

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is False
        mock_console.input.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.save_openrouter_api_key")
    def test_interactive_save_error(
        self, mock_save, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - error during save."""
        # Arrange
        mock_console.input.return_value = "sk-or-new-key-12345678"
        mock_save.side_effect = Exception("Save failed")

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is False
        mock_console.input.assert_called_once()
        mock_save.assert_called_once()

    @patch("mcp_vector_search.cli.output.console")
    @patch("mcp_vector_search.core.config_utils.get_openrouter_api_key")
    @patch("mcp_vector_search.core.config_utils.save_openrouter_api_key")
    def test_interactive_update_with_env_var_warning(
        self, mock_save, mock_get_key, mock_console, mock_project_root, clean_env
    ):
        """Test interactive mode - updating key when env var exists shows warning."""
        # Arrange
        os.environ["OPENROUTER_API_KEY"] = "sk-or-env-key-12345678"
        mock_get_key.return_value = "sk-or-env-key-12345678"
        new_key = "sk-or-new-key-12345678"
        mock_console.input.return_value = new_key

        # Act
        result = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert
        assert result is True
        mock_save.assert_called_once()
        # Should show warning about env var precedence


# ==============================================================================
# Integration Test
# ==============================================================================


class TestApiKeySetupIntegration:
    """Integration tests for full API key setup workflow."""

    @patch("mcp_vector_search.cli.output.console")
    def test_full_workflow_no_key_to_saved(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test full workflow: no key -> save new key -> key exists."""
        # Arrange
        new_key = "sk-or-integration-test-12345678"
        mock_console.input.return_value = new_key

        # Act - First call saves the key
        result1 = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert - Key was saved
        assert result1 is True

        # Verify key was actually saved to config
        from mcp_vector_search.core.config_utils import get_openrouter_api_key

        config_dir = mock_project_root / ".mcp-vector-search"
        saved_key = get_openrouter_api_key(config_dir)
        assert saved_key == new_key

        # Act - Second call should detect existing key
        mock_console.input.return_value = ""  # Keep existing
        result2 = setup_openrouter_api_key(mock_project_root, interactive=True)

        # Assert - Key still exists
        assert result2 is True

    @patch("mcp_vector_search.cli.output.console")
    def test_full_workflow_save_then_clear(
        self, mock_console, mock_project_root, clean_env
    ):
        """Test full workflow: save key -> clear key -> no key."""
        # Arrange
        new_key = "sk-or-integration-test-87654321"
        mock_console.input.return_value = new_key

        # Act - Save key
        result1 = setup_openrouter_api_key(mock_project_root, interactive=True)
        assert result1 is True

        # Act - Clear key
        mock_console.input.return_value = "clear"
        result2 = setup_openrouter_api_key(mock_project_root, interactive=True)
        assert result2 is False

        # Verify key was actually deleted
        from mcp_vector_search.core.config_utils import get_openrouter_api_key

        config_dir = mock_project_root / ".mcp-vector-search"
        saved_key = get_openrouter_api_key(config_dir)
        assert saved_key is None

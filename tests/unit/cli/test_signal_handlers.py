"""Tests for CLI signal handlers."""

import signal
import sys
from io import StringIO
from unittest.mock import patch

import pytest


def test_segfault_handler_registered():
    """Test that SIGSEGV handler is registered.

    This test verifies that the signal handler is properly registered
    when the CLI module is imported. It checks that a handler exists
    but doesn't actually trigger a segfault.
    """
    from mcp_vector_search.cli.main import _handle_segfault

    # Verify handler function exists and is callable
    assert callable(_handle_segfault)

    # Verify SIGSEGV signal has a handler (not default)
    current_handler = signal.getsignal(signal.SIGSEGV)
    assert current_handler is not signal.SIG_DFL
    assert current_handler is not signal.SIG_IGN


def test_segfault_handler_message():
    """Test that segfault handler prints correct error message."""
    from mcp_vector_search.cli.main import _handle_segfault

    # Capture stderr
    stderr_capture = StringIO()

    with patch.object(sys, "stderr", stderr_capture):
        with pytest.raises(SystemExit) as exc_info:
            _handle_segfault(signal.SIGSEGV, None)

        # Check exit code is 139 (standard segfault exit code)
        assert exc_info.value.code == 139

    # Check error message content
    error_output = stderr_capture.getvalue()
    assert "Segmentation Fault Detected" in error_output
    assert "mcp-vector-search reset index --force" in error_output
    assert "mcp-vector-search index" in error_output
    assert "corrupted index data" in error_output


def test_segfault_handler_suggestions():
    """Test that segfault handler provides helpful suggestions."""
    from mcp_vector_search.cli.main import _handle_segfault

    stderr_capture = StringIO()

    with patch.object(sys, "stderr", stderr_capture):
        with pytest.raises(SystemExit):
            _handle_segfault(signal.SIGSEGV, None)

    error_output = stderr_capture.getvalue()

    # Check for actionable suggestions
    assert "pip install -U mcp-vector-search" in error_output
    assert "github.com/bobmatnyc/mcp-vector-search" in error_output
    assert "ChromaDB" in error_output or "sentence-transformers" in error_output


def test_faulthandler_enabled():
    """Test that faulthandler is enabled for crash diagnostics."""
    import faulthandler

    # Import CLI module to ensure signal handlers are registered
    import mcp_vector_search.cli.main  # noqa: F401

    # Verify faulthandler is enabled
    # Note: faulthandler.is_enabled() returns True if enabled
    assert faulthandler.is_enabled()

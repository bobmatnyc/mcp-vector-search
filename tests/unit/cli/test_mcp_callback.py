"""Tests for the MCP CLI callback."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from mcp_vector_search.cli.commands.mcp import mcp_callback


def install_stub_mcp_server(monkeypatch):
    """Install a lightweight MCP server module for callback control-flow tests."""
    package = types.ModuleType("mcp_vector_search.mcp")
    server = types.ModuleType("mcp_vector_search.mcp.server")

    def run_mcp_server(project_root):  # pragma: no cover - asyncio.run is mocked
        return None

    server.run_mcp_server = run_mcp_server
    monkeypatch.setitem(sys.modules, "mcp_vector_search.mcp", package)
    monkeypatch.setitem(sys.modules, "mcp_vector_search.mcp.server", server)


def test_mcp_callback_success_exits_zero(monkeypatch):
    """A clean MCP server shutdown should not be caught as a failure."""
    install_stub_mcp_server(monkeypatch)
    ctx = SimpleNamespace(obj={"project_root": Path.cwd()}, invoked_subcommand=None)

    with patch("asyncio.run", return_value=None):
        with pytest.raises(typer.Exit) as exc_info:
            mcp_callback(ctx)

    assert exc_info.value.exit_code == 0


def test_mcp_callback_server_error_exits_one(monkeypatch, capsys):
    """Unexpected MCP server errors should still be surfaced as failures."""
    install_stub_mcp_server(monkeypatch)
    ctx = SimpleNamespace(obj={"project_root": Path.cwd()}, invoked_subcommand=None)

    with patch("asyncio.run", side_effect=RuntimeError("boom")):
        with pytest.raises(typer.Exit) as exc_info:
            mcp_callback(ctx)

    assert exc_info.value.exit_code == 1
    assert "MCP server error: boom" in capsys.readouterr().err

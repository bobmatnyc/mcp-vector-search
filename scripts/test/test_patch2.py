import os
import tempfile
from unittest.mock import patch

from typer.testing import CliRunner

from mcp_vector_search.cli.main import app

runner = CliRunner()
with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    os.makedirs("test_project", exist_ok=True)
    os.chdir("test_project")

    # Patch at the module level where it's used
    with patch("mcp_vector_search.cli.commands.init.confirm_action", return_value=True):
        result = runner.invoke(app, ["init", "main", "--extensions", ".py"])

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")

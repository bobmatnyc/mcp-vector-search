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

    # Patch both confirmation prompts
    with patch("mcp_vector_search.cli.commands.init.confirm_action") as mock_confirm:
        # Return True for first call (initialization), False for second (auto-index)
        mock_confirm.side_effect = [True, False]

        result = runner.invoke(app, ["init", "main", "--extensions", ".py"])

    print(f"Init exit code: {result.exit_code}")
    if result.exit_code == 0:
        print("SUCCESS!")
    else:
        print(f"Failed with output: {result.output[-500:]}")

    if result.exception:
        print(f"Exception: {result.exception}")

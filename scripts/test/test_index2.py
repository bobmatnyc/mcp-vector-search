import os
import tempfile

from typer.testing import CliRunner

from mcp_vector_search.cli.main import app

runner = CliRunner()
with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    os.makedirs("test_project", exist_ok=True)
    os.chdir("test_project")

    # Initialize first
    result = runner.invoke(app, ["init", "main", "--extensions", ".py", "--force"])
    print(f"Init exit code: {result.exit_code}")

    # Try to index
    result = runner.invoke(app, ["index", "main"])

    print(f"Index exit code: {result.exit_code}")
    print(f"Index output: {result.output[:500]}")
    if result.exception:
        print(f"Exception: {result.exception}")

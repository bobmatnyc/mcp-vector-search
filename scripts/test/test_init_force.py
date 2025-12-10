import os
import tempfile

from typer.testing import CliRunner

from mcp_vector_search.cli.main import app

runner = CliRunner()
with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    os.makedirs("test_project", exist_ok=True)
    os.chdir("test_project")

    # Initialize with force
    result = runner.invoke(app, ["init", "main", "--extensions", ".py", "--force"])
    print(f"Init exit code: {result.exit_code}")
    print(f"Init output: {result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )

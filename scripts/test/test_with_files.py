import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mcp_vector_search.cli.main import app

runner = CliRunner()
with tempfile.TemporaryDirectory() as tmpdir:
    os.chdir(tmpdir)
    project_dir = Path("test_project")
    project_dir.mkdir()
    os.chdir(str(project_dir))

    # Create a sample Python file in current directory
    Path("main.py").write_text(
        """
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
"""
    )

    print(f"Created main.py: {Path('main.py').exists()}")
    print(f"Current dir: {Path.cwd()}")
    print(f"Files in dir: {list(Path.cwd().iterdir())}")

    # Initialize
    with patch(
        "mcp_vector_search.cli.commands.init.confirm_action", return_value=False
    ):
        result = runner.invoke(app, ["init", "main", "--extensions", ".py", "--force"])

    print(f"Init exit code: {result.exit_code}")

    # Index
    result = runner.invoke(app, ["index", "main"])

    print(f"Index exit code: {result.exit_code}")
    print("Index output (last 500 chars):")
    print(result.output[-500:])

    # Check if index directory exists
    index_dir = Path.cwd() / ".mcp-vector-search" / "index"
    print(f"\nIndex dir exists: {index_dir.exists()}")

    # List all files in .mcp-vector-search
    mcp_dir = Path.cwd() / ".mcp-vector-search"
    if mcp_dir.exists():
        print(f"Files in .mcp-vector-search: {list(mcp_dir.iterdir())}")

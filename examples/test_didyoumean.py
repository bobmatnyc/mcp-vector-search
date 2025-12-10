#!/usr/bin/env python3
"""Test script to demonstrate 'did you mean' functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from mcp_vector_search.cli.didyoumean import create_enhanced_typer


def main():
    """Test the did you mean functionality."""

    # Create a test app
    app = create_enhanced_typer(
        name="test-app", help="Test app for did you mean functionality"
    )

    @app.command()
    def search(query: str):
        """Search for something."""
        print(f"Searching for: {query}")

    @app.command()
    def index():
        """Index the data."""
        print("Indexing...")

    @app.command()
    def status():
        """Show status."""
        print("Status: OK")

    @app.command()
    def config():
        """Manage configuration."""
        print("Configuration")

    # Test with various typos
    test_commands = [
        ["serach", "hello"],  # Should suggest "search"
        ["indx"],  # Should suggest "index"
        ["stat"],  # Should suggest "status"
        ["conf"],  # Should suggest "config"
        ["xyz"],  # Should show available commands
    ]

    print("Testing 'did you mean' functionality:\n")

    for cmd_args in test_commands:
        print(f"Testing: {' '.join(cmd_args)}")
        try:
            # This would normally be called by the CLI
            app(cmd_args, standalone_mode=False)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)


if __name__ == "__main__":
    main()

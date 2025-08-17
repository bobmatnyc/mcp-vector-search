#!/usr/bin/env python3
"""UV-based development environment setup script."""

import subprocess
import sys
import shutil
from pathlib import Path


def run_uv_command(cmd: list[str], description: str) -> None:
    """Run a UV command with error handling."""
    print(f"🔧 {description}...")
    try:
        subprocess.run(["uv"] + cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ {description} completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        sys.exit(1)


def check_uv_installed() -> bool:
    """Check if UV is installed."""
    return shutil.which("uv") is not None


def main() -> None:
    """Set up development environment with UV."""
    print("🚀 Setting up mcp-vector-search development environment\n")

    # Check UV installation
    if not check_uv_installed():
        print("❌ UV not found. Install with:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    print("✅ UV found, proceeding with setup\n")

    # Sync dependencies (creates venv and installs everything)
    run_uv_command(
        ["sync", "--dev"],
        "Installing all dependencies with UV"
    )

    # Install pre-commit hooks
    run_uv_command(
        ["run", "pre-commit", "install"],
        "Setting up pre-commit hooks"
    )

    # Run initial code quality checks
    run_uv_command(
        ["run", "black", "--check", "src", "tests"],
        "Checking code formatting"
    )

    run_uv_command(
        ["run", "ruff", "check", "src", "tests"],
        "Running linter checks"
    )

    run_uv_command(
        ["run", "mypy", "src"],
        "Running type checking"
    )

    # Run tests
    run_uv_command(
        ["run", "pytest", "tests/", "-v"],
        "Running test suite"
    )

    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("  • Run 'uv run mcp-vector-search init' to initialize a test project")
    print("  • Run 'uv run pytest' to run tests")
    print("  • Use 'uv add <package>' to add new dependencies")
    print("  • All commands prefixed with 'uv run' use the project venv automatically")


if __name__ == "__main__":
    main()

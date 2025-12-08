#!/usr/bin/env python3
"""Comprehensive build script for MCP Vector Search."""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class BuildManager:
    """Manages the complete build process."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src" / "mcp_vector_search"

    def run_command(self, cmd: list[str], description: str, check: bool = True) -> bool:
        """Run a command with proper logging."""
        print(f"🔧 {description}...")

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, check=check, capture_output=True, text=True
            )

            if result.returncode == 0:
                print(f"  ✅ {description} completed")
                return True
            else:
                print(f"  ❌ {description} failed")
                if result.stderr:
                    print(f"     Error: {result.stderr}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"  ❌ {description} failed with exit code {e.returncode}")
            if e.stderr:
                print(f"     Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"  ❌ {description} failed: {e}")
            return False

    def check_tools(self) -> bool:
        """Check that required tools are available."""
        print("🔍 Checking required tools...")

        tools = [
            (["python3", "--version"], "Python"),
            (["uv", "--version"], "uv"),
            (["git", "--version"], "Git"),
        ]

        all_good = True
        for cmd, name in tools:
            if self.run_command(cmd, f"Checking {name}", check=False):
                continue
            else:
                print(f"  ❌ {name} not found or not working")
                all_good = False

        return all_good

    def setup_environment(self) -> bool:
        """Set up the development environment."""
        print("\n📦 Setting up environment...")

        # Install dependencies
        if not self.run_command(["uv", "sync", "--dev"], "Installing dependencies"):
            return False

        # Install pre-commit hooks
        if not self.run_command(
            ["uv", "run", "pre-commit", "install"], "Installing pre-commit hooks"
        ):
            print("  ⚠️  Pre-commit installation failed (continuing anyway)")

        return True

    def run_linting(self, fix: bool = False) -> bool:
        """Run linting checks."""
        print("\n🔍 Running linting...")

        success = True

        # Ruff format
        if fix:
            success &= self.run_command(
                ["uv", "run", "ruff", "format", str(self.src_dir)],
                "Running ruff format",
            )
        else:
            success &= self.run_command(
                ["uv", "run", "ruff", "format", "--check", str(self.src_dir)],
                "Checking ruff format",
            )

        # Ruff check
        ruff_cmd = ["uv", "run", "ruff", "check", str(self.src_dir)]
        if fix:
            ruff_cmd.append("--fix")

        success &= self.run_command(ruff_cmd, "Running ruff check")

        # MyPy
        success &= self.run_command(
            ["uv", "run", "mypy", str(self.src_dir), "--ignore-missing-imports"],
            "Running mypy",
        )

        return success

    def run_tests(self, coverage: bool = True) -> bool:
        """Run the test suite."""
        print("\n🧪 Running tests...")

        cmd = ["uv", "run", "pytest", "tests/", "-v"]

        if coverage:
            cmd.extend(
                [
                    "--cov=src/mcp_vector_search",
                    "--cov-report=term-missing",
                    "--cov-report=xml",
                ]
            )

        return self.run_command(cmd, "Running test suite")

    def run_security_checks(self) -> bool:
        """Run security checks."""
        print("\n🛡️  Running security checks...")

        success = True

        # Safety check
        success &= self.run_command(
            ["uv", "run", "safety", "check"],
            "Running safety check",
            check=False,  # Don't fail build on security warnings
        )

        # Bandit scan
        success &= self.run_command(
            ["uv", "run", "bandit", "-r", str(self.src_dir)],
            "Running bandit security scan",
            check=False,  # Don't fail build on security warnings
        )

        return success

    def build_package(self) -> bool:
        """Build the package."""
        print("\n📦 Building package...")

        # Clean previous builds
        if not self.run_command(
            ["rm", "-rf", "dist/", "build/", "*.egg-info"],
            "Cleaning previous builds",
            check=False,
        ):
            print("  ⚠️  Clean failed (continuing anyway)")

        # Build package
        if not self.run_command(["uv", "build"], "Building package"):
            return False

        # Check package
        return self.run_command(
            ["uv", "run", "twine", "check", "dist/*"], "Checking package"
        )

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n🔗 Running integration tests...")

        # Create temporary directory for testing
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.py"
            test_file.write_text("def test_function(): pass\n")

            # Install package
            dist_files = list((self.project_root / "dist").glob("*.whl"))
            if not dist_files:
                print("  ❌ No wheel file found")
                return False

            wheel_file = dist_files[0]

            # Test installation and basic functionality
            commands = [
                (["uv", "pip", "install", str(wheel_file)], "Installing package"),
                (["mcp-vector-search", "--version"], "Testing CLI availability"),
            ]

            # Change to temp directory for testing
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                for cmd, desc in commands:
                    if not self.run_command(cmd, desc):
                        return False

                # Test basic functionality
                init_cmd = [
                    "mcp-vector-search",
                    "init",
                    "--file-extensions",
                    ".py",
                    "--embedding-model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ]

                if not self.run_command(init_cmd, "Testing project initialization"):
                    return False

                if not self.run_command(
                    ["mcp-vector-search", "index"], "Testing indexing"
                ):
                    return False

                search_cmd = ["mcp-vector-search", "search", "function", "--limit", "5"]
                if not self.run_command(search_cmd, "Testing search"):
                    return False

            finally:
                os.chdir(original_cwd)

        return True

    def increment_version(self, version_type: str = "patch") -> bool:
        """Increment version."""
        print(f"\n📈 Incrementing {version_type} version...")

        return self.run_command(
            ["python3", "scripts/version_manager.py", "--bump", version_type],
            f"Bumping {version_type} version",
        )

    def increment_build(self) -> bool:
        """Increment build number."""
        print("\n🔢 Incrementing build number...")

        return self.run_command(
            ["python3", "scripts/version_manager.py", "--increment-build"],
            "Incrementing build number",
        )


def main():
    """Main build script."""
    parser = argparse.ArgumentParser(description="Comprehensive build script")
    parser.add_argument(
        "--skip-tools-check", action="store_true", help="Skip tools check"
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip environment setup"
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting")
    parser.add_argument("--skip-tests", action="store_true", help="Skip tests")
    parser.add_argument(
        "--skip-security", action="store_true", help="Skip security checks"
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip package build")
    parser.add_argument(
        "--skip-integration", action="store_true", help="Skip integration tests"
    )
    parser.add_argument("--fix-lint", action="store_true", help="Fix linting issues")
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage in tests"
    )
    parser.add_argument(
        "--version-bump", choices=["patch", "minor", "major"], help="Bump version"
    )
    parser.add_argument(
        "--build-increment", action="store_true", help="Increment build number"
    )

    args = parser.parse_args()

    project_root = Path.cwd()
    builder = BuildManager(project_root)

    print("🚀 MCP Vector Search - Comprehensive Build")
    print("=" * 50)

    start_time = time.time()
    success = True

    # Check tools
    if not args.skip_tools_check:
        success &= builder.check_tools()

    # Setup environment
    if not args.skip_setup and success:
        success &= builder.setup_environment()

    # Version/build increment
    if args.version_bump and success:
        success &= builder.increment_version(args.version_bump)

    if args.build_increment and success:
        success &= builder.increment_build()

    # Linting
    if not args.skip_lint and success:
        success &= builder.run_linting(fix=args.fix_lint)

    # Tests
    if not args.skip_tests and success:
        success &= builder.run_tests(coverage=not args.no_coverage)

    # Security
    if not args.skip_security and success:
        builder.run_security_checks()  # Don't fail build on security warnings

    # Build
    if not args.skip_build and success:
        success &= builder.build_package()

    # Integration tests
    if not args.skip_integration and success:
        success &= builder.run_integration_tests()

    # Summary
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 50)
    if success:
        print(f"🎉 Build completed successfully in {duration:.1f}s")
        sys.exit(0)
    else:
        print(f"❌ Build failed after {duration:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Development build script with automatic version incrementing."""

import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Literal

# ============================================================================
# DEPRECATION WARNING
# ============================================================================
warnings.warn(
    "\n" + "=" * 70 + "\nDEPRECATION WARNING: This script is deprecated as of v4.0.3\n"
    "Please use the Makefile or scripts/version_manager.py instead:\n"
    "  make build-package        # Build distribution packages\n"
    "  make version-patch        # Bump patch version\n"
    "  make release-patch        # Full release workflow\n"
    "  python scripts/version_manager.py --help  # Version management\n\n"
    "This script will be removed in v5.0.0\n" + "=" * 70 + "\n",
    DeprecationWarning,
    stacklevel=2,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VERSION_FILE = PROJECT_ROOT / "src" / "mcp_vector_search" / "__init__.py"


def get_current_version() -> str:
    """Get the current version from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__ = "([^"]*)"', content)
    if not match:
        raise ValueError("Could not find version in __init__.py")
    return match.group(1)


def get_current_build() -> str:
    """Get the current build number from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__build__ = "([^"]*)"', content)
    if not match:
        raise ValueError("Could not find build number in __init__.py")
    return match.group(1)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse version string into major, minor, patch tuple."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    return tuple(int(part) for part in parts)


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple into string."""
    return f"{major}.{minor}.{patch}"


def increment_version(
    version: str, part: Literal["major", "minor", "patch"] = "patch"
) -> str:
    """Increment the specified part of the version."""
    major, minor, patch = parse_version(version)

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid version part: {part}")

    return format_version(major, minor, patch)


def increment_build() -> str:
    """Increment the build number."""
    current_build = get_current_build()
    try:
        build_num = int(current_build)
        return str(build_num + 1)
    except ValueError:
        # If build is not a number, start from 1
        return "1"


def update_version_file(new_version: str, new_build: str) -> None:
    """Update the version and build number in __init__.py."""
    content = VERSION_FILE.read_text()

    # Update version
    new_content = re.sub(
        r'__version__ = "[^"]*"', f'__version__ = "{new_version}"', content
    )

    # Update build number
    new_content = re.sub(
        r'__build__ = "[^"]*"', f'__build__ = "{new_build}"', new_content
    )

    VERSION_FILE.write_text(new_content)


def run_command(cmd: list[str], description: str) -> None:
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.stdout.strip():
            print(f"   {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   {e.stderr.strip()}")
        sys.exit(1)


def main():
    """Main development build function."""
    # Show runtime deprecation warning
    import warnings

    warnings.warn(
        "\n"
        + "=" * 70
        + "\nDEPRECATION WARNING: This script is deprecated as of v4.0.3\n"
        "Please use the Makefile or scripts/version_manager.py instead:\n"
        "  make build-package        # Build distribution packages\n"
        "  make version-patch        # Bump patch version\n"
        "  make release-patch        # Full release workflow\n"
        "  python scripts/version_manager.py --help  # Version management\n\n"
        "This script will be removed in v5.0.0\n" + "=" * 70 + "\n",
        DeprecationWarning,
        stacklevel=2,
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Development build with automatic version increment"
    )
    parser.add_argument(
        "--increment",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Version part to increment (default: patch)",
    )
    parser.add_argument(
        "--no-increment",
        action="store_true",
        help="Skip version increment, just rebuild",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Get current version and build
    current_version = get_current_version()
    current_build = get_current_build()
    print(f"📦 Current version: {current_version} (build {current_build})")

    # Always increment build number
    new_build = increment_build()

    if not args.no_increment:
        # Calculate new version
        new_version = increment_version(current_version, args.increment)
        print(f"🆙 New version: {new_version} (build {new_build})")

        if args.dry_run:
            print("🔍 Dry run - no changes made")
            return

        # Update version and build file
        print("📝 Updating version and build...")
        update_version_file(new_version, new_build)
        print(f"   Updated {VERSION_FILE.relative_to(PROJECT_ROOT)}")
    else:
        new_version = current_version
        print(f"⏭️  Skipping version increment, new build: {new_build}")

        if args.dry_run:
            print("🔍 Dry run - no changes made")
            return

        # Update only build number
        print("📝 Updating build number...")
        update_version_file(new_version, new_build)
        print(f"   Updated {VERSION_FILE.relative_to(PROJECT_ROOT)}")

    # Rebuild package
    run_command(["uv", "pip", "install", "-e", "."], "Rebuilding package")

    # Verify installation
    run_command(["uv", "run", "mcp-vector-search", "version"], "Verifying installation")

    print(f"✅ Build complete! Version: {new_version} (build {new_build})")


if __name__ == "__main__":
    main()

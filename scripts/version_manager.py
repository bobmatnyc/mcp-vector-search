#!/usr/bin/env python3
"""
Version Manager for MCP Vector Search

Centralized version management system that handles:
- Semantic versioning (major.minor.patch)
- Build number tracking
- Changelog updates
- Git operations (tagging, committing)
- Dry-run mode for safe testing
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Color codes for terminal output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


class VersionManager:
    """Manages version and build numbers for the project."""

    def __init__(self, project_root: Path | None = None):
        """Initialize version manager.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.init_file = self.project_root / "src" / "mcp_vector_search" / "__init__.py"
        self.changelog_file = self.project_root / "docs" / "CHANGELOG.md"
        self.pyproject_file = self.project_root / "pyproject.toml"

        # Ensure required files exist
        if not self.init_file.exists():
            raise FileNotFoundError(f"__init__.py not found at {self.init_file}")

    def read_version(self) -> tuple[str, int]:
        """Read current version and build number from __init__.py.

        Returns:
            Tuple of (version_string, build_number)
        """
        content = self.init_file.read_text()

        # Extract version
        version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if not version_match:
            raise ValueError("Could not find __version__ in __init__.py")
        version = version_match.group(1)

        # Extract build
        build_match = re.search(r'__build__\s*=\s*["\'](\d+)["\']', content)
        if not build_match:
            raise ValueError("Could not find __build__ in __init__.py")
        build = int(build_match.group(1))

        return version, build

    def write_version(self, version: str, build: int, dry_run: bool = False) -> None:
        """Write version and build number to __init__.py.

        Args:
            version: Version string (e.g., "4.0.3")
            build: Build number
            dry_run: If True, only show what would be done
        """
        if dry_run:
            print(f"{YELLOW}[DRY RUN]{RESET} Would update __init__.py:")
            print(f"  Version: {version}")
            print(f"  Build: {build}")
            return

        content = self.init_file.read_text()

        # Replace version
        content = re.sub(
            r'__version__\s*=\s*["\'][^"\']+["\']',
            f'__version__ = "{version}"',
            content,
        )

        # Replace build
        content = re.sub(
            r'__build__\s*=\s*["\'][^"\']+["\']', f'__build__ = "{build}"', content
        )

        self.init_file.write_text(content)
        print(
            f"{GREEN}✓{RESET} Updated __init__.py with version {version} build {build}"
        )

    def bump_version(self, current: str, bump_type: str) -> str:
        """Bump version according to semantic versioning.

        Args:
            current: Current version string
            bump_type: Type of bump (major, minor, patch)

        Returns:
            New version string
        """
        parts = current.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {current}")

        major, minor, patch = map(int, parts)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

        return f"{major}.{minor}.{patch}"

    def update_changelog(self, version: str, dry_run: bool = False) -> None:
        """Add new version section to CHANGELOG.md.

        Args:
            version: Version string for the new release
            dry_run: If True, only show what would be done
        """
        if not self.changelog_file.exists():
            print(f"{YELLOW}Warning:{RESET} CHANGELOG.md not found, skipping update")
            return

        if dry_run:
            print(
                f"{YELLOW}[DRY RUN]{RESET} Would add version {version} to CHANGELOG.md"
            )
            return

        content = self.changelog_file.read_text()

        # Check if version already exists
        if f"## [{version}]" in content or f"## {version}" in content:
            print(f"{YELLOW}Version {version} already in CHANGELOG.md{RESET}")
            return

        # Find the position to insert (after the header, before first version)
        lines = content.split("\n")
        insert_pos = 0

        for i, line in enumerate(lines):
            if line.startswith("## [") or line.startswith("## v"):
                insert_pos = i
                break
            elif line.startswith("# Changelog") or line.startswith("# CHANGELOG"):
                # Insert after the header and any blank lines
                insert_pos = i + 1
                while insert_pos < len(lines) and not lines[insert_pos].strip():
                    insert_pos += 1

        # Create new version entry
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_entry = [
            f"## [{version}] - {date_str}",
            "",
            "### Added",
            "- ",
            "",
            "### Changed",
            "- ",
            "",
            "### Fixed",
            "- ",
            "",
        ]

        # Insert the new entry
        for entry_line in reversed(new_entry):
            lines.insert(insert_pos, entry_line)

        # Write back
        self.changelog_file.write_text("\n".join(lines))
        print(f"{GREEN}✓{RESET} Added version {version} to CHANGELOG.md")

    def git_operations(self, version: str, dry_run: bool = False) -> None:
        """Handle git commit, tag, and push operations.

        Args:
            version: Version string for tagging
            dry_run: If True, only show what would be done
        """
        if dry_run:
            print(f"{YELLOW}[DRY RUN]{RESET} Would perform git operations:")
            print("  - Add all changes")
            print(f"  - Commit with message: 'Release v{version}'")
            print(f"  - Create tag: v{version}")
            return

        try:
            # Add all changes
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            # Commit
            commit_message = f"🚀 Release v{version}"
            subprocess.run(
                ["git", "commit", "-m", commit_message], check=True, capture_output=True
            )
            print(f"{GREEN}✓{RESET} Created commit: {commit_message}")

            # Tag
            tag_name = f"v{version}"
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Release version {version}"],
                check=True,
                capture_output=True,
            )
            print(f"{GREEN}✓{RESET} Created tag: {tag_name}")

        except subprocess.CalledProcessError as e:
            print(f"{RED}✗{RESET} Git operation failed: {e}")
            if e.stderr:
                print(f"  {e.stderr.decode()}")
            sys.exit(1)

    def show_version(self, format_type: str = "simple") -> None:
        """Display current version information.

        Args:
            format_type: Output format (simple, detailed, json)
        """
        version, build = self.read_version()

        if format_type == "simple":
            print(version)
        elif format_type == "detailed":
            print(f"{BLUE}MCP Vector Search Version Information{RESET}")
            print(f"  Version: {GREEN}{version}{RESET}")
            print(f"  Build:   {GREEN}{build}{RESET}")
            print("  Package: mcp-vector-search")

            # Check git status
            try:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--always"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    git_desc = result.stdout.strip()
                    print(f"  Git:     {git_desc}")
            except Exception:  # noqa: S110
                pass

        elif format_type == "json":
            data = {"version": version, "build": build, "package": "mcp-vector-search"}
            print(json.dumps(data, indent=2))
        else:
            print(f"{version} (build {build})")


def main():
    """Main entry point for the version manager CLI."""
    parser = argparse.ArgumentParser(
        description="Version manager for MCP Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --show                    # Show current version
  %(prog)s --bump patch              # Bump patch version
  %(prog)s --bump minor --dry-run    # Test minor version bump
  %(prog)s --set 4.0.3 --build 280   # Set specific version
  %(prog)s --increment-build         # Increment build number only
        """,
    )

    # Version operations
    parser.add_argument("--show", action="store_true", help="Show current version")
    parser.add_argument(
        "--format",
        choices=["simple", "detailed", "json"],
        default="detailed",
        help="Output format for --show",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version (major, minor, or patch)",
    )
    parser.add_argument(
        "--set", metavar="VERSION", help="Set specific version (e.g., 4.0.3)"
    )
    parser.add_argument(
        "--build",
        type=int,
        metavar="NUMBER",
        help="Set specific build number (use with --set)",
    )
    parser.add_argument(
        "--increment-build", action="store_true", help="Increment build number only"
    )

    # Additional operations
    parser.add_argument(
        "--update-changelog",
        action="store_true",
        help="Update CHANGELOG.md with new version",
    )
    parser.add_argument(
        "--git-commit", action="store_true", help="Create git commit and tag"
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Create version manager
    try:
        vm = VersionManager()
    except FileNotFoundError as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)

    # Handle operations
    try:
        if args.show:
            vm.show_version(args.format)

        elif args.bump:
            current_version, current_build = vm.read_version()
            new_version = vm.bump_version(current_version, args.bump)
            new_build = current_build + 1

            print(f"{BLUE}Version bump:{RESET} {current_version} → {new_version}")
            vm.write_version(new_version, new_build, args.dry_run)

            if args.update_changelog:
                vm.update_changelog(new_version, args.dry_run)

            if args.git_commit:
                vm.git_operations(new_version, args.dry_run)

        elif args.set:
            if args.build is None:
                current_version, current_build = vm.read_version()
                new_build = current_build
            else:
                new_build = args.build

            vm.write_version(args.set, new_build, args.dry_run)

        elif args.increment_build:
            current_version, current_build = vm.read_version()
            new_build = current_build + 1

            print(f"{BLUE}Build increment:{RESET} {current_build} → {new_build}")
            vm.write_version(current_version, new_build, args.dry_run)

        elif args.update_changelog:
            current_version, _ = vm.read_version()
            vm.update_changelog(current_version, args.dry_run)

        elif args.git_commit:
            current_version, _ = vm.read_version()
            vm.git_operations(current_version, args.dry_run)

        else:
            # Default: show version
            vm.show_version(args.format)

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

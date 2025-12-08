#!/usr/bin/env python3
"""
Documentation Update Script for MCP Vector Search

Automatically updates version references in documentation files:
- README.md: Alpha Release version badge
- CLAUDE.md: Recent Activity section version references
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Color codes for terminal output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


class DocumentationUpdater:
    """Updates documentation files with version information."""

    def __init__(self, project_root: Path | None = None):
        """Initialize documentation updater.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.readme_file = self.project_root / "README.md"
        self.claude_file = self.project_root / "CLAUDE.md"

    def update_readme_version(self, version: str, dry_run: bool = False) -> bool:
        """Update README.md version badge.

        Updates line 9: > ⚠️ **Alpha Release (vX.Y.Z)**

        Args:
            version: New version string (e.g., "0.7.1")
            dry_run: If True, show what would change without modifying

        Returns:
            True if update was successful, False otherwise
        """
        if not self.readme_file.exists():
            print(f"{RED}✗{RESET} README.md not found at {self.readme_file}")
            return False

        content = self.readme_file.read_text()
        lines = content.split("\n")

        # Find and update the alpha release line (should be around line 9)
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith("> ⚠️") and "Alpha Release" in line:
                old_line = line
                # Update version in the line
                new_line = re.sub(r"\(v[\d.]+\)", f"(v{version})", line)

                if old_line != new_line:
                    if dry_run:
                        print(
                            f"{YELLOW}[DRY RUN]{RESET} Would update README.md line {i + 1}:"
                        )
                        print(f"  {BLUE}Old:{RESET} {old_line}")
                        print(f"  {GREEN}New:{RESET} {new_line}")
                    else:
                        lines[i] = new_line
                        print(
                            f"{GREEN}✓{RESET} Updated README.md version badge to v{version}"
                        )
                    updated = True
                else:
                    print(f"{BLUE}ℹ{RESET} README.md already at v{version}")
                    updated = True  # Already at correct version is considered success
                break

        if not updated:
            print(f"{YELLOW}Warning:{RESET} Alpha Release line not found in README.md")
            # Don't fail if line not found - it's not critical
            return True

        # Write back if not dry run
        if not dry_run and updated:
            self.readme_file.write_text("\n".join(lines))

        return True

    def update_claude_recent_activity(
        self, version: str, release_type: str = "patch", dry_run: bool = False
    ) -> bool:
        """Update CLAUDE.md Recent Activity section.

        Updates the Recent Releases section with new version information.

        Args:
            version: New version string (e.g., "0.7.1")
            release_type: Type of release (patch, minor, major)
            dry_run: If True, show what would change without modifying

        Returns:
            True if update was successful, False otherwise
        """
        if not self.claude_file.exists():
            print(f"{YELLOW}Warning:{RESET} CLAUDE.md not found at {self.claude_file}")
            return False

        content = self.claude_file.read_text()

        # Check if version already exists in Recent Releases
        if f"**v{version}" in content:
            print(f"{BLUE}ℹ{RESET} CLAUDE.md already references v{version}")
            return True

        # Find the Recent Activity section
        lines = content.split("\n")
        recent_releases_idx = -1

        for i, line in enumerate(lines):
            if "## 📊 Recent Activity" in line:
                pass
            elif "### 🔴 Recent Releases" in line:
                recent_releases_idx = i
                break

        if recent_releases_idx == -1:
            print(
                f"{YELLOW}Warning:{RESET} Recent Releases section not found in CLAUDE.md"
            )
            return False

        # Determine release description based on type
        datetime.now().strftime("%b %d, %Y")
        month_str = datetime.now().strftime(
            "%b %#d" if sys.platform == "win32" else "%b %-d"
        )

        release_descriptions = {
            "patch": "Bug Fixes & Improvements",
            "minor": "New Features & Enhancements",
            "major": "Major Release",
        }
        description = release_descriptions.get(release_type, "Release")

        # Create new release entry
        new_entry = (
            f"**v{version} ({month_str}, {datetime.now().year})** - {description}"
        )

        # Insert after the Recent Releases header
        insert_idx = recent_releases_idx + 1

        if dry_run:
            print(f"{YELLOW}[DRY RUN]{RESET} Would add to CLAUDE.md Recent Releases:")
            print(f"  {GREEN}+ {new_entry}{RESET}")
        else:
            lines.insert(insert_idx, new_entry)
            self.claude_file.write_text("\n".join(lines))
            print(f"{GREEN}✓{RESET} Updated CLAUDE.md Recent Activity with v{version}")

        return True

    def update_last_updated(self, dry_run: bool = False) -> bool:
        """Update the 'Last Updated' timestamp in CLAUDE.md.

        Args:
            dry_run: If True, show what would change without modifying

        Returns:
            True if update was successful, False otherwise
        """
        if not self.claude_file.exists():
            return False

        content = self.claude_file.read_text()
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Find and update Last Updated line
        pattern = r"\*\*Last Updated\*\*:\s*\d{4}-\d{2}-\d{2}"
        new_text = f"**Last Updated**: {current_date}"

        if re.search(pattern, content):
            if dry_run:
                print(
                    f"{YELLOW}[DRY RUN]{RESET} Would update CLAUDE.md Last Updated to {current_date}"
                )
            else:
                updated_content = re.sub(pattern, new_text, content)
                self.claude_file.write_text(updated_content)
                print(
                    f"{GREEN}✓{RESET} Updated CLAUDE.md Last Updated to {current_date}"
                )
            return True

        return False

    def update_all(
        self, version: str, release_type: str = "patch", dry_run: bool = False
    ) -> bool:
        """Update all documentation files.

        Args:
            version: New version string
            release_type: Type of release (patch, minor, major)
            dry_run: If True, show what would change without modifying

        Returns:
            True if all updates successful, False otherwise
        """
        success = True

        print(
            f"{BLUE}Updating documentation for v{version} ({release_type} release)...{RESET}\n"
        )

        # Update README.md
        if not self.update_readme_version(version, dry_run):
            success = False

        # Update CLAUDE.md only for minor/major releases or significant patches
        if release_type in ["minor", "major"]:
            if not self.update_claude_recent_activity(version, release_type, dry_run):
                success = False
            if not self.update_last_updated(dry_run):
                success = False

        if success and not dry_run:
            print(f"\n{GREEN}✓ All documentation updated successfully!{RESET}")
        elif dry_run:
            print(f"\n{YELLOW}[DRY RUN] No files were modified{RESET}")

        return success


def main():
    """Main entry point for documentation updater CLI."""
    parser = argparse.ArgumentParser(
        description="Update documentation files with version information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --version 0.7.2
  %(prog)s --version 0.8.0 --type minor
  %(prog)s --version 1.0.0 --type major --dry-run
        """,
    )

    parser.add_argument(
        "--version", "-v", required=True, help="Version to update to (e.g., 0.7.2)"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["patch", "minor", "major"],
        default="patch",
        help="Type of release (default: patch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--readme-only", action="store_true", help="Only update README.md"
    )
    parser.add_argument(
        "--claude-only", action="store_true", help="Only update CLAUDE.md"
    )

    args = parser.parse_args()

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        print(f"{RED}Error:{RESET} Invalid version format. Use X.Y.Z (e.g., 0.7.2)")
        sys.exit(1)

    # Create updater
    try:
        updater = DocumentationUpdater()
    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)

    # Perform updates
    try:
        success = True

        if args.readme_only:
            success = updater.update_readme_version(args.version, args.dry_run)
        elif args.claude_only:
            success = updater.update_claude_recent_activity(
                args.version, args.type, args.dry_run
            ) and updater.update_last_updated(args.dry_run)
        else:
            success = updater.update_all(args.version, args.type, args.dry_run)

        if not success:
            sys.exit(1)

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

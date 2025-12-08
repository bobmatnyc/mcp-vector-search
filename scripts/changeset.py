#!/usr/bin/env python3
"""
Changeset Management for MCP Vector Search

Manages changesets for structured release notes and changelog generation.
Inspired by changesets/changesets but simplified for Python projects.
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


class Changeset:
    """Represents a single changeset."""

    def __init__(self, filepath: Path):
        """Initialize from a changeset file.

        Args:
            filepath: Path to the changeset file
        """
        self.filepath = filepath
        self.type: str = ""
        self.summary: str = ""
        self.details: list[str] = []
        self.impact: list[str] = []
        self.breaking_changes: list[str] = []
        self.related: list[str] = []

        self._parse()

    def _parse(self) -> None:
        """Parse the changeset file."""
        content = self.filepath.read_text()

        # Split frontmatter and content
        parts = content.split("---")
        if len(parts) < 3:
            raise ValueError(f"Invalid changeset format in {self.filepath}")

        # Parse YAML frontmatter (simple parsing - just extract type)
        try:
            frontmatter = parts[1].strip()
            type_match = re.search(r"type:\s*(\w+)", frontmatter)
            if type_match:
                self.type = type_match.group(1).strip().lower()
            else:
                raise ValueError("No type field found in frontmatter")
        except Exception as e:
            raise ValueError(f"Invalid frontmatter in {self.filepath}: {e}")

        if self.type not in ["patch", "minor", "major"]:
            raise ValueError(f"Invalid type '{self.type}' in {self.filepath}")

        # Parse markdown content
        markdown = parts[2].strip()

        # Extract sections
        self._extract_section(markdown, "## Summary", self._set_summary)
        self._extract_section(markdown, "## Details", self._set_details)
        self._extract_section(markdown, "## Impact", self._set_impact)
        self._extract_section(
            markdown, "## Breaking Changes", self._set_breaking_changes
        )
        self._extract_section(markdown, "## Related", self._set_related)

    def _extract_section(self, content: str, header: str, setter) -> None:
        """Extract a section from markdown content."""
        pattern = rf"{re.escape(header)}\n(.*?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            section_content = match.group(1).strip()
            # Remove HTML comments
            section_content = re.sub(
                r"<!--.*?-->", "", section_content, flags=re.DOTALL
            )
            section_content = section_content.strip()
            if section_content:
                setter(section_content)

    def _set_summary(self, content: str) -> None:
        """Set summary from content."""
        self.summary = content.split("\n")[0].strip()

    def _set_details(self, content: str) -> None:
        """Set details from content."""
        self.details = [
            line.strip().lstrip("-").strip()
            for line in content.split("\n")
            if line.strip() and line.strip().startswith("-")
        ]

    def _set_impact(self, content: str) -> None:
        """Set impact from content."""
        lines = content.split("\n")
        # First line might be description, rest are bullet points
        for line in lines:
            line = line.strip()
            if line.startswith("-"):
                self.impact.append(line.lstrip("-").strip())

    def _set_breaking_changes(self, content: str) -> None:
        """Set breaking changes from content."""
        self.breaking_changes = [
            line.strip().lstrip("-").strip()
            for line in content.split("\n")
            if line.strip() and line.strip().startswith("-")
        ]

    def _set_related(self, content: str) -> None:
        """Set related items from content."""
        self.related = [
            line.strip().lstrip("-").strip()
            for line in content.split("\n")
            if line.strip() and line.strip().startswith("-")
        ]


class ChangesetManager:
    """Manages changesets for a project."""

    def __init__(self, project_root: Path | None = None):
        """Initialize changeset manager.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.changesets_dir = self.project_root / ".changesets"
        self.template_file = self.changesets_dir / "template.md"
        self.changelog_file = self.project_root / "docs" / "CHANGELOG.md"

        # Ensure changesets directory exists
        self.changesets_dir.mkdir(exist_ok=True)

    def add(self, change_type: str, description: str) -> Path:
        """Add a new changeset.

        Args:
            change_type: Type of change (patch, minor, major)
            description: Short description of the change

        Returns:
            Path to the created changeset file
        """
        # Validate type
        if change_type not in ["patch", "minor", "major"]:
            raise ValueError(f"Invalid change type: {change_type}")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", description.lower())
        slug = slug.strip("-")[:50]  # Limit slug length
        filename = f"{timestamp}-{slug}.md"
        filepath = self.changesets_dir / filename

        # Create changeset from template
        if self.template_file.exists():
            content = self.template_file.read_text()
        else:
            content = self._default_template()

        # Update type in frontmatter
        content = re.sub(r"type:\s*\w+", f"type: {change_type}", content)

        # Update summary
        content = re.sub(r"(## Summary\n).*", f"\\1{description}", content)

        # Write changeset file
        filepath.write_text(content)

        print(f"{GREEN}✓{RESET} Created changeset: {filename}")
        print(f"  Type: {change_type}")
        print(f"  Description: {description}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"  1. Edit {filepath} to add details")
        print("  2. Run 'make changeset-view' to see all changesets")
        print("  3. Changes will be consumed during next release")

        return filepath

    def _default_template(self) -> str:
        """Return default changeset template."""
        return """---
type: patch
---

## Summary
Brief description of the change

## Details
- Change 1
- Change 2

## Impact
- User-facing impact

## Breaking Changes
<!-- Delete if none -->

## Related
<!-- Optional -->
"""

    def list_changesets(self) -> list[Changeset]:
        """List all pending changesets.

        Returns:
            List of Changeset objects
        """
        changeset_files = sorted(
            [
                f
                for f in self.changesets_dir.glob("*.md")
                if f.name not in ["README.md", "template.md"]
            ]
        )

        changesets = []
        for filepath in changeset_files:
            try:
                changeset = Changeset(filepath)
                changesets.append(changeset)
            except Exception as e:
                print(
                    f"{YELLOW}Warning:{RESET} Skipping invalid changeset {filepath.name}: {e}"
                )

        return changesets

    def display_changesets(self) -> None:
        """Display all pending changesets."""
        changesets = self.list_changesets()

        if not changesets:
            print(f"{YELLOW}No pending changesets found.{RESET}")
            print(f"\n{BLUE}Add a changeset:{RESET}")
            print('  make changeset-add TYPE=patch DESC="your change description"')
            return

        print(f"{BLUE}Pending Changesets ({len(changesets)}):{RESET}\n")

        # Group by type
        by_type: dict[str, list[Changeset]] = {"major": [], "minor": [], "patch": []}

        for cs in changesets:
            by_type[cs.type].append(cs)

        # Display by type
        for change_type in ["major", "minor", "patch"]:
            items = by_type[change_type]
            if items:
                color = (
                    RED
                    if change_type == "major"
                    else YELLOW
                    if change_type == "minor"
                    else GREEN
                )
                print(f"{color}[{change_type.upper()}]{RESET} ({len(items)} changes)")
                for cs in items:
                    print(f"  • {cs.summary}")
                    print(f"    File: {cs.filepath.name}")
                print()

        print(f"{BLUE}Next steps:{RESET}")
        print("  • Review changesets: ls -la .changesets/")
        print("  • Consume changesets: make release-patch/minor/major")

    def consume(self, version: str, dry_run: bool = False) -> dict[str, list[str]]:
        """Consume changesets and update changelog.

        Args:
            version: Version string for the release
            dry_run: If True, don't modify files

        Returns:
            Dictionary of changes grouped by category
        """
        changesets = self.list_changesets()

        if not changesets:
            print(f"{YELLOW}No changesets to consume.{RESET}")
            return {}

        # Group changes by category
        changes: dict[str, list[str]] = {
            "Added": [],
            "Changed": [],
            "Fixed": [],
            "Breaking": [],
        }

        # Process each changeset
        for cs in changesets:
            # Categorize based on type and content
            if cs.breaking_changes:
                changes["Breaking"].extend(cs.breaking_changes)

            # Add summary and details
            if cs.summary:
                entry = f"**{cs.summary}**"
                if cs.details:
                    entry += "\n  " + "\n  ".join(f"- {d}" for d in cs.details)

                # Categorize by keywords or type
                summary_lower = cs.summary.lower()
                if any(
                    kw in summary_lower for kw in ["add", "new", "introduce", "feat"]
                ):
                    changes["Added"].append(entry)
                elif any(
                    kw in summary_lower for kw in ["fix", "resolve", "correct", "bug"]
                ):
                    changes["Fixed"].append(entry)
                else:
                    changes["Changed"].append(entry)

        # Update changelog
        if not dry_run:
            self._update_changelog(version, changes)

            # Delete consumed changesets
            for cs in changesets:
                cs.filepath.unlink()
                print(f"{GREEN}✓{RESET} Consumed: {cs.filepath.name}")
        else:
            print(
                f"{YELLOW}[DRY RUN]{RESET} Would consume {len(changesets)} changesets"
            )
            self._preview_changelog(version, changes)

        return changes

    def _update_changelog(self, version: str, changes: dict[str, list[str]]) -> None:
        """Update CHANGELOG.md with changes.

        Args:
            version: Version string
            changes: Dictionary of changes by category
        """
        if not self.changelog_file.exists():
            print(f"{YELLOW}Warning:{RESET} CHANGELOG.md not found, skipping update")
            return

        content = self.changelog_file.read_text()

        # Check if version already exists
        if f"## [{version}]" in content or f"## {version}" in content:
            print(f"{YELLOW}Version {version} already in CHANGELOG.md{RESET}")
            return

        # Build new entry
        date_str = datetime.now().strftime("%Y-%m-%d")
        entry_lines = [f"## [{version}] - {date_str}", ""]

        # Add sections with content
        for category in ["Breaking", "Added", "Changed", "Fixed"]:
            if changes.get(category):
                entry_lines.append(
                    f"### {category if category != 'Breaking' else 'Breaking Changes'}"
                )
                for item in changes[category]:
                    # Handle multi-line entries
                    for line in item.split("\n"):
                        entry_lines.append(
                            line if line.startswith("  ") else f"- {line}"
                        )
                entry_lines.append("")

        # Find insertion point (after ## [Unreleased])
        lines = content.split("\n")
        insert_pos = 0

        for i, line in enumerate(lines):
            if line.startswith("## [Unreleased]"):
                # Skip to after the Unreleased section
                insert_pos = i + 1
                while insert_pos < len(lines) and not lines[insert_pos].startswith(
                    "## ["
                ):
                    insert_pos += 1
                break
            elif line.startswith("## [") or line.startswith("## v"):
                insert_pos = i
                break

        # Insert new entry
        for line in reversed(entry_lines):
            lines.insert(insert_pos, line)

        # Write back
        self.changelog_file.write_text("\n".join(lines))
        print(f"{GREEN}✓{RESET} Updated CHANGELOG.md with version {version}")

    def _preview_changelog(self, version: str, changes: dict[str, list[str]]) -> None:
        """Preview changelog entry."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{BLUE}Changelog Entry Preview:{RESET}\n")
        print(f"## [{version}] - {date_str}\n")

        for category in ["Breaking", "Added", "Changed", "Fixed"]:
            if changes.get(category):
                print(
                    f"### {category if category != 'Breaking' else 'Breaking Changes'}"
                )
                for item in changes[category]:
                    for line in item.split("\n"):
                        print(line if line.startswith("  ") else f"- {line}")
                print()

    def validate(self) -> bool:
        """Validate all changeset files.

        Returns:
            True if all valid, False otherwise
        """
        changesets = self.changesets_dir.glob("*.md")
        changesets = [
            f for f in changesets if f.name not in ["README.md", "template.md"]
        ]

        if not changesets:
            print(f"{YELLOW}No changesets to validate.{RESET}")
            return True

        all_valid = True
        for filepath in changesets:
            try:
                cs = Changeset(filepath)
                print(f"{GREEN}✓{RESET} {filepath.name}: valid ({cs.type})")
            except Exception as e:
                print(f"{RED}✗{RESET} {filepath.name}: {e}")
                all_valid = False

        return all_valid


def main():
    """Main entry point for changeset CLI."""
    parser = argparse.ArgumentParser(
        description="Changeset management for MCP Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add --type patch --description "fix: resolve search bug"
  %(prog)s list
  %(prog)s consume --version 0.7.2
  %(prog)s validate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new changeset")
    add_parser.add_argument(
        "--type",
        choices=["patch", "minor", "major"],
        required=True,
        help="Type of change",
    )
    add_parser.add_argument(
        "--description", "-d", required=True, help="Short description of the change"
    )

    # List command
    subparsers.add_parser("list", help="List pending changesets")

    # Consume command
    consume_parser = subparsers.add_parser("consume", help="Consume changesets")
    consume_parser.add_argument(
        "--version", "-v", required=True, help="Version to consume changesets for"
    )
    consume_parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )

    # Validate command
    subparsers.add_parser("validate", help="Validate changeset files")

    args = parser.parse_args()

    # Create manager
    try:
        manager = ChangesetManager()
    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)

    # Handle commands
    try:
        if args.command == "add":
            manager.add(args.type, args.description)

        elif args.command == "list":
            manager.display_changesets()

        elif args.command == "consume":
            manager.consume(args.version, args.dry_run)

        elif args.command == "validate":
            if manager.validate():
                print(f"\n{GREEN}All changesets are valid!{RESET}")
            else:
                print(f"\n{RED}Some changesets have errors.{RESET}")
                sys.exit(1)

        else:
            parser.print_help()

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

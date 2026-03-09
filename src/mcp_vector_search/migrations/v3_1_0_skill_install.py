"""Migration v3.1.0: Install or update the mcp-vector-search Claude Code skill.

This migration detects whether the skill is missing or outdated in the project's
Claude skills directory and reinstalls it from the bundled source.

The migration:
1. Locates the Claude skills directory (~/.claude/skills or ~/.config/claude/skills)
2. Reads the installed skill's metadata.json (if present)
3. Compares the installed version against the bundled skill version
4. Reinstalls the skill if missing, outdated, or metadata is corrupt
5. Writes an updated metadata.json with the new version and timestamp

This migration is idempotent: running it multiple times only reinstalls when
the bundled version differs from what is installed.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus

# Skill name (matches the .md filename stem and SKILL.md directory name)
SKILL_NAME = "mcp-vector-search"

# Candidate locations for the Claude skills directory (checked in order)
_CLAUDE_SKILLS_DIRS: list[Path] = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".config" / "claude" / "skills",
    Path.home() / "AppData" / "Roaming" / "Claude" / "skills",  # Windows
]


def _find_claude_skills_dir() -> Path | None:
    """Return the first existing Claude skills directory, or None."""
    for candidate in _CLAUDE_SKILLS_DIRS:
        if candidate.exists():
            return candidate
    return None


def _parse_bundled_version() -> str | None:
    """Extract version from the bundled skill's YAML frontmatter.

    Reads the `version: X.Y.Z` line from the frontmatter block delimited
    by triple-dashes at the top of the skill markdown file.

    Returns:
        Version string (e.g. "1.0.0") or None if not found.
    """
    try:
        from ..skills import get_skill_content

        content = get_skill_content(SKILL_NAME)
    except Exception as e:
        logger.warning(f"Could not read bundled skill content: {e}")
        return None

    # Match frontmatter block: lines between leading --- and closing ---
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        for line in frontmatter.splitlines():
            m = re.match(r"^\s*version\s*:\s*(.+)$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")

    # Fallback: look for **Version**: X.Y.Z anywhere in the document
    m = re.search(r"\*\*Version\*\*\s*:\s*([\d.]+)", content)
    if m:
        return m.group(1).strip()

    return None


def _read_installed_version(skill_dir: Path) -> str | None:
    """Read the installed version from metadata.json.

    Args:
        skill_dir: Path to .claude/skills/mcp-vector-search/

    Returns:
        Installed version string or None if missing / unreadable.
    """
    metadata_file = skill_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("version")
    except Exception as e:
        logger.warning(f"Could not read skill metadata: {e}")
        return None


def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert 'X.Y.Z' to (X, Y, Z) for comparison."""
    try:
        return tuple(int(x) for x in version.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _install_skill(claude_skills_dir: Path, bundled_version: str) -> bool:
    """Copy skill content to the Claude skills directory and write metadata.

    Args:
        claude_skills_dir: Root Claude skills directory.
        bundled_version: Version string from bundled skill frontmatter.

    Returns:
        True on success, False otherwise.
    """
    from ..skills import install_skill_to_claude

    success = install_skill_to_claude(SKILL_NAME, claude_skills_dir)
    if not success:
        return False

    # Write metadata.json alongside SKILL.md
    skill_dir = claude_skills_dir / SKILL_NAME
    metadata = {
        "version": bundled_version,
        "installed_at": datetime.now(UTC).isoformat(),
    }
    try:
        with open(skill_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Skill installed but could not write metadata: {e}")
        # Not fatal — the skill itself was installed successfully

    return True


class SkillInstallMigration(Migration):
    """Install or update the mcp-vector-search Claude Code skill."""

    version = "3.1.0"
    name = "skill_install"
    description = (
        "Install or update the mcp-vector-search skill in Claude's skills directory"
    )

    def check_needed(self, context: MigrationContext) -> bool:
        """Return True when the skill is absent or the installed version is older.

        Checks:
        1. Can we locate a Claude skills directory?
        2. Is the bundled skill readable and versioned?
        3. Is the installed skill missing or older than bundled?

        Returns:
            True if migration should run.
        """
        claude_skills_dir = _find_claude_skills_dir()
        if claude_skills_dir is None:
            logger.debug("No Claude skills directory found — skipping skill migration")
            return False

        bundled_version = _parse_bundled_version()
        if bundled_version is None:
            logger.warning("Could not determine bundled skill version — skipping")
            return False

        skill_dir = claude_skills_dir / SKILL_NAME
        installed_version = _read_installed_version(skill_dir)

        if installed_version is None:
            logger.info(
                f"Skill '{SKILL_NAME}' not installed — migration needed "
                f"(bundled: {bundled_version})"
            )
            return True

        if _version_tuple(bundled_version) > _version_tuple(installed_version):
            logger.info(
                f"Skill '{SKILL_NAME}' outdated "
                f"(installed: {installed_version}, bundled: {bundled_version}) "
                "— migration needed"
            )
            return True

        logger.debug(
            f"Skill '{SKILL_NAME}' is up to date (version {installed_version})"
        )
        return False

    def execute(self, context: MigrationContext) -> MigrationResult:
        """Install or update the skill.

        Returns:
            MigrationResult with status and version metadata.
        """
        if context.dry_run:
            bundled_version = _parse_bundled_version() or "unknown"
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message=(
                    f"DRY RUN: Would install/update skill '{SKILL_NAME}' "
                    f"to version {bundled_version}"
                ),
            )

        bundled_version = _parse_bundled_version()
        if bundled_version is None:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message="Could not determine bundled skill version",
                executed_at=datetime.now(),
            )

        claude_skills_dir = _find_claude_skills_dir()
        if claude_skills_dir is None:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SKIPPED,
                message="No Claude skills directory found — skipping skill install",
                executed_at=datetime.now(),
            )

        skill_dir = claude_skills_dir / SKILL_NAME
        previous_version = _read_installed_version(skill_dir)

        success = _install_skill(claude_skills_dir, bundled_version)
        if not success:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message=f"Failed to install skill '{SKILL_NAME}'",
                executed_at=datetime.now(),
            )

        action = "Updated" if previous_version else "Installed"
        message = f"{action} skill '{SKILL_NAME}' to version {bundled_version}"
        if previous_version:
            message += f" (was {previous_version})"

        logger.info(message)
        return MigrationResult(
            migration_id=self.migration_id,
            version=self.version,
            name=self.name,
            status=MigrationStatus.SUCCESS,
            message=message,
            executed_at=datetime.now(),
            metadata={
                "skill_name": SKILL_NAME,
                "bundled_version": bundled_version,
                "previous_version": previous_version or "not installed",
                "install_path": str(claude_skills_dir / SKILL_NAME / "SKILL.md"),
            },
        )

    def rollback(self, context: MigrationContext) -> bool:
        """Rollback not supported — skill files are non-destructive additions."""
        logger.info(
            "Rollback not supported for skill install migration. "
            "To revert, delete the skill directory manually: "
            f"~/.claude/skills/{SKILL_NAME}/"
        )
        return False

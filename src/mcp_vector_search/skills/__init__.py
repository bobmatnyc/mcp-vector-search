"""Skills for mcp-vector-search."""

import importlib.resources
from pathlib import Path


def get_available_skills() -> list[str]:
    """Get list of available skills."""
    skills = []
    try:
        # Python 3.11+ style
        files = importlib.resources.files(__package__)
        for file in files.iterdir():
            if file.name.endswith(".md"):
                skills.append(file.name[:-3])  # Remove .md extension
    except AttributeError:
        # Fallback for older Python versions
        import pkg_resources

        try:
            resource_dir = pkg_resources.resource_filename(__name__, "")
            path = Path(resource_dir)
            if path.exists():
                skills = [f.stem for f in path.glob("*.md")]
        except Exception:
            pass

    return skills


def get_skill_content(skill_name: str) -> str:
    """Get the content of a skill file."""
    skill_file = f"{skill_name}.md"
    try:
        # Python 3.11+ style
        files = importlib.resources.files(__package__)
        return (files / skill_file).read_text(encoding="utf-8")
    except AttributeError:
        # Fallback for older Python versions
        import pkg_resources

        return pkg_resources.resource_string(__name__, skill_file).decode("utf-8")


def install_skill_to_claude(
    skill_name: str,
    claude_skills_dir: Path,
    force_reinstall: bool = False,
) -> bool:
    """Install a skill to Claude's skills directory.

    Version-aware: compares the bundled skill version against the installed
    version recorded in metadata.json and skips reinstall when already
    up to date (unless force_reinstall=True).

    Args:
        skill_name: Name of the skill (without .md extension)
        claude_skills_dir: Path to Claude's skills directory
        force_reinstall: If True, reinstall even when versions match

    Returns:
        True if installed (or already up to date), False on error
    """
    try:
        skill_content = get_skill_content(skill_name)

        skill_dir = claude_skills_dir / skill_name
        metadata_file = skill_dir / "metadata.json"

        # Parse bundled version from frontmatter
        bundled_version = _parse_skill_version(skill_content)

        if not force_reinstall and bundled_version and metadata_file.exists():
            try:
                import json as _json

                installed_meta = _json.loads(metadata_file.read_text(encoding="utf-8"))
                installed_version = installed_meta.get("version")
                if installed_version and installed_version == bundled_version:
                    # Already at the right version — nothing to do
                    return True
            except Exception:
                pass  # Corrupt metadata — fall through to reinstall

        # Create skill directory
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md file
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_content, encoding="utf-8")

        # Write metadata.json
        if bundled_version:
            import json as _json
            from datetime import UTC, datetime

            metadata = {
                "version": bundled_version,
                "installed_at": datetime.now(UTC).isoformat(),
            }
            metadata_file.write_text(_json.dumps(metadata, indent=2), encoding="utf-8")

        return True
    except Exception as e:
        print(f"Failed to install skill {skill_name}: {e}")
        return False


def update_skill_if_outdated(skill_name: str, claude_skills_dir: Path) -> bool:
    """Reinstall a skill only when the bundled version is newer than installed.

    Convenience wrapper around install_skill_to_claude for callers that
    want an explicit "update" semantic without touching force_reinstall.

    Args:
        skill_name: Name of the skill (without .md extension)
        claude_skills_dir: Path to Claude's skills directory

    Returns:
        True if the skill is now up to date, False on error
    """
    try:
        import json as _json
        import re as _re

        skill_content = get_skill_content(skill_name)
        bundled_version = _parse_skill_version(skill_content)

        if bundled_version is None:
            # No version info — always reinstall to be safe
            return install_skill_to_claude(skill_name, claude_skills_dir)

        skill_dir = claude_skills_dir / skill_name
        metadata_file = skill_dir / "metadata.json"

        installed_version: str | None = None
        if metadata_file.exists():
            try:
                data = _json.loads(metadata_file.read_text(encoding="utf-8"))
                installed_version = data.get("version")
            except Exception:
                pass

        def _ver(v: str) -> tuple[int, ...]:
            try:
                return tuple(int(x) for x in v.split("."))
            except (ValueError, AttributeError):
                return (0, 0, 0)

        if installed_version and _ver(bundled_version) <= _ver(installed_version):
            return True  # Already up to date

        return install_skill_to_claude(skill_name, claude_skills_dir)

    except Exception as e:
        print(f"Failed to update skill {skill_name}: {e}")
        return False


def _parse_skill_version(content: str) -> str | None:
    """Extract version string from skill markdown frontmatter or **Version** line.

    Args:
        content: Full text of the skill markdown file

    Returns:
        Version string (e.g. "1.0.0") or None if not found
    """
    import re as _re

    # Try YAML frontmatter block first (--- ... ---)
    fm_match = _re.match(r"^---\s*\n(.*?)\n---", content, _re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            m = _re.match(r"^\s*version\s*:\s*(.+)$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")

    # Fallback: **Version**: X.Y.Z anywhere in the document
    m = _re.search(r"\*\*Version\*\*\s*:\s*([\d.]+)", content)
    if m:
        return m.group(1).strip()

    return None


def get_skill_metadata(skill_name: str) -> dict[str, str]:
    """Extract metadata from a skill file."""
    try:
        content = get_skill_content(skill_name)
        metadata = {}

        # Parse title from first header
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break

        # Parse version and other metadata from bottom of file
        for line in reversed(lines):
            if line.startswith("**Version**:"):
                metadata["version"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Compatible**:"):
                metadata["compatible"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Author**:"):
                metadata["author"] = line.split(":", 1)[1].strip()

        return metadata
    except Exception:
        return {}

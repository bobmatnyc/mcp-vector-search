"""Skills for mcp-vector-search."""

import importlib.resources
from pathlib import Path
from typing import Dict, List


def get_available_skills() -> List[str]:
    """Get list of available skills."""
    skills = []
    try:
        # Python 3.11+ style
        files = importlib.resources.files(__package__)
        for file in files.iterdir():
            if file.name.endswith('.md'):
                skills.append(file.name[:-3])  # Remove .md extension
    except AttributeError:
        # Fallback for older Python versions
        import pkg_resources
        try:
            resource_dir = pkg_resources.resource_filename(__name__, '')
            path = Path(resource_dir)
            if path.exists():
                skills = [f.stem for f in path.glob('*.md')]
        except Exception:
            pass

    return skills


def get_skill_content(skill_name: str) -> str:
    """Get the content of a skill file."""
    skill_file = f"{skill_name}.md"
    try:
        # Python 3.11+ style
        files = importlib.resources.files(__package__)
        return (files / skill_file).read_text(encoding='utf-8')
    except AttributeError:
        # Fallback for older Python versions
        import pkg_resources
        return pkg_resources.resource_string(__name__, skill_file).decode('utf-8')


def install_skill_to_claude(skill_name: str, claude_skills_dir: Path) -> bool:
    """Install a skill to Claude's skills directory.

    Args:
        skill_name: Name of the skill (without .md extension)
        claude_skills_dir: Path to Claude's skills directory

    Returns:
        True if installed successfully, False otherwise
    """
    try:
        skill_content = get_skill_content(skill_name)

        # Create skill directory
        skill_dir = claude_skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md file
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_content, encoding='utf-8')

        return True
    except Exception as e:
        print(f"Failed to install skill {skill_name}: {e}")
        return False


def get_skill_metadata(skill_name: str) -> Dict[str, str]:
    """Extract metadata from a skill file."""
    try:
        content = get_skill_content(skill_name)
        metadata = {}

        # Parse title from first header
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                metadata['title'] = line[2:].strip()
                break

        # Parse version and other metadata from bottom of file
        for line in reversed(lines):
            if line.startswith('**Version**:'):
                metadata['version'] = line.split(':', 1)[1].strip()
            elif line.startswith('**Compatible**:'):
                metadata['compatible'] = line.split(':', 1)[1].strip()
            elif line.startswith('**Author**:'):
                metadata['author'] = line.split(':', 1)[1].strip()

        return metadata
    except Exception:
        return {}
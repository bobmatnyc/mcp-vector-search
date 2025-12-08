"""Git hooks for automatic reindexing."""

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger


class GitHookManager:
    """Manages Git hooks for automatic reindexing."""

    def __init__(self, project_root: Path):
        """Initialize Git hook manager.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.git_dir = project_root / ".git"
        self.hooks_dir = self.git_dir / "hooks"

    def is_git_repo(self) -> bool:
        """Check if project is a Git repository."""
        return self.git_dir.exists() and self.git_dir.is_dir()

    def install_hooks(self, hook_types: list[str] | None = None) -> bool:
        """Install Git hooks for automatic reindexing.

        Args:
            hook_types: List of hook types to install (default: ['post-commit', 'post-merge'])

        Returns:
            True if hooks were installed successfully
        """
        if not self.is_git_repo():
            logger.error("Not a Git repository")
            return False

        if hook_types is None:
            hook_types = ["post-commit", "post-merge", "post-checkout"]

        success = True
        for hook_type in hook_types:
            if not self._install_hook(hook_type):
                success = False

        return success

    def uninstall_hooks(self, hook_types: list[str] | None = None) -> bool:
        """Uninstall Git hooks.

        Args:
            hook_types: List of hook types to uninstall (default: all MCP hooks)

        Returns:
            True if hooks were uninstalled successfully
        """
        if hook_types is None:
            hook_types = ["post-commit", "post-merge", "post-checkout"]

        success = True
        for hook_type in hook_types:
            if not self._uninstall_hook(hook_type):
                success = False

        return success

    def _install_hook(self, hook_type: str) -> bool:
        """Install a specific Git hook."""
        try:
            hook_file = self.hooks_dir / hook_type

            # Create hooks directory if it doesn't exist
            self.hooks_dir.mkdir(exist_ok=True)

            # Generate hook script
            hook_script = self._generate_hook_script(hook_type)

            if hook_file.exists():
                # If hook already exists, try to integrate with it
                return self._integrate_with_existing_hook(hook_file, hook_script)
            else:
                # Create new hook
                hook_file.write_text(hook_script)
                hook_file.chmod(0o755)  # Make executable
                logger.info(f"Installed {hook_type} hook")
                return True

        except Exception as e:
            logger.error(f"Failed to install {hook_type} hook: {e}")
            return False

    def _uninstall_hook(self, hook_type: str) -> bool:
        """Uninstall a specific Git hook."""
        try:
            hook_file = self.hooks_dir / hook_type

            if not hook_file.exists():
                return True  # Already uninstalled

            content = hook_file.read_text()

            # Check if this is our hook or integrated
            if "# MCP Vector Search Hook" in content:
                if (
                    content.strip().startswith("#!/bin/bash")
                    and "# MCP Vector Search Hook" in content
                ):
                    # This is our hook, remove it
                    hook_file.unlink()
                    logger.info(f"Uninstalled {hook_type} hook")
                else:
                    # This is integrated, remove our part
                    return self._remove_from_existing_hook(hook_file)

            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {hook_type} hook: {e}")
            return False

    def _generate_hook_script(self, hook_type: str) -> str:
        """Generate Git hook script."""
        python_path = sys.executable
        project_root = str(self.project_root)

        script = f"""#!/bin/bash
# MCP Vector Search Hook - {hook_type}
# Auto-generated - do not edit manually

# Check if mcp-vector-search is available
if ! command -v mcp-vector-search &> /dev/null; then
    # Try using Python directly
    if [ -f "{python_path}" ]; then
        PYTHON_CMD="{python_path}"
    else
        PYTHON_CMD="python3"
    fi

    # Try to run via Python module
    if $PYTHON_CMD -m mcp_vector_search --help &> /dev/null; then
        MCP_CMD="$PYTHON_CMD -m mcp_vector_search"
    else
        # Silently exit if not available
        exit 0
    fi
else
    MCP_CMD="mcp-vector-search"
fi

# Change to project directory
cd "{project_root}" || exit 0

# Run auto-indexing check
$MCP_CMD auto-index check --auto-reindex --max-files 10 &> /dev/null || true

# Exit successfully (don't block Git operations)
exit 0
"""
        return script

    def _integrate_with_existing_hook(self, hook_file: Path, our_script: str) -> bool:
        """Integrate our hook with an existing hook."""
        try:
            existing_content = hook_file.read_text()

            # Check if our hook is already integrated
            if "# MCP Vector Search Hook" in existing_content:
                logger.info(f"Hook {hook_file.name} already integrated")
                return True

            # Add our hook to the end
            integrated_content = existing_content.rstrip() + "\n\n" + our_script

            # Backup original
            backup_file = hook_file.with_suffix(hook_file.suffix + ".backup")
            backup_file.write_text(existing_content)

            # Write integrated version
            hook_file.write_text(integrated_content)

            logger.info(f"Integrated with existing {hook_file.name} hook")
            return True

        except Exception as e:
            logger.error(f"Failed to integrate with existing hook: {e}")
            return False

    def _remove_from_existing_hook(self, hook_file: Path) -> bool:
        """Remove our hook from an existing integrated hook."""
        try:
            content = hook_file.read_text()

            # Find and remove our section
            lines = content.split("\n")
            new_lines = []
            skip_section = False

            for line in lines:
                if "# MCP Vector Search Hook" in line:
                    skip_section = True
                    continue
                elif skip_section and line.strip() == "":
                    # End of our section
                    skip_section = False
                    continue
                elif not skip_section:
                    new_lines.append(line)

            # Write back the cleaned content
            hook_file.write_text("\n".join(new_lines))

            logger.info(f"Removed MCP hook from {hook_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove from existing hook: {e}")
            return False

    def get_hook_status(self) -> dict:
        """Get status of Git hooks."""
        if not self.is_git_repo():
            return {"is_git_repo": False}

        hook_types = ["post-commit", "post-merge", "post-checkout"]
        status = {
            "is_git_repo": True,
            "hooks_dir_exists": self.hooks_dir.exists(),
            "hooks": {},
        }

        for hook_type in hook_types:
            hook_file = self.hooks_dir / hook_type
            hook_status = {
                "exists": hook_file.exists(),
                "executable": False,
                "has_mcp_hook": False,
                "is_mcp_only": False,
            }

            if hook_file.exists():
                try:
                    hook_status["executable"] = os.access(hook_file, os.X_OK)
                    content = hook_file.read_text()
                    hook_status["has_mcp_hook"] = "# MCP Vector Search Hook" in content
                    hook_status["is_mcp_only"] = (
                        hook_status["has_mcp_hook"]
                        and content.strip().startswith("#!/bin/bash")
                        and content.count("# MCP Vector Search Hook") == 1
                    )
                except Exception:
                    pass

            status["hooks"][hook_type] = hook_status

        return status


class GitChangeDetector:
    """Detects changed files from Git operations."""

    @staticmethod
    def get_changed_files_since_commit(
        commit_hash: str, project_root: Path
    ) -> set[Path]:
        """Get files changed since a specific commit.

        Args:
            commit_hash: Git commit hash
            project_root: Project root directory

        Returns:
            Set of changed file paths
        """
        try:
            result = subprocess.run(  # nosec B607
                ["git", "diff", "--name-only", commit_hash, "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            changed_files = set()
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = project_root / line
                    if file_path.exists():
                        changed_files.add(file_path)

            return changed_files

        except subprocess.CalledProcessError:
            return set()

    @staticmethod
    def get_changed_files_in_last_commit(project_root: Path) -> set[Path]:
        """Get files changed in the last commit.

        Args:
            project_root: Project root directory

        Returns:
            Set of changed file paths
        """
        try:
            result = subprocess.run(  # nosec B607
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            changed_files = set()
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = project_root / line
                    if file_path.exists():
                        changed_files.add(file_path)

            return changed_files

        except subprocess.CalledProcessError:
            return set()

    @staticmethod
    def should_trigger_reindex(
        changed_files: set[Path], file_extensions: list[str]
    ) -> bool:
        """Check if changed files should trigger reindexing.

        Args:
            changed_files: Set of changed file paths
            file_extensions: List of file extensions to monitor

        Returns:
            True if reindexing should be triggered
        """
        for file_path in changed_files:
            if file_path.suffix in file_extensions:
                return True
        return False

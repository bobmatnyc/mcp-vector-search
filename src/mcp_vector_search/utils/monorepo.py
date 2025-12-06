"""Monorepo detection and subproject identification."""

import json
from pathlib import Path
from typing import NamedTuple

from loguru import logger

# Directories to exclude from subproject detection
# These are typically test/example/docs directories, not actual subprojects
EXCLUDED_SUBPROJECT_DIRS = {
    "tests",
    "test",
    "examples",
    "example",
    "docs",
    "doc",
    "scripts",
    "tools",
    "benchmarks",
    "benchmark",
    "node_modules",
    ".git",
    ".github",
    ".gitlab",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    ".coverage",
    "htmlcov",
}


class Subproject(NamedTuple):
    """Represents a subproject in a monorepo."""

    name: str  # "ewtn-plus-foundation"
    path: Path  # Absolute path to subproject
    relative_path: str  # Relative to monorepo root


class MonorepoDetector:
    """Detects monorepo structure and identifies subprojects."""

    def __init__(self, project_root: Path):
        """Initialize monorepo detector.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self._subprojects: list[Subproject] | None = None

    def _is_excluded_path(self, path: Path) -> bool:
        """Check if a path should be excluded from subproject detection.

        Args:
            path: Path to check (relative to project root)

        Returns:
            True if path should be excluded from subproject detection
        """
        try:
            relative_path = path.relative_to(self.project_root)
            # Check if any part of the path is in the excluded set
            return any(part in EXCLUDED_SUBPROJECT_DIRS for part in relative_path.parts)
        except ValueError:
            # Path is not relative to project root
            return True

    def is_monorepo(self) -> bool:
        """Check if project is a monorepo.

        Returns:
            True if monorepo structure detected
        """
        return bool(self.detect_subprojects())

    def detect_subprojects(self) -> list[Subproject]:
        """Detect all subprojects in the monorepo.

        Returns:
            List of detected subprojects
        """
        if self._subprojects is not None:
            return self._subprojects

        subprojects = []

        # Try package.json workspaces (npm/yarn/pnpm)
        subprojects.extend(self._detect_npm_workspaces())

        # Try lerna.json
        if not subprojects:
            subprojects.extend(self._detect_lerna_packages())

        # Try pnpm-workspace.yaml
        if not subprojects:
            subprojects.extend(self._detect_pnpm_workspaces())

        # Try nx workspace
        if not subprojects:
            subprojects.extend(self._detect_nx_workspace())

        # Fallback: Look for multiple package.json files
        if not subprojects:
            subprojects.extend(self._detect_by_package_json())

        self._subprojects = subprojects
        logger.debug(f"Detected {len(subprojects)} subprojects in {self.project_root}")

        return subprojects

    def _detect_npm_workspaces(self) -> list[Subproject]:
        """Detect npm/yarn/pnpm workspaces from package.json.

        Returns:
            List of subprojects from workspaces
        """
        package_json = self.project_root / "package.json"
        if not package_json.exists():
            return []

        try:
            with open(package_json) as f:
                data = json.load(f)

            workspaces = data.get("workspaces", [])

            # Handle both array and object format
            if isinstance(workspaces, dict):
                workspaces = workspaces.get("packages", [])

            return self._expand_workspace_patterns(workspaces)

        except Exception as e:
            logger.debug(f"Failed to parse package.json workspaces: {e}")
            return []

    def _detect_lerna_packages(self) -> list[Subproject]:
        """Detect lerna packages from lerna.json.

        Returns:
            List of subprojects from lerna
        """
        lerna_json = self.project_root / "lerna.json"
        if not lerna_json.exists():
            return []

        try:
            with open(lerna_json) as f:
                data = json.load(f)

            packages = data.get("packages", ["packages/*"])
            return self._expand_workspace_patterns(packages)

        except Exception as e:
            logger.debug(f"Failed to parse lerna.json: {e}")
            return []

    def _detect_pnpm_workspaces(self) -> list[Subproject]:
        """Detect pnpm workspaces from pnpm-workspace.yaml.

        Returns:
            List of subprojects from pnpm
        """
        pnpm_workspace = self.project_root / "pnpm-workspace.yaml"
        if not pnpm_workspace.exists():
            return []

        try:
            import yaml

            with open(pnpm_workspace) as f:
                data = yaml.safe_load(f)

            packages = data.get("packages", [])
            return self._expand_workspace_patterns(packages)

        except ImportError:
            logger.debug("pyyaml not installed, skipping pnpm-workspace.yaml detection")
            return []
        except Exception as e:
            logger.debug(f"Failed to parse pnpm-workspace.yaml: {e}")
            return []

    def _detect_nx_workspace(self) -> list[Subproject]:
        """Detect nx workspace projects.

        Returns:
            List of subprojects from nx workspace
        """
        nx_json = self.project_root / "nx.json"
        workspace_json = self.project_root / "workspace.json"

        if not (nx_json.exists() or workspace_json.exists()):
            return []

        # Nx projects are typically in apps/ and libs/
        subprojects = []
        for base_dir in ["apps", "libs", "packages"]:
            base_path = self.project_root / base_dir
            if base_path.exists():
                for subdir in base_path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith("."):
                        # Skip excluded directories
                        if self._is_excluded_path(subdir):
                            logger.debug(
                                f"Skipping excluded nx workspace path: {subdir.relative_to(self.project_root)}"
                            )
                            continue

                        package_json = subdir / "package.json"
                        name = self._get_package_name(package_json) or subdir.name
                        relative = str(subdir.relative_to(self.project_root))
                        subprojects.append(Subproject(name, subdir, relative))

        return subprojects

    def _detect_by_package_json(self) -> list[Subproject]:
        """Fallback: Find all directories with package.json.

        Returns:
            List of subprojects by package.json presence
        """
        subprojects = []

        # Only search up to 3 levels deep
        for package_json in self.project_root.rglob("package.json"):
            # Skip root package.json
            if package_json.parent == self.project_root:
                continue

            # Skip excluded directories (tests, examples, docs, etc.)
            if self._is_excluded_path(package_json.parent):
                logger.debug(
                    f"Skipping excluded path: {package_json.relative_to(self.project_root)}"
                )
                continue

            # Check depth
            relative_parts = package_json.relative_to(self.project_root).parts
            if len(relative_parts) > 4:  # Too deep
                continue

            subdir = package_json.parent
            name = self._get_package_name(package_json) or subdir.name
            relative = str(subdir.relative_to(self.project_root))
            subprojects.append(Subproject(name, subdir, relative))

        return subprojects

    def _expand_workspace_patterns(self, patterns: list[str]) -> list[Subproject]:
        """Expand workspace glob patterns to actual directories.

        Args:
            patterns: List of glob patterns (e.g., ["packages/*", "apps/*"])

        Returns:
            List of subprojects matching patterns
        """
        subprojects = []

        for pattern in patterns:
            # Remove negation patterns (e.g., "!packages/excluded")
            if pattern.startswith("!"):
                continue

            # Expand glob pattern
            for path in self.project_root.glob(pattern):
                if not path.is_dir():
                    continue

                if path.name.startswith("."):
                    continue

                # Skip excluded directories (tests, examples, docs, etc.)
                if self._is_excluded_path(path):
                    logger.debug(
                        f"Skipping excluded workspace path: {path.relative_to(self.project_root)}"
                    )
                    continue

                # Try to get name from package.json
                package_json = path / "package.json"
                name = self._get_package_name(package_json) or path.name
                relative = str(path.relative_to(self.project_root))

                subprojects.append(Subproject(name, path, relative))

        return subprojects

    def _get_package_name(self, package_json: Path) -> str | None:
        """Get package name from package.json.

        Args:
            package_json: Path to package.json file

        Returns:
            Package name or None
        """
        if not package_json.exists():
            return None

        try:
            with open(package_json) as f:
                data = json.load(f)
            return data.get("name")
        except Exception:
            return None

    def get_subproject_for_file(self, file_path: Path) -> Subproject | None:
        """Determine which subproject a file belongs to.

        Args:
            file_path: Path to file

        Returns:
            Subproject containing the file, or None
        """
        subprojects = self.detect_subprojects()

        if not subprojects:
            return None

        # Find the most specific (deepest) subproject containing this file
        matching_subprojects = [
            sp for sp in subprojects if file_path.is_relative_to(sp.path)
        ]

        if not matching_subprojects:
            return None

        # Return the deepest match (longest path)
        return max(matching_subprojects, key=lambda sp: len(sp.path.parts))

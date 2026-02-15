"""Project detection and management for MCP Vector Search."""

import os
from pathlib import Path

import orjson
from loguru import logger

from ..config.defaults import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_FILE_EXTENSIONS,
    DEFAULT_IGNORE_PATTERNS,
    get_default_config_path,
    get_default_index_path,
    get_language_from_extension,
)
from ..config.settings import ProjectConfig
from ..utils.gitignore import create_gitignore_parser
from .exceptions import (
    ConfigurationError,
    ProjectInitializationError,
    ProjectNotFoundError,
)
from .models import ProjectInfo


class ProjectManager:
    """Manages project detection, initialization, and configuration."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize project manager.

        Args:
            project_root: Project root directory. If None, will auto-detect.
        """
        self.project_root = project_root or self._detect_project_root()
        self._config: ProjectConfig | None = None

        # Initialize gitignore parser
        try:
            self.gitignore_parser = create_gitignore_parser(self.project_root)
        except Exception as e:
            logger.debug(f"Failed to load gitignore patterns: {e}")
            self.gitignore_parser = None

    def _detect_project_root(self) -> Path:
        """Auto-detect project root directory."""
        current = Path.cwd()

        # Look for common project indicators
        indicators = [
            ".git",
            ".mcp-vector-search",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            ".project",
        ]

        # Walk up the directory tree
        for path in [current] + list(current.parents):
            for indicator in indicators:
                if (path / indicator).exists():
                    logger.debug(f"Detected project root: {path} (found {indicator})")
                    return path

        # Default to current directory
        logger.debug(f"Using current directory as project root: {current}")
        return current

    def is_initialized(self) -> bool:
        """Check if project is initialized for MCP Vector Search."""
        config_path = get_default_config_path(self.project_root)
        index_path = get_default_index_path(self.project_root)

        return config_path.exists() and index_path.exists()

    def initialize(
        self,
        file_extensions: list[str] | None = None,
        embedding_model: str | None = None,
        similarity_threshold: float = 0.5,
        force: bool = False,
    ) -> ProjectConfig:
        """Initialize project for MCP Vector Search.

        Args:
            file_extensions: File extensions to index
            embedding_model: Embedding model to use
            similarity_threshold: Similarity threshold for search
            force: Force re-initialization if already exists

        Returns:
            Project configuration

        Raises:
            ProjectInitializationError: If initialization fails
        """
        if self.is_initialized() and not force:
            raise ProjectInitializationError(
                f"Project already initialized at {self.project_root}. Use --force to re-initialize."
            )

        # Use new default model if not specified
        if embedding_model is None:
            embedding_model = DEFAULT_EMBEDDING_MODELS["code"]
            logger.debug(f"Using default embedding model: {embedding_model}")

        try:
            # Backup existing config if forcing re-initialization
            config_path = get_default_config_path(self.project_root)
            if force and config_path.exists():
                backup_path = config_path.with_suffix(".json.bak")
                import shutil

                shutil.copy2(config_path, backup_path)
                logger.info(f"Backed up existing config to {backup_path}")

            # Create index directory
            index_path = get_default_index_path(self.project_root)
            index_path.mkdir(parents=True, exist_ok=True)

            # Ensure .mcp-vector-search/ is in .gitignore
            # This is a non-critical operation - failures are logged but don't block initialization
            try:
                from ..utils.gitignore_updater import ensure_gitignore_entry

                ensure_gitignore_entry(
                    self.project_root,
                    pattern=".mcp-vector-search/",
                    comment="MCP Vector Search index directory",
                )
            except Exception as e:
                # Log warning but continue initialization
                logger.warning(f"Could not update .gitignore: {e}")
                logger.info(
                    "Please manually add '.mcp-vector-search/' to your .gitignore file"
                )

            # When force=True, always use current defaults if no extensions specified
            # This ensures config regeneration picks up new file types
            resolved_extensions = (
                file_extensions
                if file_extensions is not None
                else DEFAULT_FILE_EXTENSIONS
            )

            # Detect languages and files
            detected_languages = self.detect_languages()
            file_count = self.count_indexable_files(resolved_extensions)

            # Create configuration
            config = ProjectConfig(
                project_root=self.project_root,
                index_path=index_path,
                file_extensions=resolved_extensions,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                languages=detected_languages,
            )

            # Save configuration
            self.save_config(config)

            action = "Re-initialized" if force else "Initialized"
            logger.info(
                f"{action} project at {self.project_root}",
                languages=detected_languages,
                file_count=file_count,
                extensions=config.file_extensions,
            )

            self._config = config
            return config

        except Exception as e:
            raise ProjectInitializationError(
                f"Failed to initialize project: {e}"
            ) from e

    def load_config(self) -> ProjectConfig:
        """Load project configuration.

        Returns:
            Project configuration

        Raises:
            ProjectNotFoundError: If project is not initialized
            ConfigurationError: If configuration is invalid
        """
        if not self.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first."
            )

        config_path = get_default_config_path(self.project_root)

        try:
            with open(config_path, "rb") as f:
                config_data = orjson.loads(f.read())

            # Convert paths back to Path objects
            config_data["project_root"] = Path(config_data["project_root"])
            config_data["index_path"] = Path(config_data["index_path"])

            # Merge new file extensions from DEFAULT_FILE_EXTENSIONS
            # This ensures upgrades automatically pick up newly-supported file types
            if "file_extensions" in config_data:
                saved_extensions = set(config_data["file_extensions"])
                current_defaults = set(DEFAULT_FILE_EXTENSIONS)
                new_extensions = current_defaults - saved_extensions

                if new_extensions:
                    # Merge new extensions (preserving user's existing extensions)
                    merged_extensions = sorted(saved_extensions | new_extensions)
                    config_data["file_extensions"] = merged_extensions

                    logger.info(
                        f"Added {len(new_extensions)} new file extensions from updated defaults: {sorted(new_extensions)}"
                    )

            config = ProjectConfig(**config_data)
            self._config = config
            return config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def save_config(self, config: ProjectConfig) -> None:
        """Save project configuration.

        Args:
            config: Project configuration to save

        Raises:
            ConfigurationError: If saving fails
        """
        config_path = get_default_config_path(self.project_root)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert to JSON-serializable format
            config_data = config.model_dump()
            config_data["project_root"] = str(config.project_root)
            config_data["index_path"] = str(config.index_path)

            with open(config_path, "wb") as f:
                # orjson.dumps returns bytes, OPT_INDENT_2 for readability
                f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))

            logger.debug(f"Saved configuration to {config_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}") from e

    @property
    def config(self) -> ProjectConfig:
        """Get project configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def detect_languages(self) -> list[str]:
        """Detect programming languages in the project.

        Returns:
            List of detected language names
        """
        languages: set[str] = set()

        for file_path in self._iter_source_files():
            language = get_language_from_extension(file_path.suffix)
            if language != "text":
                languages.add(language)

        return sorted(languages)

    def count_indexable_files(self, extensions: list[str]) -> int:
        """Count files that can be indexed.

        Args:
            extensions: File extensions to count

        Returns:
            Number of indexable files
        """
        count = 0
        for file_path in self._iter_source_files():
            if file_path.suffix in extensions:
                count += 1
        return count

    def get_project_info(self, file_count: int | None = None) -> ProjectInfo:
        """Get comprehensive project information.

        Args:
            file_count: Optional pre-computed file count (avoids expensive filesystem scan)

        Returns:
            Project information
        """
        config_path = get_default_config_path(self.project_root)
        index_path = get_default_index_path(self.project_root)

        is_initialized = self.is_initialized()
        languages = []
        computed_file_count = 0

        if is_initialized:
            try:
                config = self.config
                languages = config.languages
                # Use provided file_count if available to avoid filesystem scan
                if file_count is not None:
                    computed_file_count = file_count
                else:
                    computed_file_count = self.count_indexable_files(
                        config.file_extensions
                    )
            except Exception:
                # Ignore errors when getting detailed info
                pass

        return ProjectInfo(
            name=self.project_root.name,
            root_path=self.project_root,
            config_path=config_path,
            index_path=index_path,
            is_initialized=is_initialized,
            languages=languages,
            file_count=computed_file_count,
        )

    def _iter_source_files(self) -> list[Path]:
        """Iterate over source files in the project using optimized os.scandir.

        Uses os.scandir for better performance than Path.rglob because it returns
        DirEntry objects with cached stat info, avoiding redundant filesystem calls.

        Returns:
            List of source file paths
        """
        files = []

        # Use os.walk for efficient recursive traversal with early directory filtering
        for root, dirs, filenames in os.walk(self.project_root):
            root_path = Path(root)

            # Filter out ignored directories IN-PLACE to prevent traversal
            # This is much more efficient than checking every file in ignored dirs
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore_path(root_path / d, is_directory=True)
            ]

            # Process files in current directory
            for filename in filenames:
                file_path = root_path / filename

                # Skip symlinks to prevent traversing outside project
                if file_path.is_symlink():
                    continue

                # Skip ignored patterns
                # PERFORMANCE: Pass is_directory=False since os.walk guarantees files
                if self._should_ignore_path(file_path, is_directory=False):
                    continue

                files.append(file_path)

        return files

    def _pattern_could_match_inside_dir(self, dir_path: str, pattern: str) -> bool:
        """Check if a glob pattern could potentially match files inside a directory.

        Args:
            dir_path: Directory path (e.g., "repos" or "repos/subdir")
            pattern: Glob pattern (e.g., "repos/**/*.java")

        Returns:
            True if pattern could match files/subdirectories inside dir_path
        """
        # Normalize
        dir_path = dir_path.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # If pattern starts with the directory path, it could match inside
        # E.g., "repos/**/*.java" could match files in "repos/" or "repos/subdir/"
        if pattern.startswith(dir_path + "/"):
            return True

        # If pattern has ** and the dir_path is a prefix/subdirectory of the pattern's base
        # E.g., pattern "repos/**/*.java" should allow traversing into "repos/subdir/"
        # Check if dir_path is within the pattern's scope
        pattern_parts = pattern.split("/")
        dir_parts = dir_path.split("/")

        # Find ** in pattern
        if "**" in pattern_parts:
            doublestar_idx = pattern_parts.index("**")
            pattern_prefix_parts = pattern_parts[:doublestar_idx]

            # Check if dir_path starts with pattern prefix
            # E.g., pattern "repos/**/*.java" has prefix ["repos"]
            # dir_path "repos/subdir" starts with "repos", so we should traverse
            if len(dir_parts) >= len(pattern_prefix_parts):
                if dir_parts[: len(pattern_prefix_parts)] == pattern_prefix_parts:
                    return True

        # If pattern starts with **/
        # E.g., "**/*.java" could match files in any directory
        if pattern.startswith("**/"):
            return True

        return False

    def _matches_glob_pattern(self, path_str: str, pattern: str) -> bool:
        """Check if a path matches a glob pattern with ** support.

        Args:
            path_str: Path string with forward slashes
            pattern: Glob pattern (supports ** for recursive matching)

        Returns:
            True if path matches pattern
        """
        import fnmatch
        import re

        # Normalize separators
        path_str = path_str.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # Handle ** patterns with regex
        # ** matches zero or more path segments
        if "**" in pattern:
            # Replace **/ with optional directory pattern
            # repos/**/*.java should match:
            #   - repos/test.java (zero directories)
            #   - repos/subdir/test.java (one directory)
            #   - repos/a/b/c/test.java (multiple directories)

            regex_pattern = re.escape(pattern)

            # Handle **/ pattern (matches zero or more directories)
            regex_pattern = regex_pattern.replace(r"\*\*/", "(.*/)?")

            # Handle /** pattern (at end, matches anything)
            regex_pattern = regex_pattern.replace(r"/\*\*", "/.*")

            # Handle standalone ** (matches everything)
            regex_pattern = regex_pattern.replace(r"\*\*", ".*")

            # Handle single * (matches anything except /)
            regex_pattern = regex_pattern.replace(r"\*", "[^/]*")

            # Handle ? (matches one char except /)
            regex_pattern = regex_pattern.replace(r"\?", "[^/]")

            regex_pattern = f"^{regex_pattern}$"

            try:
                if re.match(regex_pattern, path_str):
                    return True
            except re.error:
                pass

        # Try fnmatch for simple patterns
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Try matching any suffix (similar to gitignore behavior)
        path_parts = path_str.split("/")
        for i in range(len(path_parts)):
            subpath = "/".join(path_parts[i:])
            if fnmatch.fnmatch(subpath, pattern):
                return True

        return False

    def _should_ignore_path(self, path: Path, is_directory: bool | None = None) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Path to check
            is_directory: Optional hint if path is a directory (avoids filesystem check)

        Returns:
            True if path should be ignored
        """
        # Load config if needed (for force_include_patterns and force_include_paths)
        try:
            config = self._config or (
                self.load_config() if self.is_initialized() else None
            )
        except Exception:
            config = None

        # FIRST: Check force_include_patterns - they can override EVERYTHING (even DEFAULT_IGNORE_PATTERNS)
        # This allows explicitly including files like vendor/**/*.kt even though vendor is in default ignores
        if config and config.force_include_patterns:
            try:
                relative_path = path.relative_to(self.project_root)
                relative_path_str = str(relative_path).replace("\\", "/")

                # Check if path matches any force_include pattern
                for pattern in config.force_include_patterns:
                    if self._matches_glob_pattern(relative_path_str, pattern):
                        logger.debug(
                            f"Force-including {relative_path} (matched pattern: {pattern})"
                        )
                        return False  # Don't ignore this file

                    # CRITICAL: For directories, check if pattern could match files inside
                    # E.g., if pattern is "repos/**/*.java" and path is "repos/",
                    # we need to traverse into "repos/" to check files inside
                    if is_directory:
                        # Add trailing slash and /* to check if files inside could match
                        dir_pattern_prefix = relative_path_str + "/"
                        if pattern.startswith(
                            dir_pattern_prefix
                        ) or self._pattern_could_match_inside_dir(
                            relative_path_str, pattern
                        ):
                            logger.debug(
                                f"Not ignoring directory {relative_path} (could contain force-included files matching: {pattern})"
                            )
                            return False  # Don't ignore this directory
            except ValueError:
                # Path is not relative to project root
                pass

        # SECOND: Check DEFAULT_IGNORE_PATTERNS (node_modules, __pycache__, etc.)
        # These block force_include_paths but not force_include_patterns
        # This prevents accidentally indexing node_modules with force_include_paths
        for part in path.parts:
            if part in DEFAULT_IGNORE_PATTERNS:
                return True

        # Check relative path from project root
        try:
            relative_path = path.relative_to(self.project_root)
            for part in relative_path.parts:
                if part in DEFAULT_IGNORE_PATTERNS:
                    return True
        except ValueError:
            # Path is not relative to project root
            return True

        # THIRD: Check force_include_paths - they override gitignore (but not default ignores)
        if config and config.force_include_paths:
            try:
                relative_path = path.relative_to(self.project_root)
                relative_path_str = str(relative_path).replace("\\", "/")

                # Check if path is within any force_include_path
                for include_path in config.force_include_paths:
                    # Normalize include_path (remove trailing slash for comparison)
                    include_path_normalized = include_path.rstrip("/")

                    # Check if this exact path or a parent matches
                    if relative_path_str == include_path_normalized:
                        logger.debug(
                            f"Force-including {relative_path} (matched path: {include_path})"
                        )
                        return False  # Don't ignore

                    # Check if path starts with include_path (i.e., it's inside the directory)
                    if relative_path_str.startswith(include_path_normalized + "/"):
                        logger.debug(
                            f"Force-including {relative_path} (inside force_include_path: {include_path})"
                        )
                        return False  # Don't ignore

                    # For directories, check if force_include_path is inside this directory
                    # E.g., if checking "repos/" and force_include_path is "repos/subdir/",
                    # we need to allow traversal into "repos/"
                    if is_directory and include_path_normalized.startswith(
                        relative_path_str + "/"
                    ):
                        logger.debug(
                            f"Not ignoring directory {relative_path} (contains force_include_path: {include_path})"
                        )
                        return False  # Don't ignore
            except ValueError:
                # Path is not relative to project root
                pass

        # FOURTH: Check gitignore rules if available and respect_gitignore is enabled
        # PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
        respect_gitignore = config.respect_gitignore if config else True
        if respect_gitignore:
            if self.gitignore_parser and self.gitignore_parser.is_ignored(
                path, is_directory=is_directory
            ):
                return True

        # DEFAULT_IGNORE_PATTERNS already checked at the top of this function
        return False

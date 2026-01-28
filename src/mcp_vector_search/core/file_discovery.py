"""File discovery and filtering for semantic indexing."""

import asyncio
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from ..config.defaults import ALLOWED_DOTFILES, DEFAULT_IGNORE_PATTERNS
from ..config.settings import ProjectConfig
from ..utils.cancellation import CancellationToken
from ..utils.gitignore import GitignoreParser, create_gitignore_parser


class FileDiscovery:
    """Handles file discovery, filtering, and caching for indexing.

    This class encapsulates all logic related to finding files that should
    be indexed, including gitignore parsing, extension filtering, and
    directory traversal.
    """

    def __init__(
        self,
        project_root: Path,
        file_extensions: set[str],
        config: ProjectConfig | None = None,
        ignore_patterns: set[str] | None = None,
    ) -> None:
        """Initialize file discovery.

        Args:
            project_root: Project root directory
            file_extensions: Set of file extensions to index (e.g., {'.py', '.js'})
            config: Project configuration for filtering behavior
            ignore_patterns: Additional patterns to ignore (merged with defaults)
        """
        self.project_root = project_root
        self.file_extensions = file_extensions
        self.config = config
        self._ignore_patterns = (
            set(DEFAULT_IGNORE_PATTERNS) | ignore_patterns
            if ignore_patterns
            else set(DEFAULT_IGNORE_PATTERNS)
        )

        # Cache for indexable files to avoid repeated filesystem scans
        self._indexable_files_cache: list[Path] | None = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60.0  # 60 second TTL

        # Cache for _should_ignore_path to avoid repeated parent path checks
        # Key: str(path), Value: bool (should ignore)
        self._ignore_path_cache: dict[str, bool] = {}

        # Initialize gitignore parser (only if respect_gitignore is True)
        self.gitignore_parser: GitignoreParser | None = None
        if config is None or config.respect_gitignore:
            try:
                self.gitignore_parser = create_gitignore_parser(project_root)
                logger.debug(
                    f"Loaded {len(self.gitignore_parser.patterns)} gitignore patterns"
                )
            except Exception as e:
                logger.warning(f"Failed to load gitignore patterns: {e}")
        else:
            logger.debug("Gitignore filtering disabled by configuration")

    def find_indexable_files(self) -> list[Path]:
        """Find all files that should be indexed with caching.

        Returns:
            List of file paths to index
        """
        import time

        # Check cache
        current_time = time.time()
        if (
            self._indexable_files_cache is not None
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            logger.debug(
                f"Using cached indexable files ({len(self._indexable_files_cache)} files)"
            )
            return self._indexable_files_cache

        # Rebuild cache using efficient directory filtering
        logger.debug("Rebuilding indexable files cache...")
        indexable_files = self.scan_files_sync()

        self._indexable_files_cache = sorted(indexable_files)
        self._cache_timestamp = current_time
        logger.debug(f"Rebuilt indexable files cache ({len(indexable_files)} files)")

        return self._indexable_files_cache

    def scan_files_sync(
        self, cancel_token: CancellationToken | None = None
    ) -> list[Path]:
        """Synchronous file scanning (runs in thread pool).

        Uses os.walk with directory filtering to avoid traversing ignored directories.

        Args:
            cancel_token: Optional cancellation token to interrupt scanning

        Returns:
            List of indexable file paths

        Raises:
            OperationCancelledError: If cancelled via cancel_token
        """
        indexable_files = []
        dir_count = 0

        # Use os.walk for efficient directory traversal with early filtering
        for root, dirs, files in os.walk(self.project_root):
            # Check for cancellation periodically (every directory)
            if cancel_token:
                cancel_token.check()

            root_path = Path(root)
            dir_count += 1

            # Log progress periodically
            if dir_count % 100 == 0:
                logger.debug(
                    f"Scanned {dir_count} directories, found {len(indexable_files)} indexable files"
                )

            # Filter out ignored directories IN-PLACE to prevent os.walk from traversing them
            # This is much more efficient than checking every file in ignored directories
            # PERFORMANCE: Pass is_directory=True hint to skip filesystem stat() calls
            dirs[:] = [
                d
                for d in dirs
                if not self.should_ignore_path(root_path / d, is_directory=True)
            ]

            # Check each file in the current directory
            # PERFORMANCE: skip_file_check=True because os.walk guarantees these are files
            for filename in files:
                file_path = root_path / filename
                if self.should_index_file(file_path, skip_file_check=True):
                    indexable_files.append(file_path)

        logger.debug(
            f"File scan complete: {dir_count} directories, {len(indexable_files)} indexable files"
        )
        return indexable_files

    async def find_indexable_files_async(
        self, cancel_token: CancellationToken | None = None
    ) -> list[Path]:
        """Find all files asynchronously without blocking event loop.

        Args:
            cancel_token: Optional cancellation token to interrupt scanning

        Returns:
            List of file paths to index

        Raises:
            OperationCancelledError: If cancelled via cancel_token
        """
        import time

        # Check cache first
        current_time = time.time()
        if (
            self._indexable_files_cache is not None
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            logger.debug(
                f"Using cached indexable files ({len(self._indexable_files_cache)} files)"
            )
            return self._indexable_files_cache

        # Run filesystem scan in thread pool to avoid blocking
        logger.debug("Scanning files in background thread...")
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            indexable_files = await loop.run_in_executor(
                executor, lambda: self.scan_files_sync(cancel_token)
            )

        # Update cache
        self._indexable_files_cache = sorted(indexable_files)
        self._cache_timestamp = current_time
        logger.debug(f"Found {len(indexable_files)} indexable files")

        return self._indexable_files_cache

    def should_index_file(self, file_path: Path, skip_file_check: bool = False) -> bool:
        """Check if a file should be indexed.

        Args:
            file_path: Path to check
            skip_file_check: Skip is_file() check if caller knows it's a file (optimization)

        Returns:
            True if file should be indexed
        """
        # PERFORMANCE: Check file extension FIRST (cheapest operation, no I/O)
        # This eliminates most files without any filesystem calls
        if file_path.suffix.lower() not in self.file_extensions:
            return False

        # PERFORMANCE: Only check is_file() if not coming from os.walk
        # os.walk already guarantees files, so we skip this expensive check
        if not skip_file_check and not file_path.is_file():
            return False

        # Check if path should be ignored
        # PERFORMANCE: Pass is_directory=False to skip stat() call (we know it's a file)
        if self.should_ignore_path(file_path, is_directory=False):
            return False

        # Check file size (skip very large files)
        try:
            file_size = file_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
                return False
        except OSError:
            return False

        return True

    def should_ignore_path(
        self, file_path: Path, is_directory: bool | None = None
    ) -> bool:
        """Check if a path should be ignored.

        PERFORMANCE: Cached to avoid repeated checks on parent directories.

        Args:
            file_path: Path to check
            is_directory: Optional hint if path is a directory (avoids filesystem check)

        Returns:
            True if path should be ignored
        """
        # Check cache first
        cache_key = str(file_path)
        if cache_key in self._ignore_path_cache:
            return self._ignore_path_cache[cache_key]

        try:
            # Get relative path from project root for checking
            relative_path = file_path.relative_to(self.project_root)

            # 1. Check dotfile filtering (ENABLED BY DEFAULT)
            # Skip dotfiles unless config explicitly disables it
            skip_dotfiles = self.config.skip_dotfiles if self.config else True
            if skip_dotfiles:
                for part in relative_path.parts:
                    # Skip dotfiles unless they're in the whitelist
                    if part.startswith(".") and part not in ALLOWED_DOTFILES:
                        logger.debug(
                            f"Path ignored by dotfile filter '{part}': {file_path}"
                        )
                        self._ignore_path_cache[cache_key] = True
                        return True

            # 2. Check gitignore rules if available and enabled
            # PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
            if self.config and self.config.respect_gitignore:
                if self.gitignore_parser and self.gitignore_parser.is_ignored(
                    file_path, is_directory=is_directory
                ):
                    logger.debug(f"Path ignored by .gitignore: {file_path}")
                    self._ignore_path_cache[cache_key] = True
                    return True

            # 3. Check each part of the path against default ignore patterns
            # PERFORMANCE: Combine part and parent checks to avoid duplicate iteration
            # Supports both exact matches and wildcard patterns (e.g., ".*" for all dotfiles)
            for part in relative_path.parts:
                for pattern in self._ignore_patterns:
                    # Use fnmatch for wildcard support (*, ?, [seq], [!seq])
                    if fnmatch.fnmatch(part, pattern):
                        logger.debug(
                            f"Path ignored by pattern '{pattern}' matching '{part}': {file_path}"
                        )
                        self._ignore_path_cache[cache_key] = True
                        return True

            # Cache negative result
            self._ignore_path_cache[cache_key] = False
            return False

        except ValueError:
            # Path is not relative to project root
            self._ignore_path_cache[cache_key] = True
            return True

    def add_ignore_pattern(self, pattern: str) -> None:
        """Add a pattern to ignore during indexing.

        Args:
            pattern: Pattern to ignore (directory or file name)
        """
        self._ignore_patterns.add(pattern)
        # Clear cache since ignore rules changed
        self._ignore_path_cache.clear()

    def remove_ignore_pattern(self, pattern: str) -> None:
        """Remove an ignore pattern.

        Args:
            pattern: Pattern to remove
        """
        self._ignore_patterns.discard(pattern)
        # Clear cache since ignore rules changed
        self._ignore_path_cache.clear()

    def get_ignore_patterns(self) -> set[str]:
        """Get current ignore patterns.

        Returns:
            Set of ignore patterns
        """
        return self._ignore_patterns.copy()

    def clear_cache(self) -> None:
        """Clear file discovery caches."""
        self._indexable_files_cache = None
        self._cache_timestamp = 0
        self._ignore_path_cache.clear()

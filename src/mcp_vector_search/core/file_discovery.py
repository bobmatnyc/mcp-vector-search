"""File discovery and filtering for semantic indexing."""

import asyncio
import fnmatch
import os
import re
from collections.abc import Callable
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

        # Pre-compile ignore patterns for performance
        # This converts fnmatch patterns to regex and compiles them once at init
        # instead of calling fnmatch.fnmatch() in hot loop (1-2μs per call)
        self._compiled_patterns = self._compile_ignore_patterns(self._ignore_patterns)

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

    def _compile_ignore_patterns(
        self, patterns: set[str]
    ) -> dict[str, list[re.Pattern[str]]]:
        """Compile fnmatch patterns to regex for fast matching.

        Groups patterns by first character for bucketing optimization.
        This reduces O(N×M) nested loop to O(N×M/k) where k is number of buckets.

        Args:
            patterns: Set of fnmatch patterns (e.g., {'node_modules', 'build', '.*'})

        Returns:
            Dict mapping first char to list of compiled regex patterns
            Special key '*' holds patterns starting with wildcards
        """
        compiled: dict[str, list[re.Pattern[str]]] = {}

        for pattern in patterns:
            # Convert fnmatch pattern to regex using fnmatch.translate()
            # This handles *, ?, [seq], [!seq] wildcards correctly
            try:
                regex_str = fnmatch.translate(pattern)
                compiled_pattern = re.compile(regex_str)

                # Bucket by first character for faster lookup
                # Patterns starting with * go to wildcard bucket
                first_char = (
                    "*"
                    if pattern.startswith("*") or pattern.startswith("?")
                    else pattern[0]
                )

                if first_char not in compiled:
                    compiled[first_char] = []
                compiled[first_char].append(compiled_pattern)

            except re.error as e:
                logger.warning(f"Failed to compile pattern '{pattern}': {e}")
                continue

        # Log bucketing statistics
        bucket_sizes = {k: len(v) for k, v in compiled.items()}
        logger.debug(
            f"Compiled {len(patterns)} ignore patterns into {len(compiled)} buckets: {bucket_sizes}"
        )

        return compiled

    def _matches_compiled_patterns(self, part: str) -> bool:
        """Check if a path part matches any compiled ignore pattern.

        Uses bucketing for faster lookup - only checks patterns that could match.

        Args:
            part: Single path component (e.g., 'node_modules', '.git', 'src')

        Returns:
            True if part matches any ignore pattern
        """
        if not part:
            return False

        # Check patterns bucketed by first character (O(M/k) instead of O(M))
        first_char = part[0]
        patterns_to_check: list[re.Pattern[str]] = []

        # Add patterns from matching bucket
        if first_char in self._compiled_patterns:
            patterns_to_check.extend(self._compiled_patterns[first_char])

        # Always check wildcard patterns (they match any first char)
        if "*" in self._compiled_patterns:
            patterns_to_check.extend(self._compiled_patterns["*"])

        # Test against bucketed patterns only (faster than all patterns)
        for compiled_pattern in patterns_to_check:
            if compiled_pattern.match(part):
                return True

        return False

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
        self,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        file_limit: int | None = None,
    ) -> list[Path]:
        """Synchronous file scanning (runs in thread pool).

        Uses os.walk with directory filtering to avoid traversing ignored directories.

        Args:
            cancel_token: Optional cancellation token to interrupt scanning
            progress_callback: Optional callback(dirs_scanned, files_found) for progress updates
            file_limit: Optional limit - stop scanning once this many files found

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

            # Early exit if we've reached the file limit
            if file_limit is not None and len(indexable_files) >= file_limit:
                logger.debug(
                    f"Reached file limit ({file_limit}), stopping scan at {dir_count} directories"
                )
                break

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
                    # Check limit inside inner loop for fast exit
                    if file_limit is not None and len(indexable_files) >= file_limit:
                        break

            # Call progress callback every 10 directories or when files found
            if progress_callback and (dir_count % 10 == 0 or len(indexable_files) > 0):
                progress_callback(dir_count, len(indexable_files))

        logger.debug(
            f"File scan complete: {dir_count} directories, {len(indexable_files)} indexable files"
        )
        return indexable_files

    async def find_indexable_files_async(
        self,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        file_limit: int | None = None,
    ) -> list[Path]:
        """Find all files asynchronously without blocking event loop.

        Args:
            cancel_token: Optional cancellation token to interrupt scanning
            progress_callback: Optional callback(dirs_scanned, files_found) for progress updates
            file_limit: Optional limit - stop scanning once this many files found

        Returns:
            List of file paths to index

        Raises:
            OperationCancelledError: If cancelled via cancel_token
        """
        import time

        # Skip cache if limit specified (to allow re-scanning with different limits)
        if file_limit is None:
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
        else:
            current_time = time.time()

        # Run filesystem scan in thread pool to avoid blocking
        logger.debug(
            f"Scanning files in background thread{f' (limit: {file_limit})' if file_limit else ''}..."
        )
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            indexable_files = await loop.run_in_executor(
                executor,
                lambda: self.scan_files_sync(
                    cancel_token, progress_callback, file_limit
                ),
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
        if pattern.startswith(dir_path + "/"):
            return True

        # If pattern has ** and the dir_path is within pattern scope
        pattern_parts = pattern.split("/")
        dir_parts = dir_path.split("/")

        # Find ** in pattern
        if "**" in pattern_parts:
            doublestar_idx = pattern_parts.index("**")
            pattern_prefix_parts = pattern_parts[:doublestar_idx]

            # Check if dir_path starts with pattern prefix
            if len(dir_parts) >= len(pattern_prefix_parts):
                if dir_parts[: len(pattern_prefix_parts)] == pattern_prefix_parts:
                    return True

        # If pattern starts with **/
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
        import re

        # Normalize separators
        path_str = path_str.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # Handle ** patterns with regex
        if "**" in pattern:
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
            relative_path_str = str(relative_path).replace("\\", "/")

            # 0. Check force_include_patterns FIRST - they can override everything (even DEFAULT_IGNORE_PATTERNS)
            # This allows explicitly including files like vendor/**/*.kt even though vendor is in default ignores
            if self.config and self.config.force_include_patterns:
                for pattern in self.config.force_include_patterns:
                    if self._matches_glob_pattern(relative_path_str, pattern):
                        self._ignore_path_cache[cache_key] = False
                        return False  # Don't ignore this file

                    # For directories, check if pattern could match files inside
                    if is_directory:
                        if self._pattern_could_match_inside_dir(
                            relative_path_str, pattern
                        ):
                            self._ignore_path_cache[cache_key] = False
                            return False  # Don't ignore this directory

            # 1. Check DEFAULT_IGNORE_PATTERNS SECOND - these block force_include_paths but not force_include_patterns
            # This prevents accidentally indexing node_modules via force_include_paths while allowing explicit patterns
            # PERFORMANCE OPTIMIZED: Use pre-compiled patterns with bucketing
            # Instead of O(N×M) nested loop with fnmatch calls, use O(N×M/k) with compiled regex
            # where N=path parts, M=patterns, k=buckets (~10x faster for 283 patterns)
            for part in relative_path.parts:
                if self._matches_compiled_patterns(part):
                    self._ignore_path_cache[cache_key] = True
                    return True

            # 2. Check force_include_paths THIRD - they override gitignore only (not default ignores)
            if self.config and self.config.force_include_paths:
                for include_path in self.config.force_include_paths:
                    # Normalize include_path (remove trailing slash for comparison)
                    include_path_normalized = include_path.rstrip("/")

                    # Check if this exact path or a parent matches
                    if relative_path_str == include_path_normalized:
                        self._ignore_path_cache[cache_key] = False
                        return False  # Don't ignore

                    # Check if path starts with include_path (i.e., it's inside the directory)
                    if relative_path_str.startswith(include_path_normalized + "/"):
                        self._ignore_path_cache[cache_key] = False
                        return False  # Don't ignore

                    # For directories, check if force_include_path is inside this directory
                    # E.g., if checking "repos/" and force_include_path is "repos/subdir/",
                    # we need to allow traversal into "repos/"
                    if is_directory and include_path_normalized.startswith(
                        relative_path_str + "/"
                    ):
                        self._ignore_path_cache[cache_key] = False
                        return False  # Don't ignore

            # 3. Check dotfile filtering (ENABLED BY DEFAULT)
            # Skip dotfiles unless config explicitly disables it
            skip_dotfiles = self.config.skip_dotfiles if self.config else True
            if skip_dotfiles:
                for part in relative_path.parts:
                    # Skip dotfiles unless they're in the whitelist
                    if part.startswith(".") and part not in ALLOWED_DOTFILES:
                        self._ignore_path_cache[cache_key] = True
                        return True

            # 4. Check gitignore rules if available and enabled
            # PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
            if self.config and self.config.respect_gitignore:
                if self.gitignore_parser and self.gitignore_parser.is_ignored(
                    file_path, is_directory=is_directory
                ):
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
        # Recompile patterns for performance
        self._compiled_patterns = self._compile_ignore_patterns(self._ignore_patterns)
        # Clear cache since ignore rules changed
        self._ignore_path_cache.clear()

    def remove_ignore_pattern(self, pattern: str) -> None:
        """Remove an ignore pattern.

        Args:
            pattern: Pattern to remove
        """
        self._ignore_patterns.discard(pattern)
        # Recompile patterns for performance
        self._compiled_patterns = self._compile_ignore_patterns(self._ignore_patterns)
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

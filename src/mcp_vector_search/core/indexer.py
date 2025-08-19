"""Semantic indexer for MCP Vector Search."""

import asyncio
import json
import os
from pathlib import Path

from loguru import logger

from ..config.defaults import DEFAULT_IGNORE_PATTERNS
from ..parsers.registry import get_parser_registry
from ..utils.gitignore import create_gitignore_parser, GitignoreParser
from .database import VectorDatabase
from .exceptions import ParsingError
from .models import CodeChunk


class SemanticIndexer:
    """Semantic indexer for parsing and indexing code files."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        file_extensions: list[str],
        max_workers: int | None = None,
        batch_size: int = 10,
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index
            max_workers: Maximum number of worker threads for parallel processing
            batch_size: Number of files to process in each batch
        """
        self.database = database
        self.project_root = project_root
        self.file_extensions = {ext.lower() for ext in file_extensions}
        self.parser_registry = get_parser_registry()
        self._ignore_patterns = set(DEFAULT_IGNORE_PATTERNS)
        self.max_workers = max_workers or min(
            4, (asyncio.get_event_loop().get_debug() and 1) or 4
        )
        self.batch_size = batch_size
        self._index_metadata_file = (
            project_root / ".mcp-vector-search" / "index_metadata.json"
        )

        # Initialize gitignore parser
        try:
            self.gitignore_parser = create_gitignore_parser(project_root)
            logger.debug(f"Loaded {len(self.gitignore_parser.patterns)} gitignore patterns")
        except Exception as e:
            logger.warning(f"Failed to load gitignore patterns: {e}")
            self.gitignore_parser = None

    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
    ) -> int:
        """Index all files in the project.

        Args:
            force_reindex: Whether to reindex existing files
            show_progress: Whether to show progress information

        Returns:
            Number of files indexed
        """
        logger.info(f"Starting indexing of project: {self.project_root}")

        # Find all indexable files
        all_files = self._find_indexable_files()

        if not all_files:
            logger.warning("No indexable files found")
            return 0

        # Load existing metadata for incremental indexing
        metadata = self._load_index_metadata()

        # Filter files that need indexing
        if force_reindex:
            files_to_index = all_files
            logger.info(f"Force reindex: processing all {len(files_to_index)} files")
        else:
            files_to_index = [
                f for f in all_files if self._needs_reindexing(f, metadata)
            ]
            logger.info(
                f"Incremental index: {len(files_to_index)} of {len(all_files)} files need updating"
            )

        if not files_to_index:
            logger.info("All files are up to date")
            return 0

        # Index files in parallel batches
        indexed_count = 0
        failed_count = 0

        # Process files in batches for better memory management
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]

            if show_progress:
                logger.info(
                    f"Processing batch {i // self.batch_size + 1}/{(len(files_to_index) + self.batch_size - 1) // self.batch_size} ({len(batch)} files)"
                )

            # Process batch in parallel
            batch_results = await self._process_file_batch(batch, force_reindex)

            # Count results
            for success in batch_results:
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1

        # Update metadata for successfully indexed files
        if indexed_count > 0:
            for file_path in files_to_index:
                try:
                    metadata[str(file_path)] = os.path.getmtime(file_path)
                except OSError:
                    pass  # File might have been deleted during indexing

            self._save_index_metadata(metadata)

        logger.info(
            f"Indexing complete: {indexed_count} files indexed, {failed_count} failed"
        )

        return indexed_count

    async def _process_file_batch(
        self, file_paths: list[Path], force_reindex: bool = False
    ) -> list[bool]:
        """Process a batch of files in parallel.

        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing

        Returns:
            List of success flags for each file
        """
        # Create tasks for parallel processing
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self._index_file_safe(file_path, force_reindex))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to success flags
        success_flags = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to index {file_paths[i]}: {result}")
                success_flags.append(False)
            else:
                success_flags.append(result)

        return success_flags

    def _load_index_metadata(self) -> dict[str, float]:
        """Load file modification times from metadata file.

        Returns:
            Dictionary mapping file paths to modification times
        """
        if not self._index_metadata_file.exists():
            return {}

        try:
            with open(self._index_metadata_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return {}

    def _save_index_metadata(self, metadata: dict[str, float]) -> None:
        """Save file modification times to metadata file.

        Args:
            metadata: Dictionary mapping file paths to modification times
        """
        try:
            # Ensure directory exists
            self._index_metadata_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._index_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index metadata: {e}")

    def _needs_reindexing(self, file_path: Path, metadata: dict[str, float]) -> bool:
        """Check if a file needs reindexing based on modification time.

        Args:
            file_path: Path to the file
            metadata: Current metadata dictionary

        Returns:
            True if file needs reindexing
        """
        try:
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = metadata.get(str(file_path), 0)
            return current_mtime > stored_mtime
        except OSError:
            # File doesn't exist or can't be accessed
            return False

    async def _index_file_safe(
        self, file_path: Path, force_reindex: bool = False
    ) -> bool:
        """Safely index a single file with error handling.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to force reindexing

        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.index_file(file_path, force_reindex)
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return False

    async def index_file(
        self,
        file_path: Path,
        force_reindex: bool = False,
    ) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file to index
            force_reindex: Whether to reindex if already indexed

        Returns:
            True if file was successfully indexed
        """
        try:
            # Check if file should be indexed
            if not self._should_index_file(file_path):
                return False

            # Always remove existing chunks when reindexing a file
            # This prevents duplicate chunks and ensures consistency
            await self.database.delete_by_file(file_path)

            # Parse file into chunks
            chunks = await self._parse_file(file_path)

            if not chunks:
                logger.debug(f"No chunks extracted from {file_path}")
                return True  # Not an error, just empty file

            # Add chunks to database
            await self.database.add_chunks(chunks)

            # Update metadata after successful indexing
            metadata = self._load_index_metadata()
            metadata[str(file_path)] = os.path.getmtime(file_path)
            self._save_index_metadata(metadata)

            logger.debug(f"Indexed {len(chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            raise ParsingError(f"Failed to index file {file_path}: {e}") from e

    async def reindex_file(self, file_path: Path) -> bool:
        """Reindex a single file (removes existing chunks first).

        Args:
            file_path: Path to the file to reindex

        Returns:
            True if file was successfully reindexed
        """
        return await self.index_file(file_path, force_reindex=True)

    async def remove_file(self, file_path: Path) -> int:
        """Remove all chunks for a file from the index.

        Args:
            file_path: Path to the file to remove

        Returns:
            Number of chunks removed
        """
        try:
            count = await self.database.delete_by_file(file_path)
            logger.debug(f"Removed {count} chunks for {file_path}")
            return count
        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {e}")
            return 0

    def _find_indexable_files(self) -> list[Path]:
        """Find all files that should be indexed.

        Returns:
            List of file paths to index
        """
        indexable_files = []

        for file_path in self.project_root.rglob("*"):
            if self._should_index_file(file_path):
                indexable_files.append(file_path)

        return sorted(indexable_files)

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed.

        Args:
            file_path: Path to check

        Returns:
            True if file should be indexed
        """
        # Must be a file
        if not file_path.is_file():
            return False

        # Check file extension
        if file_path.suffix.lower() not in self.file_extensions:
            return False

        # Check if path should be ignored
        if self._should_ignore_path(file_path):
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

    def _should_ignore_path(self, file_path: Path) -> bool:
        """Check if a path should be ignored.

        Args:
            file_path: Path to check

        Returns:
            True if path should be ignored
        """
        try:
            # First check gitignore rules if available
            if self.gitignore_parser and self.gitignore_parser.is_ignored(file_path):
                logger.debug(f"Path ignored by .gitignore: {file_path}")
                return True

            # Get relative path from project root
            relative_path = file_path.relative_to(self.project_root)

            # Check each part of the path against default ignore patterns
            for part in relative_path.parts:
                if part in self._ignore_patterns:
                    logger.debug(f"Path ignored by default pattern '{part}': {file_path}")
                    return True

            # Check if any parent directory should be ignored
            for parent in relative_path.parents:
                for part in parent.parts:
                    if part in self._ignore_patterns:
                        logger.debug(f"Path ignored by parent pattern '{part}': {file_path}")
                        return True

            return False

        except ValueError:
            # Path is not relative to project root
            return True

    async def _parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a file into code chunks.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of code chunks
        """
        try:
            # Get appropriate parser
            parser = self.parser_registry.get_parser_for_file(file_path)

            # Parse file
            chunks = await parser.parse_file(file_path)

            # Filter out empty chunks
            valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]

            return valid_chunks

        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            raise ParsingError(f"Failed to parse file {file_path}: {e}") from e

    def add_ignore_pattern(self, pattern: str) -> None:
        """Add a pattern to ignore during indexing.

        Args:
            pattern: Pattern to ignore (directory or file name)
        """
        self._ignore_patterns.add(pattern)

    def remove_ignore_pattern(self, pattern: str) -> None:
        """Remove an ignore pattern.

        Args:
            pattern: Pattern to remove
        """
        self._ignore_patterns.discard(pattern)

    def get_ignore_patterns(self) -> set[str]:
        """Get current ignore patterns.

        Returns:
            Set of ignore patterns
        """
        return self._ignore_patterns.copy()

    async def get_indexing_stats(self) -> dict:
        """Get statistics about the indexing process.

        Returns:
            Dictionary with indexing statistics
        """
        try:
            # Get database stats
            db_stats = await self.database.get_stats()

            # Count indexable files
            indexable_files = self._find_indexable_files()

            return {
                "total_indexable_files": len(indexable_files),
                "indexed_files": db_stats.total_files,
                "total_chunks": db_stats.total_chunks,
                "languages": db_stats.languages,
                "file_extensions": list(self.file_extensions),
                "ignore_patterns": list(self._ignore_patterns),
                "parser_info": self.parser_registry.get_parser_info(),
            }

        except Exception as e:
            logger.error(f"Failed to get indexing stats: {e}")
            return {
                "error": str(e),
                "total_indexable_files": 0,
                "indexed_files": 0,
                "total_chunks": 0,
            }

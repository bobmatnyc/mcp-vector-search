"""Semantic indexer for MCP Vector Search."""

import asyncio
from pathlib import Path
from typing import List, Optional, Set

from loguru import logger

from ..config.defaults import DEFAULT_IGNORE_PATTERNS
from ..parsers.registry import get_parser_registry
from .database import VectorDatabase
from .exceptions import ParsingError
from .models import CodeChunk


class SemanticIndexer:
    """Semantic indexer for parsing and indexing code files."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        file_extensions: List[str],
    ) -> None:
        """Initialize semantic indexer.
        
        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index
        """
        self.database = database
        self.project_root = project_root
        self.file_extensions = set(ext.lower() for ext in file_extensions)
        self.parser_registry = get_parser_registry()
        self._ignore_patterns = set(DEFAULT_IGNORE_PATTERNS)

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
        files_to_index = self._find_indexable_files()
        
        if not files_to_index:
            logger.warning("No indexable files found")
            return 0
        
        logger.info(f"Found {len(files_to_index)} files to index")
        
        # Index files
        indexed_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(files_to_index):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Indexing progress: {i + 1}/{len(files_to_index)}")
            
            try:
                success = await self.index_file(file_path, force_reindex)
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                failed_count += 1
        
        logger.info(
            f"Indexing complete: {indexed_count} files indexed, {failed_count} failed"
        )
        
        return indexed_count

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
            
            # Remove existing chunks for this file if reindexing
            if force_reindex:
                await self.database.delete_by_file(file_path)
            
            # Parse file into chunks
            chunks = await self._parse_file(file_path)
            
            if not chunks:
                logger.debug(f"No chunks extracted from {file_path}")
                return True  # Not an error, just empty file
            
            # Add chunks to database
            await self.database.add_chunks(chunks)
            
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

    def _find_indexable_files(self) -> List[Path]:
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
            # Get relative path from project root
            relative_path = file_path.relative_to(self.project_root)
            
            # Check each part of the path
            for part in relative_path.parts:
                if part in self._ignore_patterns:
                    return True
            
            # Check if any parent directory should be ignored
            for parent in relative_path.parents:
                for part in parent.parts:
                    if part in self._ignore_patterns:
                        return True
            
            return False
            
        except ValueError:
            # Path is not relative to project root
            return True

    async def _parse_file(self, file_path: Path) -> List[CodeChunk]:
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

    def get_ignore_patterns(self) -> Set[str]:
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

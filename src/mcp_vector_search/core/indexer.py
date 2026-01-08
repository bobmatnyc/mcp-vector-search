"""Semantic indexer for MCP Vector Search."""

import asyncio
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from packaging import version

from .. import __version__
from ..analysis.collectors.base import MetricCollector
from ..analysis.metrics import ChunkMetrics
from ..analysis.trends import TrendTracker
from ..config.defaults import ALLOWED_DOTFILES, DEFAULT_IGNORE_PATTERNS
from ..config.settings import ProjectConfig
from ..parsers.registry import get_parser_registry
from ..utils.gitignore import create_gitignore_parser
from ..utils.monorepo import MonorepoDetector
from .database import VectorDatabase
from .directory_index import DirectoryIndex
from .exceptions import ParsingError
from .models import CodeChunk, IndexStats
from .relationships import RelationshipStore

# Extension to language mapping for metric collection
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
}


def _parse_file_standalone(
    args: tuple[Path, str | None],
) -> tuple[Path, list[CodeChunk], Exception | None]:
    """Parse a single file - standalone function for multiprocessing.

    This function must be at module level (not a method) to be picklable for
    multiprocessing. It creates its own parser registry to avoid serialization issues.

    Args:
        args: Tuple of (file_path, subproject_info_json)
            - file_path: Path to the file to parse
            - subproject_info_json: JSON string with subproject info or None

    Returns:
        Tuple of (file_path, chunks, error)
        - file_path: The file path that was parsed
        - chunks: List of parsed CodeChunk objects (empty if error)
        - error: Exception if parsing failed, None if successful
    """
    file_path, subproject_info_json = args

    try:
        # Create parser registry in this process
        parser_registry = get_parser_registry()

        # Get appropriate parser
        parser = parser_registry.get_parser_for_file(file_path)

        # Parse file synchronously (tree-sitter is synchronous anyway)
        # We need to use the synchronous version of parse_file
        # Since parsers may have async methods, we'll read and parse directly
        import asyncio

        # Create event loop for this process if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async parse_file in this process's event loop
        chunks = loop.run_until_complete(parser.parse_file(file_path))

        # Filter out empty chunks
        valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]

        # Apply subproject information if available
        if subproject_info_json:
            subproject_info = json.loads(subproject_info_json)
            for chunk in valid_chunks:
                chunk.subproject_name = subproject_info.get("name")
                chunk.subproject_path = subproject_info.get("relative_path")

        return (file_path, valid_chunks, None)

    except Exception as e:
        # Return error instead of raising to avoid process crashes
        logger.error(f"Failed to parse file {file_path} in worker process: {e}")
        return (file_path, [], e)


class SemanticIndexer:
    """Semantic indexer for parsing and indexing code files."""

    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        file_extensions: list[str] | None = None,
        config: ProjectConfig | None = None,
        max_workers: int | None = None,
        batch_size: int = 10,
        debug: bool = False,
        collectors: list[MetricCollector] | None = None,
        use_multiprocessing: bool = True,
    ) -> None:
        """Initialize semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory
            file_extensions: File extensions to index (deprecated, use config)
            config: Project configuration (preferred over file_extensions)
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            batch_size: Number of files to process in each batch
            debug: Enable debug output for hierarchy building
            collectors: Metric collectors to run during indexing (defaults to all complexity collectors)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
        """
        self.database = database
        self.project_root = project_root

        # Store config for filtering behavior
        self.config = config

        # Handle backward compatibility: use config.file_extensions or fallback to parameter
        if config is not None:
            self.file_extensions = {ext.lower() for ext in config.file_extensions}
        elif file_extensions is not None:
            self.file_extensions = {ext.lower() for ext in file_extensions}
        else:
            raise ValueError("Either config or file_extensions must be provided")

        self.parser_registry = get_parser_registry()
        self._ignore_patterns = set(DEFAULT_IGNORE_PATTERNS)
        self.debug = debug

        # Initialize metric collectors
        self.collectors = (
            collectors if collectors is not None else self._default_collectors()
        )

        # Configure multiprocessing for parallel parsing
        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            # Use 75% of CPU cores for parsing, but cap at 8 to avoid overhead
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max_workers or min(max(1, int(cpu_count * 0.75)), 8)
            logger.debug(
                f"Multiprocessing enabled with {self.max_workers} workers (CPU count: {cpu_count})"
            )
        else:
            self.max_workers = 1
            logger.debug("Multiprocessing disabled (single-threaded mode)")

        self.batch_size = batch_size
        self._index_metadata_file = (
            project_root / ".mcp-vector-search" / "index_metadata.json"
        )

        # Add cache for indexable files to avoid repeated filesystem scans
        self._indexable_files_cache: list[Path] | None = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60.0  # 60 second TTL

        # Initialize gitignore parser (only if respect_gitignore is True)
        if config is None or config.respect_gitignore:
            try:
                self.gitignore_parser = create_gitignore_parser(project_root)
                logger.debug(
                    f"Loaded {len(self.gitignore_parser.patterns)} gitignore patterns"
                )
            except Exception as e:
                logger.warning(f"Failed to load gitignore patterns: {e}")
                self.gitignore_parser = None
        else:
            self.gitignore_parser = None
            logger.debug("Gitignore filtering disabled by configuration")

        # Initialize monorepo detector
        self.monorepo_detector = MonorepoDetector(project_root)
        if self.monorepo_detector.is_monorepo():
            subprojects = self.monorepo_detector.detect_subprojects()
            logger.info(f"Detected monorepo with {len(subprojects)} subprojects")
            for sp in subprojects:
                logger.debug(f"  - {sp.name} ({sp.relative_path})")

        # Initialize directory index
        self.directory_index = DirectoryIndex(
            project_root / ".mcp-vector-search" / "directory_index.json"
        )
        # Load existing directory index
        self.directory_index.load()

        # Initialize relationship store for pre-computing visualization relationships
        self.relationship_store = RelationshipStore(project_root)

        # Initialize trend tracker for historical metrics
        self.trend_tracker = TrendTracker(project_root)

    def _default_collectors(self) -> list[MetricCollector]:
        """Return default set of metric collectors.

        Returns:
            List of all complexity collectors (cognitive, cyclomatic, nesting, parameters, methods)
        """
        from ..analysis.collectors.complexity import (
            CognitiveComplexityCollector,
            CyclomaticComplexityCollector,
            MethodCountCollector,
            NestingDepthCollector,
            ParameterCountCollector,
        )

        return [
            CognitiveComplexityCollector(),
            CyclomaticComplexityCollector(),
            NestingDepthCollector(),
            ParameterCountCollector(),
            MethodCountCollector(),
        ]

    def _collect_metrics(
        self, chunk: CodeChunk, source_code: bytes, language: str
    ) -> ChunkMetrics | None:
        """Collect metrics for a code chunk.

        This is a simplified version that estimates metrics from chunk content
        without full TreeSitter traversal. Future implementation will use
        TreeSitter node traversal for accurate metric collection.

        Args:
            chunk: The parsed code chunk
            source_code: Raw source code bytes
            language: Programming language identifier

        Returns:
            ChunkMetrics for the chunk, or None if no metrics collected
        """
        # For now, create basic metrics from chunk content
        # TODO: Implement full TreeSitter traversal in Phase 2
        lines_of_code = chunk.line_count

        # Estimate complexity from simple heuristics
        content = chunk.content
        cognitive_complexity = self._estimate_cognitive_complexity(content)
        cyclomatic_complexity = self._estimate_cyclomatic_complexity(content)
        max_nesting_depth = self._estimate_nesting_depth(content)
        parameter_count = len(chunk.parameters) if chunk.parameters else 0

        metrics = ChunkMetrics(
            cognitive_complexity=cognitive_complexity,
            cyclomatic_complexity=cyclomatic_complexity,
            max_nesting_depth=max_nesting_depth,
            parameter_count=parameter_count,
            lines_of_code=lines_of_code,
        )

        return metrics

    def _estimate_cognitive_complexity(self, content: str) -> int:
        """Estimate cognitive complexity from content (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated cognitive complexity score
        """
        # Simple heuristic: count control flow keywords
        keywords = [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "case",
            "when",
        ]
        complexity = 0
        for keyword in keywords:
            complexity += content.count(f" {keyword} ")
            complexity += content.count(f"\t{keyword} ")
            complexity += content.count(f"\n{keyword} ")
        return complexity

    def _estimate_cyclomatic_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity from content (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated cyclomatic complexity score (minimum 1)
        """
        # Start with baseline of 1
        complexity = 1

        # Count decision points
        keywords = [
            "if",
            "elif",
            "for",
            "while",
            "case",
            "when",
            "&&",
            "||",
            "and",
            "or",
        ]
        for keyword in keywords:
            complexity += content.count(keyword)

        return complexity

    def _estimate_nesting_depth(self, content: str) -> int:
        """Estimate maximum nesting depth from indentation (simplified heuristic).

        Args:
            content: Code content

        Returns:
            Estimated maximum nesting depth
        """
        max_depth = 0
        for line in content.split("\n"):
            # Count leading whitespace (4 spaces or 1 tab = 1 level)
            leading = len(line) - len(line.lstrip())
            if "\t" in line[:leading]:
                depth = line[:leading].count("\t")
            else:
                depth = leading // 4
            max_depth = max(max_depth, depth)
        return max_depth

    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        skip_relationships: bool = False,
    ) -> int:
        """Index all files in the project.

        Args:
            force_reindex: Whether to reindex existing files
            show_progress: Whether to show progress information
            skip_relationships: Skip computing relationships for visualization (faster, but visualize will be slower)

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

            # Rebuild directory index from successfully indexed files
            try:
                logger.debug("Rebuilding directory index...")
                # We don't have chunk counts here, but we have file modification times
                # Build a simple stats dict with file mod times for recency tracking
                chunk_stats = {}
                for file_path in files_to_index:
                    try:
                        mtime = os.path.getmtime(file_path)
                        # For now, just track modification time
                        # Chunk counts will be aggregated from the database later if needed
                        chunk_stats[str(file_path)] = {
                            "modified": mtime,
                            "chunks": 1,  # Placeholder - real count from chunks
                        }
                    except OSError:
                        pass

                self.directory_index.rebuild_from_files(
                    files_to_index, self.project_root, chunk_stats=chunk_stats
                )
                self.directory_index.save()
                dir_stats = self.directory_index.get_stats()
                logger.info(
                    f"Directory index updated: {dir_stats['total_directories']} directories, "
                    f"{dir_stats['total_files']} files"
                )
            except Exception as e:
                logger.error(f"Failed to update directory index: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        logger.info(
            f"Indexing complete: {indexed_count} files indexed, {failed_count} failed"
        )

        # Mark relationships for background computation (unless skipped)
        # Default behavior: skip blocking computation, mark for background processing
        if not skip_relationships and indexed_count > 0:
            try:
                logger.info("Marking relationships for background computation...")
                # Get all chunks from database for relationship computation
                all_chunks = await self.database.get_all_chunks()

                if len(all_chunks) > 0:
                    # Mark for background computation (non-blocking)
                    await self.relationship_store.compute_and_store(
                        all_chunks, self.database, background=True
                    )
                    logger.info("✓ Relationships marked for background computation")
                    logger.info(
                        "  Use 'mcp-vector-search index relationships' to compute now or wait for background task"
                    )
                else:
                    logger.warning("No chunks found for relationship computation")
            except Exception as e:
                logger.warning(f"Failed to mark relationships: {e}")
                logger.debug("Visualization will compute relationships on demand")

        # Save trend snapshot after successful indexing
        if indexed_count > 0:
            try:
                logger.info("Saving metrics snapshot for trend tracking...")
                # Get database stats
                stats = await self.database.get_stats()
                # Get all chunks for detailed metrics
                all_chunks = await self.database.get_all_chunks()
                # Compute metrics from stats and chunks
                metrics = self.trend_tracker.compute_metrics_from_stats(
                    stats.to_dict(), all_chunks
                )
                # Save snapshot (updates today's entry if exists)
                self.trend_tracker.save_snapshot(metrics)
                logger.info(
                    f"✓ Saved trend snapshot: {metrics['total_files']} files, "
                    f"{metrics['total_chunks']} chunks, health score {metrics['health_score']}"
                )
            except Exception as e:
                logger.warning(f"Failed to save trend snapshot: {e}")

        return indexed_count

    async def _parse_and_prepare_file(
        self, file_path: Path, force_reindex: bool = False
    ) -> tuple[list[CodeChunk], dict[str, Any] | None]:
        """Parse file and prepare chunks with metrics (no database insertion).

        This method extracts the parsing and metric collection logic from index_file()
        to enable batch processing across multiple files.

        Args:
            file_path: Path to the file to parse
            force_reindex: Whether to force reindexing (always deletes existing chunks)

        Returns:
            Tuple of (chunks_with_hierarchy, chunk_metrics)

        Raises:
            ParsingError: If file parsing fails
        """
        # Check if file should be indexed
        if not self._should_index_file(file_path):
            return ([], None)

        # Always remove existing chunks when reindexing a file
        # This prevents duplicate chunks and ensures consistency
        await self.database.delete_by_file(file_path)

        # Parse file into chunks
        chunks = await self._parse_file(file_path)

        if not chunks:
            logger.debug(f"No chunks extracted from {file_path}")
            return ([], None)

        # Build hierarchical relationships between chunks
        chunks_with_hierarchy = self._build_chunk_hierarchy(chunks)

        # Debug: Check if hierarchy was built
        methods_with_parents = sum(
            1
            for c in chunks_with_hierarchy
            if c.chunk_type in ("method", "function") and c.parent_chunk_id
        )
        logger.debug(
            f"After hierarchy build: {methods_with_parents}/{len([c for c in chunks_with_hierarchy if c.chunk_type in ('method', 'function')])} methods have parents"
        )

        # Collect metrics for chunks (if collectors are enabled)
        chunk_metrics: dict[str, Any] | None = None
        if self.collectors:
            try:
                # Read source code
                source_code = file_path.read_bytes()

                # Detect language from file extension
                language = EXTENSION_TO_LANGUAGE.get(
                    file_path.suffix.lower(), "unknown"
                )

                # Collect metrics for each chunk
                chunk_metrics = {}
                for chunk in chunks_with_hierarchy:
                    metrics = self._collect_metrics(chunk, source_code, language)
                    if metrics:
                        chunk_metrics[chunk.chunk_id] = metrics.to_metadata()

                logger.debug(
                    f"Collected metrics for {len(chunk_metrics)} chunks from {file_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to collect metrics for {file_path}: {e}")
                chunk_metrics = None

        return (chunks_with_hierarchy, chunk_metrics)

    async def _process_file_batch(
        self, file_paths: list[Path], force_reindex: bool = False
    ) -> list[bool]:
        """Process a batch of files and accumulate chunks for batch embedding.

        This method processes multiple files in parallel (using multiprocessing for
        CPU-bound parsing) and then performs a single database insertion for all chunks,
        enabling efficient batch embedding generation.

        Args:
            file_paths: List of file paths to process
            force_reindex: Whether to force reindexing

        Returns:
            List of success flags for each file
        """
        all_chunks: list[CodeChunk] = []
        all_metrics: dict[str, Any] = {}
        file_to_chunks_map: dict[str, tuple[int, int]] = {}
        success_flags: list[bool] = []

        # Filter files that should be indexed and delete old chunks
        files_to_parse = []
        for file_path in file_paths:
            if not self._should_index_file(file_path):
                success_flags.append(True)  # Skipped file is not an error
                continue
            # Delete old chunks before parsing
            await self.database.delete_by_file(file_path)
            files_to_parse.append(file_path)

        if not files_to_parse:
            return success_flags

        # Parse files using multiprocessing if enabled
        if self.use_multiprocessing and len(files_to_parse) > 1:
            # Use ProcessPoolExecutor for CPU-bound parsing
            parse_results = await self._parse_files_multiprocess(files_to_parse)
        else:
            # Fall back to async processing (for single file or disabled multiprocessing)
            parse_results = await self._parse_files_async(files_to_parse)

        # Accumulate chunks from all successfully parsed files
        metadata = self._load_index_metadata()
        for file_path, chunks, error in parse_results:
            if error:
                logger.error(f"Failed to parse {file_path}: {error}")
                success_flags.append(False)
                continue

            if chunks:
                # Build hierarchy and collect metrics for parsed chunks
                chunks_with_hierarchy = self._build_chunk_hierarchy(chunks)

                # Collect metrics if enabled
                chunk_metrics = None
                if self.collectors:
                    try:
                        source_code = file_path.read_bytes()
                        language = EXTENSION_TO_LANGUAGE.get(
                            file_path.suffix.lower(), "unknown"
                        )
                        chunk_metrics = {}
                        for chunk in chunks_with_hierarchy:
                            metrics = self._collect_metrics(
                                chunk, source_code, language
                            )
                            if metrics:
                                chunk_metrics[chunk.chunk_id] = metrics.to_metadata()
                    except Exception as e:
                        logger.warning(
                            f"Failed to collect metrics for {file_path}: {e}"
                        )

                # Accumulate chunks
                start_idx = len(all_chunks)
                all_chunks.extend(chunks_with_hierarchy)
                end_idx = len(all_chunks)
                file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

                # Merge metrics
                if chunk_metrics:
                    all_metrics.update(chunk_metrics)

                # Update metadata for successfully parsed file
                metadata[str(file_path)] = os.path.getmtime(file_path)
                success_flags.append(True)
            else:
                # Empty file is not an error
                metadata[str(file_path)] = os.path.getmtime(file_path)
                success_flags.append(True)

        # Single database insertion for entire batch
        if all_chunks:
            logger.info(
                f"Batch inserting {len(all_chunks)} chunks from {len(file_paths)} files"
            )
            try:
                await self.database.add_chunks(all_chunks, metrics=all_metrics)
                logger.debug(
                    f"Successfully indexed {len(all_chunks)} chunks from {sum(success_flags)} files"
                )
            except Exception as e:
                logger.error(f"Failed to insert batch of chunks: {e}")
                # Mark all files in this batch as failed
                return [False] * len(file_paths)

        # Save updated metadata after successful batch
        self._save_index_metadata(metadata)

        return success_flags

    async def _parse_files_multiprocess(
        self, file_paths: list[Path]
    ) -> list[tuple[Path, list[CodeChunk], Exception | None]]:
        """Parse multiple files using multiprocessing for CPU-bound parallelism.

        Args:
            file_paths: List of file paths to parse

        Returns:
            List of tuples (file_path, chunks, error) for each file
        """
        # Prepare arguments for worker processes
        parse_args = []
        for file_path in file_paths:
            # Get subproject info if available
            subproject = self.monorepo_detector.get_subproject_for_file(file_path)
            subproject_info_json = None
            if subproject:
                subproject_info_json = json.dumps(
                    {
                        "name": subproject.name,
                        "relative_path": subproject.relative_path,
                    }
                )
            parse_args.append((file_path, subproject_info_json))

        # Limit workers to avoid overhead
        max_workers = min(self.max_workers, len(file_paths))

        # Run parsing in ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and wait for results
            results = await loop.run_in_executor(
                None, lambda: list(executor.map(_parse_file_standalone, parse_args))
            )

        logger.debug(
            f"Multiprocess parsing completed: {len(results)} files parsed with {max_workers} workers"
        )
        return results

    async def _parse_files_async(
        self, file_paths: list[Path]
    ) -> list[tuple[Path, list[CodeChunk], Exception | None]]:
        """Parse multiple files using async (fallback for single file or disabled multiprocessing).

        Args:
            file_paths: List of file paths to parse

        Returns:
            List of tuples (file_path, chunks, error) for each file
        """
        results = []
        for file_path in file_paths:
            try:
                chunks = await self._parse_file(file_path)
                results.append((file_path, chunks, None))
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                results.append((file_path, [], e))

        return results

    def _load_index_metadata(self) -> dict[str, float]:
        """Load file modification times from metadata file.

        Returns:
            Dictionary mapping file paths to modification times
        """
        if not self._index_metadata_file.exists():
            return {}

        try:
            with open(self._index_metadata_file) as f:
                data = json.load(f)
                # Handle legacy format (just file_mtimes dict) and new format
                if "file_mtimes" in data:
                    return data["file_mtimes"]
                else:
                    # Legacy format - just return as-is
                    return data
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

            # New metadata format with version tracking
            data = {
                "index_version": __version__,
                "indexed_at": datetime.now(UTC).isoformat(),
                "file_mtimes": metadata,
            }

            with open(self._index_metadata_file, "w") as f:
                json.dump(data, f, indent=2)
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

            # Build hierarchical relationships between chunks
            chunks_with_hierarchy = self._build_chunk_hierarchy(chunks)

            # Debug: Check if hierarchy was built
            methods_with_parents = sum(
                1
                for c in chunks_with_hierarchy
                if c.chunk_type in ("method", "function") and c.parent_chunk_id
            )
            logger.debug(
                f"After hierarchy build: {methods_with_parents}/{len([c for c in chunks_with_hierarchy if c.chunk_type in ('method', 'function')])} methods have parents"
            )

            # Collect metrics for chunks (if collectors are enabled)
            chunk_metrics: dict[str, Any] | None = None
            if self.collectors:
                try:
                    # Read source code
                    source_code = file_path.read_bytes()

                    # Detect language from file extension
                    language = EXTENSION_TO_LANGUAGE.get(
                        file_path.suffix.lower(), "unknown"
                    )

                    # Collect metrics for each chunk
                    chunk_metrics = {}
                    for chunk in chunks_with_hierarchy:
                        metrics = self._collect_metrics(chunk, source_code, language)
                        if metrics:
                            chunk_metrics[chunk.chunk_id] = metrics.to_metadata()

                    logger.debug(
                        f"Collected metrics for {len(chunk_metrics)} chunks from {file_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for {file_path}: {e}")
                    chunk_metrics = None

            # Add chunks to database with metrics
            await self.database.add_chunks(chunks_with_hierarchy, metrics=chunk_metrics)

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
        indexable_files = self._scan_files_sync()

        self._indexable_files_cache = sorted(indexable_files)
        self._cache_timestamp = current_time
        logger.debug(f"Rebuilt indexable files cache ({len(indexable_files)} files)")

        return self._indexable_files_cache

    def _scan_files_sync(self) -> list[Path]:
        """Synchronous file scanning (runs in thread pool).

        Uses os.walk with directory filtering to avoid traversing ignored directories.

        Returns:
            List of indexable file paths
        """
        indexable_files = []

        # Use os.walk for efficient directory traversal with early filtering
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)

            # Filter out ignored directories IN-PLACE to prevent os.walk from traversing them
            # This is much more efficient than checking every file in ignored directories
            # PERFORMANCE: Pass is_directory=True hint to skip filesystem stat() calls
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore_path(root_path / d, is_directory=True)
            ]

            # Check each file in the current directory
            # PERFORMANCE: skip_file_check=True because os.walk guarantees these are files
            for filename in files:
                file_path = root_path / filename
                if self._should_index_file(file_path, skip_file_check=True):
                    indexable_files.append(file_path)

        return indexable_files

    async def _find_indexable_files_async(self) -> list[Path]:
        """Find all files asynchronously without blocking event loop.

        Returns:
            List of file paths to index
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

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
                executor, self._scan_files_sync
            )

        # Update cache
        self._indexable_files_cache = sorted(indexable_files)
        self._cache_timestamp = current_time
        logger.debug(f"Found {len(indexable_files)} indexable files")

        return self._indexable_files_cache

    def _should_index_file(
        self, file_path: Path, skip_file_check: bool = False
    ) -> bool:
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
        if self._should_ignore_path(file_path, is_directory=False):
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

    def _should_ignore_path(
        self, file_path: Path, is_directory: bool | None = None
    ) -> bool:
        """Check if a path should be ignored.

        Args:
            file_path: Path to check
            is_directory: Optional hint if path is a directory (avoids filesystem check)

        Returns:
            True if path should be ignored
        """
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
                        return True

            # 2. Check gitignore rules if available and enabled
            # PERFORMANCE: Pass is_directory hint to avoid redundant stat() calls
            if self.config and self.config.respect_gitignore:
                if self.gitignore_parser and self.gitignore_parser.is_ignored(
                    file_path, is_directory=is_directory
                ):
                    logger.debug(f"Path ignored by .gitignore: {file_path}")
                    return True

            # 3. Check each part of the path against default ignore patterns
            for part in relative_path.parts:
                if part in self._ignore_patterns:
                    logger.debug(
                        f"Path ignored by default pattern '{part}': {file_path}"
                    )
                    return True

            # 4. Check if any parent directory should be ignored
            for parent in relative_path.parents:
                for part in parent.parts:
                    if part in self._ignore_patterns:
                        logger.debug(
                            f"Path ignored by parent pattern '{part}': {file_path}"
                        )
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
            List of code chunks with subproject information
        """
        try:
            # Get appropriate parser
            parser = self.parser_registry.get_parser_for_file(file_path)

            # Parse file
            chunks = await parser.parse_file(file_path)

            # Filter out empty chunks
            valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]

            # Assign subproject information for monorepos
            subproject = self.monorepo_detector.get_subproject_for_file(file_path)
            if subproject:
                for chunk in valid_chunks:
                    chunk.subproject_name = subproject.name
                    chunk.subproject_path = subproject.relative_path

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

    def get_index_version(self) -> str | None:
        """Get the version of the tool that created the current index.

        Returns:
            Version string or None if not available
        """
        if not self._index_metadata_file.exists():
            return None

        try:
            with open(self._index_metadata_file) as f:
                data = json.load(f)
                return data.get("index_version")
        except Exception as e:
            logger.warning(f"Failed to read index version: {e}")
            return None

    def needs_reindex_for_version(self) -> bool:
        """Check if reindex is needed due to version upgrade.

        Returns:
            True if reindex is needed for version compatibility
        """
        index_version = self.get_index_version()

        if not index_version:
            # No version recorded - this is either a new index or legacy format
            # Reindex to establish version tracking
            return True

        try:
            current = version.parse(__version__)
            indexed = version.parse(index_version)

            # Reindex on major or minor version change
            # Patch versions (0.5.1 -> 0.5.2) don't require reindex
            needs_reindex = (
                current.major != indexed.major or current.minor != indexed.minor
            )

            if needs_reindex:
                logger.info(
                    f"Version upgrade detected: {index_version} -> {__version__} "
                    f"(reindex recommended)"
                )

            return needs_reindex

        except Exception as e:
            logger.warning(f"Failed to compare versions: {e}")
            # If we can't parse versions, be safe and reindex
            return True

    async def get_indexing_stats(self, db_stats: IndexStats | None = None) -> dict:
        """Get statistics about the indexing process.

        Args:
            db_stats: Optional pre-fetched database stats to avoid duplicate queries

        Returns:
            Dictionary with indexing statistics

        Note:
            Uses database statistics only for performance on large projects.
            Filesystem scanning would timeout on 100K+ file projects.
            Pass db_stats parameter to avoid calling database.get_stats() twice.
        """
        try:
            # Get database stats if not provided (fast, no filesystem scan)
            if db_stats is None:
                db_stats = await self.database.get_stats()

            # Use database stats for all file counts
            # This avoids expensive filesystem scans on large projects
            return {
                "total_indexable_files": db_stats.total_files,
                "indexed_files": db_stats.total_files,
                "total_files": db_stats.total_files,  # For backward compatibility
                "total_chunks": db_stats.total_chunks,
                "languages": db_stats.languages,
                "file_types": db_stats.file_types,  # Include file type distribution
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
                "total_files": 0,
                "total_chunks": 0,
            }

    async def get_files_to_index(
        self, force_reindex: bool = False
    ) -> tuple[list[Path], list[Path]]:
        """Get all indexable files and those that need indexing.

        Args:
            force_reindex: Whether to force reindex of all files

        Returns:
            Tuple of (all_indexable_files, files_to_index)
        """
        # Find all indexable files
        all_files = await self._find_indexable_files_async()

        if not all_files:
            return [], []

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

        return all_files, files_to_index

    async def index_files_with_progress(
        self,
        files_to_index: list[Path],
        force_reindex: bool = False,
    ):
        """Index files and yield progress updates for each file.

        This method processes files in batches and accumulates chunks across files
        before performing a single database insertion per batch for better performance.

        Args:
            files_to_index: List of file paths to index
            force_reindex: Whether to force reindexing

        Yields:
            Tuple of (file_path, chunks_added, success) for each processed file
        """
        # Write version header to error log at start of indexing run
        self._write_indexing_run_header()

        # Process files in batches for better memory management and embedding efficiency
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]

            # Accumulate chunks from all files in batch
            all_chunks: list[CodeChunk] = []
            all_metrics: dict[str, Any] = {}
            file_to_chunks_map: dict[str, tuple[int, int]] = {}
            file_results: dict[Path, tuple[int, bool]] = {}

            # Parse all files in parallel
            tasks = []
            for file_path in batch:
                task = asyncio.create_task(
                    self._parse_and_prepare_file(file_path, force_reindex)
                )
                tasks.append(task)

            parse_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Accumulate chunks from successfully parsed files
            metadata = self._load_index_metadata()
            for file_path, result in zip(batch, parse_results, strict=True):
                if isinstance(result, Exception):
                    error_msg = f"Failed to index file {file_path}: {type(result).__name__}: {str(result)}"
                    logger.error(error_msg)
                    file_results[file_path] = (0, False)

                    # Save error to error log file
                    try:
                        error_log_path = (
                            self.project_root
                            / ".mcp-vector-search"
                            / "indexing_errors.log"
                        )
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            timestamp = datetime.now().isoformat()
                            f.write(f"[{timestamp}] {error_msg}\n")
                    except Exception as log_err:
                        logger.debug(f"Failed to write error log: {log_err}")
                    continue

                chunks, metrics = result
                if chunks:
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    end_idx = len(all_chunks)
                    file_to_chunks_map[str(file_path)] = (start_idx, end_idx)

                    # Merge metrics
                    if metrics:
                        all_metrics.update(metrics)

                    # Update metadata for successfully parsed file
                    metadata[str(file_path)] = os.path.getmtime(file_path)
                    file_results[file_path] = (len(chunks), True)
                    logger.debug(f"Prepared {len(chunks)} chunks from {file_path}")
                else:
                    # Empty file is not an error
                    metadata[str(file_path)] = os.path.getmtime(file_path)
                    file_results[file_path] = (0, True)

            # Single database insertion for entire batch
            if all_chunks:
                logger.info(
                    f"Batch inserting {len(all_chunks)} chunks from {len(batch)} files"
                )
                try:
                    await self.database.add_chunks(all_chunks, metrics=all_metrics)
                    logger.debug(
                        f"Successfully indexed {len(all_chunks)} chunks from batch"
                    )
                except Exception as e:
                    error_msg = f"Failed to insert batch of chunks: {e}"
                    logger.error(error_msg)
                    # Mark all files with chunks in this batch as failed
                    for file_path in file_to_chunks_map.keys():
                        file_results[Path(file_path)] = (0, False)

                    # Save error to error log file
                    try:
                        error_log_path = (
                            self.project_root
                            / ".mcp-vector-search"
                            / "indexing_errors.log"
                        )
                        with open(error_log_path, "a", encoding="utf-8") as f:
                            timestamp = datetime.now().isoformat()
                            f.write(f"[{timestamp}] {error_msg}\n")
                    except Exception as log_err:
                        logger.debug(f"Failed to write error log: {log_err}")

            # Save metadata after batch
            self._save_index_metadata(metadata)

            # Yield progress updates for each file in batch
            for file_path in batch:
                chunks_added, success = file_results.get(file_path, (0, False))
                yield (file_path, chunks_added, success)

    def _build_chunk_hierarchy(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Build parent-child relationships between chunks.

        Logic:
        - Module chunks (chunk_type="module") have depth 0
        - Class chunks have depth 1, parent is module
        - Method chunks have depth 2, parent is class
        - Function chunks outside classes have depth 1, parent is module
        - Nested classes increment depth

        Args:
            chunks: List of code chunks to process

        Returns:
            List of chunks with hierarchy relationships established
        """
        if not chunks:
            return chunks

        # Group chunks by type and name
        # Only actual module chunks (not imports) serve as parents for top-level code
        # imports chunks should remain siblings of classes/functions, not parents
        module_chunks = [c for c in chunks if c.chunk_type == "module"]
        class_chunks = [
            c for c in chunks if c.chunk_type in ("class", "interface", "mixin")
        ]
        function_chunks = [
            c for c in chunks if c.chunk_type in ("function", "method", "constructor")
        ]

        # DEBUG: Print what we have (if debug enabled)
        if self.debug:
            import sys

            print(
                f"\n[DEBUG] Building hierarchy: {len(module_chunks)} modules, {len(class_chunks)} classes, {len(function_chunks)} functions",
                file=sys.stderr,
            )
            if class_chunks:
                print(
                    f"[DEBUG] Class names: {[c.class_name for c in class_chunks[:5]]}",
                    file=sys.stderr,
                )
            if function_chunks:
                print(
                    f"[DEBUG] First 5 functions with class_name: {[(f.function_name, f.class_name) for f in function_chunks[:5]]}",
                    file=sys.stderr,
                )

        # Build relationships
        for func in function_chunks:
            if func.class_name:
                # Find parent class
                parent_class = next(
                    (c for c in class_chunks if c.class_name == func.class_name), None
                )
                if parent_class:
                    func.parent_chunk_id = parent_class.chunk_id
                    func.chunk_depth = parent_class.chunk_depth + 1
                    if func.chunk_id not in parent_class.child_chunk_ids:
                        parent_class.child_chunk_ids.append(func.chunk_id)
                    if self.debug:
                        import sys

                        print(
                            f"[DEBUG] ✓ Linked '{func.function_name}' to class '{parent_class.class_name}'",
                            file=sys.stderr,
                        )
                    logger.debug(
                        f"Linked method '{func.function_name}' (ID: {func.chunk_id[:8]}) to class '{parent_class.class_name}' (ID: {parent_class.chunk_id[:8]})"
                    )
            else:
                # Top-level function
                if not func.chunk_depth:
                    func.chunk_depth = 1
                # Link to module if exists
                if module_chunks and not func.parent_chunk_id:
                    func.parent_chunk_id = module_chunks[0].chunk_id
                    if func.chunk_id not in module_chunks[0].child_chunk_ids:
                        module_chunks[0].child_chunk_ids.append(func.chunk_id)

        for cls in class_chunks:
            # Classes without parent are top-level (depth 1)
            if not cls.chunk_depth:
                cls.chunk_depth = 1
            # Link to module if exists
            if module_chunks and not cls.parent_chunk_id:
                cls.parent_chunk_id = module_chunks[0].chunk_id
                if cls.chunk_id not in module_chunks[0].child_chunk_ids:
                    module_chunks[0].child_chunk_ids.append(cls.chunk_id)

        # Module chunks stay at depth 0
        for mod in module_chunks:
            if not mod.chunk_depth:
                mod.chunk_depth = 0

        # DEBUG: Print summary
        if self.debug:
            import sys

            funcs_with_parents = sum(1 for f in function_chunks if f.parent_chunk_id)
            classes_with_parents = sum(1 for c in class_chunks if c.parent_chunk_id)
            print(
                f"[DEBUG] Hierarchy built: {funcs_with_parents}/{len(function_chunks)} functions linked, {classes_with_parents}/{len(class_chunks)} classes linked\n",
                file=sys.stderr,
            )

        return chunks

    def _write_indexing_run_header(self) -> None:
        """Write version and timestamp header to error log at start of indexing run."""
        try:
            error_log_path = (
                self.project_root / ".mcp-vector-search" / "indexing_errors.log"
            )
            error_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(error_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now(UTC).isoformat()
                separator = "=" * 80
                f.write(f"\n{separator}\n")
                f.write(
                    f"[{timestamp}] Indexing run started - mcp-vector-search v{__version__}\n"
                )
                f.write(f"{separator}\n")
        except Exception as e:
            logger.debug(f"Failed to write indexing run header: {e}")

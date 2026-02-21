"""Chunk parsing and processing for semantic indexing."""

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import orjson
from loguru import logger

from ..parsers.registry import ParserRegistry, get_parser_registry
from ..utils.monorepo import MonorepoDetector
from .exceptions import ParsingError
from .git_blame import GitBlameCache
from .models import CodeChunk


def _detect_optimal_workers() -> int:
    """Detect optimal worker count based on CPU architecture.

    Returns:
        Optimal number of worker processes:
        - Apple Silicon (16+ cores): 14 workers (leave 2 for system)
        - Apple Silicon (10-15 cores): 10 workers (leave headroom)
        - Apple Silicon (<10 cores): cores - 1
        - Other architectures: 75% of CPU cores

    Environment Variables:
        MCP_VECTOR_SEARCH_MAX_WORKERS: Override auto-detection
    """
    env_workers = os.environ.get("MCP_VECTOR_SEARCH_MAX_WORKERS")
    if env_workers:
        return int(env_workers)

    cpu_count = multiprocessing.cpu_count()

    try:
        import platform

        # Check if Apple Silicon (ARM64 on macOS)
        if platform.processor() == "arm" and platform.system() == "Darwin":
            # M4 Max: 12 perf + 4 efficiency = 16 cores
            # Use 14 workers (leave 2 for system + background tasks)
            if cpu_count >= 16:
                workers = min(14, cpu_count - 2)
                logger.debug(
                    f"Apple Silicon detected ({cpu_count} cores): using {workers} workers (M4 Max/Ultra optimized)"
                )
                return workers
            elif cpu_count >= 10:
                workers = min(10, cpu_count - 2)
                logger.debug(
                    f"Apple Silicon detected ({cpu_count} cores): using {workers} workers"
                )
                return workers
            else:
                workers = max(4, cpu_count - 1)
                logger.debug(
                    f"Apple Silicon detected ({cpu_count} cores): using {workers} workers"
                )
                return workers
    except Exception as e:
        logger.debug(
            f"CPU architecture detection failed: {e}, using default worker calculation"
        )

    # Default: 75% of cores
    workers = max(1, cpu_count * 3 // 4)
    logger.debug(f"Using {workers} workers ({cpu_count} CPU cores detected)")
    return workers


def _deduplicate_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Remove duplicate chunks based on their unique ID.

    Some parsers (especially tree-sitter on minified files) may extract the same
    code region multiple times during AST traversal. This function ensures each
    unique chunk appears only once.

    Uses chunk_id (which includes content hash) to properly handle minified files
    where multiple functions on the same line would have the same file:line:line ID.

    Args:
        chunks: List of chunks that may contain duplicates

    Returns:
        List of unique chunks (preserving order of first occurrence)
    """
    if not chunks:
        return chunks

    seen_ids: set[str] = set()
    unique_chunks: list[CodeChunk] = []

    for chunk in chunks:
        # Use chunk_id (includes content hash) for proper deduplication
        chunk_key = chunk.chunk_id or chunk.id
        if chunk_key not in seen_ids:
            seen_ids.add(chunk_key)
            unique_chunks.append(chunk)

    removed_count = len(chunks) - len(unique_chunks)
    if removed_count > 0:
        # Use TRACE level (below DEBUG) to avoid cluttering progress displays
        # This log fires per-file during indexing and breaks Rich progress bars
        logger.trace(
            f"Removed {removed_count} duplicate chunks (kept {len(unique_chunks)})"
        )

    return unique_chunks


def _parse_file_standalone(
    args: tuple[Path, str | None],
) -> tuple[Path, list[CodeChunk], Exception | None]:
    """Parse a single file - standalone function for multiprocessing.

    This function must be at module level (not a method) to be picklable for
    multiprocessing. It creates its own parser registry to avoid serialization issues.

    OPTIMIZATION: Uses synchronous parse_file_sync() to avoid async overhead,
    since tree-sitter operations are CPU-bound and synchronous anyway.

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
        # Create parser registry in this process (lazy loading, only creates needed parser)
        parser_registry = get_parser_registry()

        # Get appropriate parser (lazy instantiation)
        parser = parser_registry.get_parser_for_file(file_path)

        # Parse file synchronously - no async/event loop overhead!
        # Parser grammar is loaded lazily on first actual parse
        chunks = parser.parse_file_sync(file_path)

        # Filter out empty chunks
        valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]

        # Deduplicate chunks (some parsers may extract same region multiple times)
        valid_chunks = _deduplicate_chunks(valid_chunks)

        # Apply subproject information if available
        if subproject_info_json:
            subproject_info = orjson.loads(subproject_info_json)
            for chunk in valid_chunks:
                chunk.subproject_name = subproject_info.get("name")
                chunk.subproject_path = subproject_info.get("relative_path")

        return (file_path, valid_chunks, None)

    except FileNotFoundError:
        # File was deleted during indexing - this is not an error
        logger.warning(f"Skipping deleted file: {file_path}")
        return (file_path, [], None)
    except Exception as e:
        # Return error instead of raising to avoid process crashes
        logger.error(f"Failed to parse file {file_path} in worker process: {e}")
        return (file_path, [], e)


class ChunkProcessor:
    """Handles parsing files into code chunks and building hierarchies.

    This class encapsulates all logic related to parsing files, managing
    multiprocessing for parallel parsing, and building hierarchical
    relationships between chunks.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        monorepo_detector: MonorepoDetector,
        max_workers: int | None = None,
        use_multiprocessing: bool = True,
        debug: bool = False,
        repo_root: Path | None = None,
    ) -> None:
        """Initialize chunk processor.

        Args:
            parser_registry: Parser registry for file parsing
            monorepo_detector: Monorepo detector for subproject information
            max_workers: Maximum number of worker processes for parallel parsing (ignored if use_multiprocessing=False)
            use_multiprocessing: Enable multiprocess parallel parsing (default: True, disable for debugging)
            debug: Enable debug output for hierarchy building
            repo_root: Git repository root for blame extraction (optional)
        """
        self.parser_registry = parser_registry
        self.monorepo_detector = monorepo_detector
        self.debug = debug

        # Configure multiprocessing for parallel parsing
        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            # Auto-detect optimal workers based on CPU architecture
            self.max_workers = max_workers or _detect_optimal_workers()
            logger.debug(f"Multiprocessing enabled with {self.max_workers} workers")
        else:
            self.max_workers = 1
            logger.debug("Multiprocessing disabled (single-threaded mode)")

        # Initialize git blame cache if repo root provided
        self.git_blame_cache = GitBlameCache(repo_root) if repo_root else None
        if self.git_blame_cache:
            logger.debug(f"Git blame tracking enabled for repo: {repo_root}")

    def parse_file_sync(self, file_path: Path) -> list[CodeChunk]:
        """Parse a file into code chunks synchronously (for asyncio.to_thread).

        This is the synchronous version used by the async wrapper to avoid
        blocking the event loop during CPU-bound parsing operations.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of code chunks with subproject information
        """
        try:
            # Get appropriate parser
            parser = self.parser_registry.get_parser_for_file(file_path)

            # Parse file synchronously (no event loop blocking)
            chunks = parser.parse_file_sync(file_path)

            # Filter out empty chunks
            valid_chunks = [chunk for chunk in chunks if chunk.content.strip()]

            # Deduplicate chunks (some parsers may extract same region multiple times)
            valid_chunks = _deduplicate_chunks(valid_chunks)

            # Assign subproject information for monorepos
            subproject = self.monorepo_detector.get_subproject_for_file(file_path)
            if subproject:
                for chunk in valid_chunks:
                    chunk.subproject_name = subproject.name
                    chunk.subproject_path = subproject.relative_path

            # Enrich chunks with git blame metadata
            if self.git_blame_cache:
                self._enrich_chunks_with_blame(file_path, valid_chunks)

            return valid_chunks

        except FileNotFoundError:
            # File was deleted during indexing - return empty list instead of raising
            logger.warning(f"Skipping deleted file: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            raise ParsingError(f"Failed to parse file {file_path}: {e}") from e

    async def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a file into code chunks (async wrapper for thread pool execution).

        In pipeline mode, this runs in a thread pool to avoid blocking the
        event loop during CPU-bound tree-sitter parsing operations.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of code chunks with subproject information
        """
        # Run synchronous parsing in thread pool to avoid blocking event loop
        return await asyncio.to_thread(self.parse_file_sync, file_path)

    async def parse_files_multiprocess(
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
                subproject_info_json = orjson.dumps(
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

    async def parse_files_async(
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
                chunks = await self.parse_file(file_path)
                results.append((file_path, chunks, None))
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                results.append((file_path, [], e))

        return results

    def _enrich_chunks_with_blame(
        self, file_path: Path, chunks: list[CodeChunk]
    ) -> None:
        """Enrich chunks with git blame metadata (in-place).

        Args:
            file_path: Path to file being processed
            chunks: List of chunks to enrich (modified in-place)
        """
        if not self.git_blame_cache:
            return

        for chunk in chunks:
            try:
                # Get blame for chunk's line range
                blame = self.git_blame_cache.get_blame_for_range(
                    file_path, chunk.start_line, chunk.end_line
                )

                if blame:
                    chunk.last_author = blame.author
                    chunk.last_modified = blame.timestamp
                    chunk.commit_hash = blame.commit_hash
            except Exception as e:
                # Non-fatal: log and continue without blame data
                # Use TRACE level to avoid cluttering progress displays
                logger.trace(
                    f"Failed to get blame for {file_path}:{chunk.start_line}-{chunk.end_line}: {e}"
                )

    def build_chunk_hierarchy(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
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
            c
            for c in chunks
            if c.chunk_type in ("class", "interface", "mixin", "widget")
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
                    # Use TRACE level to avoid cluttering progress displays
                    logger.trace(
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

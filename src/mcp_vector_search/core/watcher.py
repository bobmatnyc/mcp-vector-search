"""File system watcher for incremental indexing."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from pathlib import Path

from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..config.settings import ProjectConfig
from .database import VectorDatabase
from .indexer import SemanticIndexer


class CodeFileHandler(FileSystemEventHandler):
    """Handler for code file changes."""

    def __init__(
        self,
        file_extensions: list[str],
        ignore_patterns: list[str],
        callback: Callable[[str, str], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        debounce_delay: float = 1.0,
    ):
        """Initialize file handler.

        Args:
            file_extensions: List of file extensions to watch
            ignore_patterns: List of patterns to ignore
            callback: Async callback function for file changes
            loop: Event loop to schedule tasks on
            debounce_delay: Delay in seconds to debounce rapid changes
        """
        super().__init__()
        self.file_extensions = set(file_extensions)
        self.ignore_patterns = ignore_patterns
        self.callback = callback
        self.loop = loop
        self.debounce_delay = debounce_delay
        self.pending_changes: set[str] = set()
        self.last_change_time: float = 0
        self.debounce_task: asyncio.Task | Future | None = None

    def should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)

        # Check file extension
        if path.suffix not in self.file_extensions:
            return False

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path):
                return False

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_change(event.src_path, "modified")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_change(event.src_path, "created")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if not event.is_directory and self.should_process_file(event.src_path):
            self._schedule_change(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename."""
        if hasattr(event, "dest_path"):
            # Handle rename/move
            if not event.is_directory:
                if self.should_process_file(event.src_path):
                    self._schedule_change(event.src_path, "deleted")
                if self.should_process_file(event.dest_path):
                    self._schedule_change(event.dest_path, "created")

    def _schedule_change(self, file_path: str, change_type: str) -> None:
        """Schedule a file change for processing with debouncing."""
        self.pending_changes.add(f"{change_type}:{file_path}")
        self.last_change_time = time.time()

        # Cancel existing debounce task
        if self.debounce_task and not self.debounce_task.done():
            self.debounce_task.cancel()

        # Schedule new debounce task using the stored loop
        future = asyncio.run_coroutine_threadsafe(self._debounced_process(), self.loop)
        # Store the future as our task (it has a done() method)
        self.debounce_task = future

    async def _debounced_process(self) -> None:
        """Process pending changes after debounce delay."""
        await asyncio.sleep(self.debounce_delay)

        # Check if more changes occurred during debounce
        if time.time() - self.last_change_time < self.debounce_delay:
            return

        # Process all pending changes
        changes = self.pending_changes.copy()
        self.pending_changes.clear()

        for change in changes:
            change_type, file_path = change.split(":", 1)
            try:
                await self.callback(file_path, change_type)
            except Exception as e:
                logger.error(f"Error processing file change {file_path}: {e}")


class FileWatcher:
    """File system watcher for incremental indexing."""

    def __init__(
        self,
        project_root: Path,
        config: ProjectConfig,
        indexer: SemanticIndexer,
        database: VectorDatabase,
    ):
        """Initialize file watcher.

        Args:
            project_root: Root directory to watch
            config: Project configuration
            indexer: Semantic indexer instance
            database: Vector database instance
        """
        self.project_root = project_root
        self.config = config
        self.indexer = indexer
        self.database = database
        self.observer: Observer | None = None
        self.handler: CodeFileHandler | None = None
        self.is_running = False

    async def start(self) -> None:
        """Start watching for file changes."""
        if self.is_running:
            logger.warning("File watcher is already running")
            return

        logger.info(f"Starting file watcher for {self.project_root}")

        # Create handler
        loop = asyncio.get_running_loop()
        self.handler = CodeFileHandler(
            file_extensions=self.config.file_extensions,
            ignore_patterns=self._get_ignore_patterns(),
            callback=self._handle_file_change,
            loop=loop,
            debounce_delay=1.0,
        )

        # Create observer
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.project_root), recursive=True)

        # Start observer in a separate thread
        self.observer.start()
        self.is_running = True

        logger.info("File watcher started successfully")

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if not self.is_running:
            return

        logger.info("Stopping file watcher")

        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None

        self.handler = None
        self.is_running = False

        logger.info("File watcher stopped")

    def _get_ignore_patterns(self) -> list[str]:
        """Get patterns to ignore during watching."""
        default_patterns = [
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".DS_Store",
            "Thumbs.db",
            ".idea",
            ".vscode",
            "build",
            "dist",
            "target",
            ".mcp-vector-search",  # Ignore our own index directory
        ]

        # Add any custom ignore patterns from config
        # TODO: Add custom ignore patterns to config
        return default_patterns

    async def _handle_file_change(self, file_path: str, change_type: str) -> None:
        """Handle a file change event.

        Args:
            file_path: Path to the changed file
            change_type: Type of change (created, modified, deleted)
        """
        path = Path(file_path)
        logger.debug(f"Processing file change: {change_type} {path}")

        try:
            if change_type == "deleted":
                # Remove chunks for deleted file
                await self._remove_file_chunks(path)
            elif change_type in ("created", "modified"):
                # Re-index the file
                await self._reindex_file(path)

            logger.info(f"Processed {change_type} for {path.name}")

        except Exception as e:
            logger.error(f"Failed to process {change_type} for {path}: {e}")

    async def _remove_file_chunks(self, file_path: Path) -> None:
        """Remove all chunks for a deleted file."""
        # Get relative path for consistent IDs
        try:
            relative_path = file_path.relative_to(self.project_root)
        except ValueError:
            relative_path = file_path

        # Remove chunks from database
        await self.database.remove_file_chunks(str(relative_path))
        logger.debug(f"Removed chunks for deleted file: {relative_path}")

    async def _reindex_file(self, file_path: Path) -> None:
        """Re-index a single file."""
        if not file_path.exists():
            logger.warning(f"File no longer exists: {file_path}")
            return

        # Remove existing chunks first
        await self._remove_file_chunks(file_path)

        # Index the file
        chunks_indexed = await self.indexer.index_file(file_path)
        logger.debug(f"Re-indexed {file_path.name}: {chunks_indexed} chunks")

        # Also update code_vectors if it exists
        await self._update_code_vectors_for_file(file_path)

    async def _update_code_vectors_for_file(self, file_path: Path) -> None:
        """Update code_vectors index for a single file if code index exists.

        This keeps code_vectors in sync with the main index when files change.
        """
        try:
            # Check if code_vectors table exists
            index_path = self.config.index_path
            if index_path.name == "lance":
                lance_path = index_path
            else:
                lance_path = index_path / "lance"

            code_vectors_path = lance_path / "code_vectors.lance"
            if not code_vectors_path.exists() or not code_vectors_path.is_dir():
                # No code_vectors table, skip
                return

            # Code vectors exists, update it for this file
            from .chunks_backend import ChunksBackend
            from .embeddings import create_embedding_function
            from .vectors_backend import VectorsBackend

            # Get chunks for this file
            chunks_backend = ChunksBackend(lance_path)
            await chunks_backend.initialize()

            # Get relative path
            try:
                relative_path = str(file_path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(file_path)

            # Query chunks for this file
            chunks = await chunks_backend.get_chunks_by_file(relative_path)

            if not chunks:
                logger.debug(
                    f"No chunks found for {file_path.name}, skipping code vectors update"
                )
                return

            # Create CodeT5+ embedding function
            from ..config.defaults import DEFAULT_EMBEDDING_MODELS

            code_model = DEFAULT_EMBEDDING_MODELS["code_specialized"]
            embedding_func, _ = create_embedding_function(
                model_name=code_model, cache_dir=None
            )

            # Generate embeddings for chunks
            chunk_contents = [chunk["content"] for chunk in chunks]
            embeddings = embedding_func(chunk_contents)

            # Add vectors to code_vectors backend
            code_vectors_backend = VectorsBackend(
                lance_path, vector_dim=256, table_name="code_vectors"
            )
            await code_vectors_backend.initialize()

            # Delete existing vectors for this file first
            await code_vectors_backend.delete_file_vectors(relative_path)

            # Prepare chunks with embeddings
            chunks_with_vectors = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                chunk_with_vector = {
                    "chunk_id": chunk["chunk_id"],
                    "vector": embedding,
                    "file_path": chunk["file_path"],
                    "content": chunk["content"],
                    "language": chunk["language"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "chunk_type": chunk["chunk_type"],
                    "name": chunk["name"],
                    "hierarchy_path": chunk.get("hierarchy_path", ""),
                }
                chunks_with_vectors.append(chunk_with_vector)

            # Add new vectors
            count = await code_vectors_backend.add_vectors(
                chunks_with_vectors, model_version=code_model
            )

            logger.debug(f"Updated {count} code vectors for {file_path.name}")
            await code_vectors_backend.close()
            await chunks_backend.close()

        except Exception as e:
            # Non-fatal: code vectors update failure shouldn't break file watching
            logger.debug(f"Failed to update code vectors for {file_path.name}: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class WatcherManager:
    """Manager for file watchers across multiple projects."""

    def __init__(self):
        """Initialize watcher manager."""
        self.watchers: dict[str, FileWatcher] = {}

    async def start_watcher(
        self,
        project_root: Path,
        config: ProjectConfig,
        indexer: SemanticIndexer,
        database: VectorDatabase,
    ) -> FileWatcher:
        """Start a file watcher for a project."""
        project_key = str(project_root)

        if project_key in self.watchers:
            logger.warning(f"Watcher already exists for {project_root}")
            return self.watchers[project_key]

        watcher = FileWatcher(project_root, config, indexer, database)
        await watcher.start()

        self.watchers[project_key] = watcher
        return watcher

    async def stop_watcher(self, project_root: Path) -> None:
        """Stop a file watcher for a project."""
        project_key = str(project_root)

        if project_key not in self.watchers:
            logger.warning(f"No watcher found for {project_root}")
            return

        watcher = self.watchers.pop(project_key)
        await watcher.stop()

    async def stop_all(self) -> None:
        """Stop all file watchers."""
        for watcher in list(self.watchers.values()):
            await watcher.stop()
        self.watchers.clear()

    def is_watching(self, project_root: Path) -> bool:
        """Check if a project is being watched."""
        return str(project_root) in self.watchers

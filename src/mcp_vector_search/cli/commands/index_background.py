"""Background indexing entry point for detached process execution."""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.indexer import SemanticIndexer
from ...core.project import ProjectManager


class BackgroundIndexer:
    """Background indexer with progress tracking."""

    def __init__(self, project_root: Path, progress_file: Path, log_file: Path):
        """Initialize background indexer.

        Args:
            project_root: Project root directory
            progress_file: Path to progress JSON file
            log_file: Path to log file
        """
        self.project_root = project_root
        self.progress_file = progress_file
        self.log_file = log_file
        self.progress_data = {
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "status": "initializing",
            "total_files": 0,
            "processed_files": 0,
            "current_file": None,
            "chunks_created": 0,
            "errors": 0,
            "last_updated": datetime.now(UTC).isoformat(),
            "eta_seconds": 0,
        }
        self.start_time = time.time()

    def _write_progress(self) -> None:
        """Write progress data to file atomically."""
        # Atomic write: temp file + rename
        self.progress_data["last_updated"] = datetime.now(UTC).isoformat()

        temp_file = self.progress_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(self.progress_data, f, indent=2)
            # Atomic rename
            temp_file.replace(self.progress_file)
        except Exception as e:
            logger.error(f"Failed to write progress file: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _update_progress(
        self,
        status: str | None = None,
        current_file: str | None = None,
        processed_increment: int = 0,
        chunks_increment: int = 0,
        error_increment: int = 0,
    ) -> None:
        """Update progress data and write to file.

        Args:
            status: New status value
            current_file: Current file being processed
            processed_increment: Number of files processed to add
            chunks_increment: Number of chunks created to add
            error_increment: Number of errors to add
        """
        if status:
            self.progress_data["status"] = status

        if current_file is not None:
            self.progress_data["current_file"] = current_file

        self.progress_data["processed_files"] += processed_increment
        self.progress_data["chunks_created"] += chunks_increment
        self.progress_data["errors"] += error_increment

        # Calculate ETA
        elapsed = time.time() - self.start_time
        processed = self.progress_data["processed_files"]
        total = self.progress_data["total_files"]

        if processed > 0 and total > processed:
            rate = elapsed / processed
            remaining = total - processed
            self.progress_data["eta_seconds"] = int(rate * remaining)
        else:
            self.progress_data["eta_seconds"] = 0

        self._write_progress()

    async def run(
        self, force_reindex: bool = False, extensions: str | None = None
    ) -> None:
        """Run background indexing process.

        Args:
            force_reindex: Force reindexing of all files
            extensions: Override file extensions (comma-separated)
        """
        try:
            # Load project configuration
            logger.info(f"Loading project configuration from {self.project_root}")
            project_manager = ProjectManager(self.project_root)

            if not project_manager.is_initialized():
                raise RuntimeError(
                    f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first."
                )

            config = project_manager.load_config()

            # Override extensions if provided
            if extensions:
                file_extensions = [ext.strip() for ext in extensions.split(",")]
                file_extensions = [
                    ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
                ]
                config = config.model_copy(update={"file_extensions": file_extensions})

            logger.info(f"File extensions: {', '.join(config.file_extensions)}")
            logger.info(f"Embedding model: {config.embedding_model}")

            # Setup embedding function and cache
            from ...config.defaults import get_default_cache_path

            cache_dir = (
                get_default_cache_path(self.project_root)
                if config.cache_embeddings
                else None
            )
            embedding_function, cache = create_embedding_function(
                model_name=config.embedding_model,
                cache_dir=cache_dir,
                cache_size=config.max_cache_size,
            )

            # Setup database
            database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )

            # Setup indexer
            indexer = SemanticIndexer(
                database=database,
                project_root=self.project_root,
                config=config,
            )

            async with database:
                # Get files to index
                self._update_progress(status="scanning")
                logger.info("Scanning for indexable files...")
                indexable_files, files_to_index = await indexer.get_files_to_index(
                    force_reindex=force_reindex
                )

                total_files = len(files_to_index)
                self.progress_data["total_files"] = total_files
                self._write_progress()

                if total_files == 0:
                    logger.info("No files need indexing")
                    self._update_progress(status="completed")
                    return

                logger.info(f"Found {total_files} files to index")
                self._update_progress(status="running")

                # Index files with progress tracking
                async for (
                    file_path,
                    chunks_added,
                    success,
                ) in indexer.index_files_with_progress(files_to_index, force_reindex):
                    # Update progress
                    try:
                        relative_path = str(file_path.relative_to(self.project_root))
                    except ValueError:
                        relative_path = str(file_path)

                    self._update_progress(
                        current_file=relative_path,
                        processed_increment=1,
                        chunks_increment=chunks_added if success else 0,
                        error_increment=0 if success else 1,
                    )

                # Rebuild directory index
                try:
                    logger.info("Rebuilding directory index...")
                    chunk_stats = {}
                    for file_path in files_to_index:
                        try:
                            mtime = os.path.getmtime(file_path)
                            chunk_stats[str(file_path)] = {
                                "modified": mtime,
                                "chunks": 1,
                            }
                        except OSError:
                            pass

                    indexer.directory_index.rebuild_from_files(
                        files_to_index, self.project_root, chunk_stats=chunk_stats
                    )
                    indexer.directory_index.save()
                except Exception as e:
                    logger.error(f"Failed to update directory index: {e}")

                # Compute relationships
                try:
                    logger.info("Marking relationships for background computation...")
                    self._update_progress(status="computing_relationships")
                    all_chunks = await indexer.database.get_all_chunks()

                    if len(all_chunks) > 0:
                        # Use background=True to avoid blocking startup
                        await indexer.relationship_store.compute_and_store(
                            all_chunks, indexer.database, background=True
                        )
                        logger.info("✓ Relationships marked for background computation")
                        logger.info(
                            "  → Relationships will be lazy-loaded during visualization"
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute relationships: {e}")

                # Mark as completed
                self._update_progress(status="completed", current_file=None)
                logger.info(
                    f"Indexing completed: {self.progress_data['processed_files']} files, "
                    f"{self.progress_data['chunks_created']} chunks, "
                    f"{self.progress_data['errors']} errors"
                )

        except Exception as e:
            logger.error(f"Background indexing failed: {e}", exc_info=True)
            self._update_progress(status="failed")
            raise

    async def run_relationships_only(self) -> None:
        """Run relationship computation only (skip file indexing).

        This is used when user wants to compute relationships in background
        after indexing has already completed.
        """
        try:
            # Load project configuration
            logger.info(f"Loading project configuration from {self.project_root}")
            project_manager = ProjectManager(self.project_root)

            if not project_manager.is_initialized():
                raise RuntimeError(
                    f"Project not initialized at {self.project_root}. Run 'mcp-vector-search init' first."
                )

            config = project_manager.load_config()

            logger.info(f"Embedding model: {config.embedding_model}")

            # Setup embedding function and cache
            from ...config.defaults import get_default_cache_path

            cache_dir = (
                get_default_cache_path(self.project_root)
                if config.cache_embeddings
                else None
            )
            embedding_function, cache = create_embedding_function(
                model_name=config.embedding_model,
                cache_dir=cache_dir,
                cache_size=config.max_cache_size,
            )

            # Setup database
            database = ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
            )

            # Setup indexer (for relationship store access)
            indexer = SemanticIndexer(
                database=database,
                project_root=self.project_root,
                config=config,
            )

            async with database:
                # Get chunks for relationship computation
                self._update_progress(status="loading_chunks")
                logger.info("Loading chunks from database...")
                all_chunks = await indexer.database.get_all_chunks()

                if len(all_chunks) == 0:
                    logger.warning("No chunks found in database")
                    self._update_progress(status="completed")
                    return

                logger.info(f"Found {len(all_chunks)} chunks")

                # Compute relationships
                logger.info("Computing semantic relationships...")
                self._update_progress(status="computing_relationships")

                rel_stats = await indexer.relationship_store.compute_and_store(
                    all_chunks, indexer.database, background=False
                )

                logger.info(
                    f"Computed {rel_stats['semantic_links']} semantic links "
                    f"in {rel_stats['computation_time']:.1f}s"
                )

                # Mark as completed
                self._update_progress(status="completed", current_file=None)
                logger.info("Relationship computation completed")

        except Exception as e:
            logger.error(
                f"Background relationship computation failed: {e}", exc_info=True
            )
            self._update_progress(status="failed")
            raise


def setup_logging(log_file: Path) -> None:
    """Setup logging to file.

    Args:
        log_file: Path to log file
    """
    # Remove default handler
    logger.remove()

    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
    )

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def cleanup_progress_file(progress_file: Path) -> None:
    """Clean up progress file on exit.

    Args:
        progress_file: Path to progress file
    """
    try:
        if progress_file.exists():
            progress_file.unlink()
            logger.info("Cleaned up progress file")
    except Exception as e:
        logger.error(f"Failed to clean up progress file: {e}")


def main() -> None:
    """Main entry point for background indexing."""
    parser = argparse.ArgumentParser(
        description="Background indexing process for MCP Vector Search"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project root directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing of all files",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        help="Override file extensions (comma-separated)",
    )
    parser.add_argument(
        "--relationships-only",
        action="store_true",
        help="Only compute relationships (skip file indexing)",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = args.project_root.resolve()
    config_dir = project_root / ".mcp-vector-search"

    # Use different files for relationships-only mode
    if args.relationships_only:
        progress_file = config_dir / "relationships_progress.json"
        log_file = config_dir / "relationships_background.log"
    else:
        progress_file = config_dir / "indexing_progress.json"
        log_file = config_dir / "indexing_background.log"

    # Setup logging
    setup_logging(log_file)

    if args.relationships_only:
        logger.info(
            f"Starting background relationship computation (PID: {os.getpid()})"
        )
    else:
        logger.info(f"Starting background indexing (PID: {os.getpid()})")

    logger.info(f"Project root: {project_root}")
    logger.info(f"Force reindex: {args.force}")

    # Create background indexer
    bg_indexer = BackgroundIndexer(project_root, progress_file, log_file)

    # Handle SIGTERM for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        bg_indexer._update_progress(status="cancelled")
        cleanup_progress_file(progress_file)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run indexing or relationships-only
    try:
        if args.relationships_only:
            asyncio.run(bg_indexer.run_relationships_only())
        else:
            asyncio.run(
                bg_indexer.run(force_reindex=args.force, extensions=args.extensions)
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        bg_indexer._update_progress(status="cancelled")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        bg_indexer._update_progress(status="failed")
        sys.exit(1)
    finally:
        # Keep progress file on completion for status command to read
        pass


if __name__ == "__main__":
    main()

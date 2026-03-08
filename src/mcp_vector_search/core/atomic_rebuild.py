"""Atomic database rebuild manager for SemanticIndexer.

Handles the two-phase atomic swap pattern:
1. Build index into .new directories
2. Atomically rename .new -> final, old -> .old, then delete .old

Extracted from indexer.py to reduce complexity.
"""

import shutil
from pathlib import Path

from loguru import logger

from .chunks_backend import ChunksBackend
from .vectors_backend import VectorsBackend


class AtomicRebuildManager:
    """Manages atomic database rebuild: prepare new dirs, swap on success.

    Usage pattern in SemanticIndexer::

        manager = AtomicRebuildManager(mcp_dir, config)
        active = await manager.rebuild()
        if active:
            self.chunks_backend = manager.chunks_backend
            self.vectors_backend = manager.vectors_backend
            self.database       = manager.database
            # ... index into the new backends ...
            await manager.finalize()
            # Caller must re-init backends pointing at final paths after finalize.

    The ``active`` attribute tracks whether a rebuild is in progress; callers
    should copy this flag back onto ``SemanticIndexer._atomic_rebuild_active``
    when needed.
    """

    def __init__(self, mcp_dir: Path, config) -> None:
        self._mcp_dir = mcp_dir
        self._config = config
        self.active: bool = False

        # Populated by rebuild() on success; callers re-assign these onto self.
        self.chunks_backend: ChunksBackend | None = None
        self.vectors_backend: VectorsBackend | None = None
        self.database = None

    async def rebuild(self, force: bool = False) -> bool:
        """Prepare .new database directories for atomic rebuild.

        Creates fresh backend instances pointing at ``<mcp_dir>/lance.new``.
        On success sets ``self.chunks_backend``, ``self.vectors_backend``,
        ``self.database``, and ``self.active = True``.

        Args:
            force: If False, does nothing and returns False immediately.

        Returns:
            True if atomic rebuild was prepared, False otherwise.
        """
        if not force:
            return False

        base_path = self._mcp_dir

        lance_new = base_path / "lance.new"
        kg_new = base_path / "knowledge_graph.new"
        chroma_new = base_path / "chroma.sqlite3.new"
        code_search_new = base_path / "code_search.lance.new"

        try:
            logger.info("Atomic rebuild: preparing new database directories...")

            # Remove any stale .new directories from previous interrupted rebuilds
            for new_path in [lance_new, kg_new, code_search_new]:
                if new_path.exists():
                    try:
                        shutil.rmtree(new_path)
                        logger.debug(f"Removed stale directory: {new_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove stale directory {new_path}: {e}. "
                            "Attempting to continue anyway."
                        )
            if chroma_new.exists():
                try:
                    chroma_new.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove stale file {chroma_new}: {e}")

            # Create new database directories
            lance_new.mkdir(parents=True, exist_ok=True)
            kg_new.mkdir(parents=True, exist_ok=True)

            # Create new backend instances pointing at .new locations
            self.chunks_backend = ChunksBackend(lance_new)
            self.vectors_backend = VectorsBackend(lance_new)

            # Create new database instance pointing at .new location
            from .embeddings import create_embedding_function
            from .factory import create_database

            model_name = self._config.embedding_model if self._config else None
            embedding_function, _ = create_embedding_function(model_name=model_name)
            new_db_path = base_path / "code_search.lance.new"
            self.database = create_database(
                persist_directory=new_db_path, embedding_function=embedding_function
            )

            logger.info("New database directories prepared")
            self.active = True
            return True

        except Exception as e:
            logger.error(f"Failed to prepare atomic rebuild: {e}")
            # Cleanup .new directories on failure
            for new_path in [lance_new, kg_new, code_search_new]:
                if new_path.exists():
                    shutil.rmtree(new_path, ignore_errors=True)
            if chroma_new.exists():
                chroma_new.unlink()
            self.active = False
            return False

    async def finalize(self) -> None:
        """Atomically switch .new databases to final paths.

        Performs rename operations: existing -> .old, .new -> final, deletes .old.
        Rolls back if any step fails.

        Raises:
            Exception: Re-raises after attempting rollback if finalization fails.
        """
        base_path = self._mcp_dir

        lance_path = base_path / "lance"
        lance_new = base_path / "lance.new"
        lance_old = base_path / "lance.old"

        kg_path = base_path / "knowledge_graph"
        kg_new = base_path / "knowledge_graph.new"
        kg_old = base_path / "knowledge_graph.old"

        chroma_path = base_path / "chroma.sqlite3"
        chroma_new = base_path / "chroma.sqlite3.new"
        chroma_old = base_path / "chroma.sqlite3.old"

        code_search_path = base_path / "code_search.lance"
        code_search_new = base_path / "code_search.lance.new"
        code_search_old = base_path / "code_search.lance.old"

        try:
            logger.info("Atomic rebuild: finalizing database switch...")

            # Atomic switch for lance directory
            if lance_new.exists():
                if lance_path.exists():
                    lance_path.rename(lance_old)
                lance_new.rename(lance_path)
                if lance_old.exists():
                    try:
                        shutil.rmtree(lance_old)
                        logger.debug(f"Removed old directory: {lance_old}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old directory {lance_old}: {e}"
                        )

            # Atomic switch for knowledge_graph directory
            if kg_new.exists():
                if kg_path.exists():
                    kg_path.rename(kg_old)
                kg_new.rename(kg_path)
                if kg_old.exists():
                    try:
                        shutil.rmtree(kg_old)
                        logger.debug(f"Removed old directory: {kg_old}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old directory {kg_old}: {e}")

            # Atomic switch for code_search.lance
            if code_search_new.exists():
                if code_search_path.exists():
                    code_search_path.rename(code_search_old)
                code_search_new.rename(code_search_path)
                if code_search_old.exists():
                    try:
                        shutil.rmtree(code_search_old)
                        logger.debug(f"Removed old directory: {code_search_old}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove old directory {code_search_old}: {e}"
                        )

            # Handle chroma.sqlite3 (deprecated, but may exist)
            if chroma_new.exists():
                if chroma_path.exists():
                    chroma_path.rename(chroma_old)
                chroma_new.rename(chroma_path)
                if chroma_old.exists():
                    chroma_old.unlink()

            logger.info("Atomic rebuild complete: databases switched successfully")
            self.active = False

        except Exception as e:
            logger.error(f"Failed to finalize atomic rebuild: {e}")
            # Attempt rollback
            try:
                logger.warning("Attempting rollback to old databases...")
                if lance_old.exists() and not lance_path.exists():
                    lance_old.rename(lance_path)
                if kg_old.exists() and not kg_path.exists():
                    kg_old.rename(kg_path)
                if code_search_old.exists() and not code_search_path.exists():
                    code_search_old.rename(code_search_path)
                if chroma_old.exists() and not chroma_path.exists():
                    chroma_old.rename(chroma_path)
                logger.info("Rollback successful, old databases restored")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            self.active = False
            raise

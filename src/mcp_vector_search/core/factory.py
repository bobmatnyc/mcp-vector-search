"""Component factory for creating commonly used objects."""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import typer
from loguru import logger

from ..cli.output import print_error
from ..config.settings import ProjectConfig
from .auto_indexer import AutoIndexer
from .database import VectorDatabase
from .embeddings import CodeBERTEmbeddingFunction, create_embedding_function
from .indexer import SemanticIndexer
from .lancedb_backend import LanceVectorDatabase
from .project import ProjectManager
from .search import SemanticSearchEngine

F = TypeVar("F", bound=Callable[..., Any])


def resolve_index_path(
    config: ProjectConfig,
    index_path: str | Path | None = None,
) -> Path:
    """Resolve the effective index root directory.

    Priority order:
    1. Explicit ``index_path`` argument
    2. ``INDEX_PATH`` environment variable
    3. ``config.index_path`` (project default, usually ``<project_root>/.mcp-vector-search``)

    Args:
        config: Project configuration (provides the fallback path).
        index_path: Optional caller-supplied override.

    Returns:
        Resolved absolute path to the ``.mcp-vector-search`` parent directory.
        The caller is responsible for appending ``/ ".mcp-vector-search"`` or
        sub-paths like ``/ ".mcp-vector-search" / "lance"`` as required.
    """
    if index_path:
        return Path(index_path).resolve()
    env_path = os.environ.get("INDEX_PATH")
    if env_path:
        return Path(env_path).resolve()
    # config.index_path already points to the .mcp-vector-search directory
    return config.index_path


@dataclass
class ComponentBundle:
    """Bundle of commonly used components."""

    project_manager: ProjectManager
    config: ProjectConfig
    database: VectorDatabase
    indexer: SemanticIndexer
    embedding_function: CodeBERTEmbeddingFunction
    search_engine: SemanticSearchEngine | None = None
    auto_indexer: AutoIndexer | None = None


class ComponentFactory:
    """Factory for creating commonly used components."""

    @staticmethod
    def create_project_manager(project_root: Path) -> ProjectManager:
        """Create a project manager."""
        return ProjectManager(project_root)

    @staticmethod
    def load_config(project_root: Path) -> tuple[ProjectManager, ProjectConfig]:
        """Load project configuration."""
        project_manager = ComponentFactory.create_project_manager(project_root)
        config = project_manager.load_config()
        return project_manager, config

    @staticmethod
    def create_embedding_function(
        model_name: str,
    ) -> tuple[CodeBERTEmbeddingFunction, Any]:
        """Create embedding function."""
        return create_embedding_function(model_name)

    @staticmethod
    def create_database(
        config: ProjectConfig,
        embedding_function: CodeBERTEmbeddingFunction,
        use_pooling: bool = True,  # Kept for backward compatibility (ignored)
        backend: str | None = None,  # Kept for backward compatibility (ignored)
        index_path: str | Path | None = None,
        **pool_kwargs,
    ) -> VectorDatabase:
        """Create vector database (LanceDB).

        Args:
            config: Project configuration
            embedding_function: Embedding function
            use_pooling: DEPRECATED - kept for backward compatibility
            backend: DEPRECATED - kept for backward compatibility
            index_path: Optional separate directory for index data.
                Overrides config.index_path and INDEX_PATH env var.
                When provided, the database is placed at
                ``<index_path>/.mcp-vector-search/lance``.
            **pool_kwargs: DEPRECATED - kept for backward compatibility

        Returns:
            LanceDB vector database instance
        """
        # Resolve the effective index root (honours INDEX_PATH env var and explicit override)
        effective_root = resolve_index_path(config, index_path)
        # config.index_path already IS the .mcp-vector-search directory, so when
        # resolve_index_path returns config.index_path we append "lance" directly.
        # When it returns an external path (INDEX_PATH / explicit arg) we must also
        # descend into .mcp-vector-search/lance to match SemanticIndexer._mcp_dir layout.
        if effective_root == config.index_path:
            lance_path = effective_root / "lance"
        else:
            lance_path = effective_root / ".mcp-vector-search" / "lance"
            logger.debug(f"Using separate lance index path: {lance_path}")

        logger.debug("Using LanceDB backend")
        return LanceVectorDatabase(
            persist_directory=lance_path,
            embedding_function=embedding_function,
            collection_name="chunks",  # Standard table name
        )

    @staticmethod
    def create_indexer(
        database: VectorDatabase,
        project_root: Path,
        config: ProjectConfig,
        index_path: str | None = None,
    ) -> SemanticIndexer:
        """Create semantic indexer.

        Args:
            database: Vector database instance
            project_root: Project root directory (where source code lives)
            config: Project configuration
            index_path: Optional separate directory for .mcp-vector-search/ index data.
                Falls back to INDEX_PATH env var, then project_root.
        """
        return SemanticIndexer(
            database=database,
            project_root=project_root,
            config=config,
            index_path=index_path,
        )

    @staticmethod
    def create_search_engine(
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.3,
        auto_indexer: AutoIndexer | None = None,
        enable_auto_reindex: bool = True,
    ) -> SemanticSearchEngine:
        """Create semantic search engine."""
        return SemanticSearchEngine(
            database=database,
            project_root=project_root,
            similarity_threshold=similarity_threshold,
            auto_indexer=auto_indexer,
            enable_auto_reindex=enable_auto_reindex,
        )

    @staticmethod
    def create_auto_indexer(
        indexer: SemanticIndexer,
        database: VectorDatabase,
        auto_reindex_threshold: int = 5,
        staleness_threshold: float = 300.0,
    ) -> AutoIndexer:
        """Create auto-indexer."""
        return AutoIndexer(
            indexer=indexer,
            database=database,
            auto_reindex_threshold=auto_reindex_threshold,
            staleness_threshold=staleness_threshold,
        )

    @staticmethod
    async def create_standard_components(
        project_root: Path,
        use_pooling: bool = True,  # DEPRECATED - kept for backward compatibility
        include_search_engine: bool = False,
        include_auto_indexer: bool = False,
        similarity_threshold: float = 0.3,
        auto_reindex_threshold: int = 5,
        index_path: str | Path | None = None,
        **pool_kwargs,
    ) -> ComponentBundle:
        """Create standard set of components for CLI commands.

        Args:
            project_root: Project root directory
            use_pooling: DEPRECATED - kept for backward compatibility
            include_search_engine: Whether to create search engine
            include_auto_indexer: Whether to create auto-indexer
            similarity_threshold: Default similarity threshold for search
            auto_reindex_threshold: Max files to auto-reindex
            index_path: Optional separate directory for .mcp-vector-search/ index data.
                Falls back to INDEX_PATH env var, then config.index_path.
            **pool_kwargs: DEPRECATED - kept for backward compatibility

        Returns:
            ComponentBundle with requested components
        """
        # Load configuration
        project_manager, config = ComponentFactory.load_config(project_root)

        # Create embedding function
        embedding_function, _ = ComponentFactory.create_embedding_function(
            config.embedding_model
        )

        # Create database (respects index_path / INDEX_PATH env var)
        database = ComponentFactory.create_database(
            config=config,
            embedding_function=embedding_function,
            use_pooling=use_pooling,
            index_path=index_path,
            **pool_kwargs,
        )

        # Create indexer (respects index_path / INDEX_PATH env var)
        indexer = ComponentFactory.create_indexer(
            database=database,
            project_root=project_root,
            config=config,
            index_path=str(index_path) if index_path else None,
        )

        # Create optional components
        search_engine = None
        auto_indexer = None

        if include_auto_indexer:
            auto_indexer = ComponentFactory.create_auto_indexer(
                indexer=indexer,
                database=database,
                auto_reindex_threshold=auto_reindex_threshold,
            )

        if include_search_engine:
            search_engine = ComponentFactory.create_search_engine(
                database=database,
                project_root=project_root,
                similarity_threshold=similarity_threshold,
                auto_indexer=auto_indexer,
                enable_auto_reindex=include_auto_indexer,
            )

        return ComponentBundle(
            project_manager=project_manager,
            config=config,
            database=database,
            indexer=indexer,
            embedding_function=embedding_function,
            search_engine=search_engine,
            auto_indexer=auto_indexer,
        )


class DatabaseContext:
    """Context manager for database lifecycle management."""

    def __init__(self, database: VectorDatabase):
        """Initialize database context.

        Args:
            database: Vector database instance
        """
        self.database = database

    async def __aenter__(self) -> VectorDatabase:
        """Enter context and initialize database."""
        await self.database.initialize()
        return self.database

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and close database."""
        await self.database.close()


def handle_cli_errors(operation_name: str) -> Callable[[F], F]:
    """Decorator for consistent CLI error handling.

    Args:
        operation_name: Name of the operation for error messages

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                print_error(f"{operation_name} failed: {e}")
                raise typer.Exit(1)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                print_error(f"{operation_name} failed: {e}")
                raise typer.Exit(1)

        # Return appropriate wrapper based on function type
        if hasattr(func, "__code__") and "await" in func.__code__.co_names:
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def create_database(
    persist_directory: str | Path,
    collection_name: str = "chunks",
    embedding_function: CodeBERTEmbeddingFunction | None = None,
    use_pooling: bool = True,  # DEPRECATED - kept for backward compatibility
    backend: str | None = None,  # DEPRECATED - kept for backward compatibility
    **pool_kwargs,
) -> VectorDatabase:
    """Create vector database instance (LanceDB).

    This is a standalone convenience function for creating a LanceDB instance.

    Args:
        persist_directory: Directory to persist database
        collection_name: Name of the collection (default: "chunks")
        embedding_function: Embedding function to use (required)
        use_pooling: DEPRECATED - kept for backward compatibility
        backend: DEPRECATED - kept for backward compatibility
        **pool_kwargs: DEPRECATED - kept for backward compatibility

    Returns:
        LanceDB vector database instance

    Example:
        >>> from mcp_vector_search.core.factory import create_database
        >>> from mcp_vector_search.core.embeddings import create_embedding_function
        >>>
        >>> embedding_fn, _ = create_embedding_function("microsoft/codebert-base")
        >>> async with create_database("./index", embedding_function=embedding_fn) as db:
        ...     await db.add_chunk(...)
    """
    # Convert persist_directory to string
    persist_dir_str = str(persist_directory)

    # Always use LanceDB backend
    logger.debug("Using LanceDB backend")
    if embedding_function is None:
        raise ValueError("embedding_function is required")
    return LanceVectorDatabase(
        persist_directory=persist_dir_str,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )


class ConfigurationService:
    """Centralized configuration management service."""

    def __init__(self, project_root: Path):
        """Initialize configuration service.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self._project_manager: ProjectManager | None = None
        self._config: ProjectConfig | None = None

    @property
    def project_manager(self) -> ProjectManager:
        """Get project manager (lazy loaded)."""
        if self._project_manager is None:
            self._project_manager = ProjectManager(self.project_root)
        return self._project_manager

    @property
    def config(self) -> ProjectConfig:
        """Get project configuration (lazy loaded)."""
        if self._config is None:
            self._config = self.project_manager.load_config()
        return self._config

    def ensure_initialized(self) -> bool:
        """Ensure project is initialized.

        Returns:
            True if project is initialized, False otherwise
        """
        if not self.project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            return False
        return True

    def reload_config(self) -> None:
        """Reload configuration from disk."""
        self._config = None

    def save_config(self, config: ProjectConfig) -> None:
        """Save configuration to disk.

        Args:
            config: Configuration to save
        """
        self.project_manager.save_config(config)
        self._config = config

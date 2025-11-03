"""Component factory for creating commonly used objects."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import typer
from loguru import logger

from ..cli.output import print_error
from ..config.settings import ProjectConfig
from .auto_indexer import AutoIndexer
from .database import ChromaVectorDatabase, PooledChromaVectorDatabase, VectorDatabase
from .embeddings import CodeBERTEmbeddingFunction, create_embedding_function
from .indexer import SemanticIndexer
from .project import ProjectManager
from .search import SemanticSearchEngine

F = TypeVar("F", bound=Callable[..., Any])


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
        use_pooling: bool = True,  # Enable pooling by default for 13.6% performance boost
        **pool_kwargs,
    ) -> VectorDatabase:
        """Create vector database."""
        if use_pooling:
            # Set default pool parameters if not provided
            pool_defaults = {
                "max_connections": 10,
                "min_connections": 2,
                "max_idle_time": 300.0,
            }
            pool_defaults.update(pool_kwargs)

            return PooledChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
                **pool_defaults,
            )
        else:
            return ChromaVectorDatabase(
                persist_directory=config.index_path,
                embedding_function=embedding_function,
                collection_name="code_search",
            )

    @staticmethod
    def create_indexer(
        database: VectorDatabase, project_root: Path, config: ProjectConfig
    ) -> SemanticIndexer:
        """Create semantic indexer."""
        return SemanticIndexer(
            database=database,
            project_root=project_root,
            config=config,
        )

    @staticmethod
    def create_search_engine(
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.7,
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
        use_pooling: bool = True,  # Enable pooling by default for performance
        include_search_engine: bool = False,
        include_auto_indexer: bool = False,
        similarity_threshold: float = 0.7,
        auto_reindex_threshold: int = 5,
        **pool_kwargs,
    ) -> ComponentBundle:
        """Create standard set of components for CLI commands.

        Args:
            project_root: Project root directory
            use_pooling: Whether to use connection pooling
            include_search_engine: Whether to create search engine
            include_auto_indexer: Whether to create auto-indexer
            similarity_threshold: Default similarity threshold for search
            auto_reindex_threshold: Max files to auto-reindex
            **pool_kwargs: Additional arguments for connection pool

        Returns:
            ComponentBundle with requested components
        """
        # Load configuration
        project_manager, config = ComponentFactory.load_config(project_root)

        # Create embedding function
        embedding_function, _ = ComponentFactory.create_embedding_function(
            config.embedding_model
        )

        # Create database
        database = ComponentFactory.create_database(
            config=config,
            embedding_function=embedding_function,
            use_pooling=use_pooling,
            **pool_kwargs,
        )

        # Create indexer
        indexer = ComponentFactory.create_indexer(
            database=database,
            project_root=project_root,
            config=config,
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

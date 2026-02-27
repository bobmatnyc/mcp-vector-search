"""Tests for the typed exception hierarchy (Issue #110).

Validates:
- Class hierarchy is correct (isinstance checks)
- New exceptions are exported from the package root
- SemanticIndexer.index_project() raises IndexingError on failure
- SemanticSearchEngine.search() raises SearchError on failure
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Hierarchy tests (no I/O, no async)
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the class hierarchy defined in core/exceptions.py."""

    def test_mcp_vector_search_error_is_base_exception(self):
        from mcp_vector_search.core.exceptions import MCPVectorSearchError

        err = MCPVectorSearchError("base")
        assert isinstance(err, Exception)

    def test_mvs_error_inherits_from_mcp_vector_search_error(self):
        from mcp_vector_search.core.exceptions import MCPVectorSearchError, MVSError

        err = MVSError("mvs")
        assert isinstance(err, MCPVectorSearchError)

    def test_search_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import MVSError, SearchError

        err = SearchError("search")
        assert isinstance(err, MVSError)

    def test_indexing_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import IndexingError, MVSError

        err = IndexingError("index")
        assert isinstance(err, MVSError)

    def test_config_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import ConfigError, MVSError

        err = ConfigError("config")
        assert isinstance(err, MVSError)

    def test_initialization_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import InitializationError, MVSError

        err = InitializationError("init")
        assert isinstance(err, MVSError)

    def test_database_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import DatabaseError, MVSError

        err = DatabaseError("db")
        assert isinstance(err, MVSError)

    def test_database_initialization_error_inherits_from_database_error(self):
        from mcp_vector_search.core.exceptions import (
            DatabaseError,
            DatabaseInitializationError,
        )

        err = DatabaseInitializationError("db init")
        assert isinstance(err, DatabaseError)

    def test_parsing_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import MVSError, ParsingError

        err = ParsingError("parse")
        assert isinstance(err, MVSError)

    def test_embedding_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import EmbeddingError, MVSError

        err = EmbeddingError("embed")
        assert isinstance(err, MVSError)

    def test_configuration_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import ConfigurationError, MVSError

        err = ConfigurationError("conf")
        assert isinstance(err, MVSError)

    def test_project_error_inherits_from_mvs_error(self):
        from mcp_vector_search.core.exceptions import MVSError, ProjectError

        err = ProjectError("proj")
        assert isinstance(err, MVSError)

    def test_project_not_found_inherits_from_project_error(self):
        from mcp_vector_search.core.exceptions import ProjectError, ProjectNotFoundError

        err = ProjectNotFoundError("not found")
        assert isinstance(err, ProjectError)

    def test_project_initialization_error_inherits_from_project_error(self):
        from mcp_vector_search.core.exceptions import (
            ProjectError,
            ProjectInitializationError,
        )

        err = ProjectInitializationError("proj init")
        assert isinstance(err, ProjectError)

    def test_indexing_error_alias_equals_indexing_error(self):
        """IndexError alias in exceptions module must point to IndexingError."""
        import mcp_vector_search.core.exceptions as exc_mod

        # The module-level alias IndexError should be the same class
        assert exc_mod.IndexError is exc_mod.IndexingError

    def test_context_dict_is_preserved(self):
        from mcp_vector_search.core.exceptions import MVSError

        err = MVSError("msg", context={"key": "value"})
        assert err.context == {"key": "value"}

    def test_context_defaults_to_empty_dict(self):
        from mcp_vector_search.core.exceptions import MVSError

        err = MVSError("msg")
        assert err.context == {}

    def test_catch_all_with_mvs_error(self):
        """Demonstrate that all public-API exceptions can be caught as MVSError."""
        from mcp_vector_search.core.exceptions import (
            ConfigError,
            DatabaseError,
            IndexingError,
            InitializationError,
            MVSError,
            SearchError,
        )

        for exc_class in (
            SearchError,
            IndexingError,
            ConfigError,
            InitializationError,
            DatabaseError,
        ):
            with pytest.raises(MVSError):
                raise exc_class("test")


# ---------------------------------------------------------------------------
# Package-level export tests
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify new exceptions are importable from the mcp_vector_search root."""

    def test_mvs_error_exported(self):
        from mcp_vector_search import MVSError  # noqa: F401

    def test_search_error_exported(self):
        from mcp_vector_search import SearchError  # noqa: F401

    def test_indexing_error_exported(self):
        from mcp_vector_search import IndexingError  # noqa: F401

    def test_index_error_alias_exported(self):
        import mcp_vector_search as pkg
        from mcp_vector_search import IndexingError

        # IndexError alias must resolve to the same class
        assert pkg.IndexError is IndexingError

    def test_config_error_exported(self):
        from mcp_vector_search import ConfigError  # noqa: F401

    def test_initialization_error_exported(self):
        from mcp_vector_search import InitializationError  # noqa: F401

    def test_database_error_exported(self):
        from mcp_vector_search import DatabaseError  # noqa: F401

    def test_all_includes_new_exceptions(self):
        import mcp_vector_search as pkg

        for name in (
            "MVSError",
            "SearchError",
            "IndexingError",
            "ConfigError",
            "InitializationError",
        ):
            assert name in pkg.__all__, f"{name!r} missing from __all__"

    def test_legacy_exceptions_still_exported(self):
        """Ensure backward-compatible exports are preserved."""
        import mcp_vector_search as pkg

        for name in (
            "MCPVectorSearchError",
            "DatabaseError",
            "DatabaseInitializationError",
            "DatabaseNotInitializedError",
            "EmbeddingError",
            "ParsingError",
            "ProjectError",
            "ProjectInitializationError",
            "ProjectNotFoundError",
        ):
            assert name in pkg.__all__, f"Legacy export {name!r} missing from __all__"


# ---------------------------------------------------------------------------
# Indexer wrapping tests
# ---------------------------------------------------------------------------


class TestIndexerExceptionWrapping:
    """Verify SemanticIndexer.index_project raises IndexingError on failure."""

    @pytest.mark.asyncio
    async def test_index_project_wraps_unexpected_exception(self, tmp_path):
        """Unexpected exceptions during index_project are re-raised as IndexingError."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from mcp_vector_search.core.exceptions import IndexingError
        from mcp_vector_search.core.indexer import SemanticIndexer

        mock_db = MagicMock()
        mock_db.get_stats = AsyncMock(
            return_value=MagicMock(total_chunks=0, total_files=0)
        )

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            file_extensions=[".py"],
        )

        # Make the internal implementation raise an unexpected error
        with patch.object(
            indexer,
            "_index_project_impl",
            side_effect=RuntimeError("unexpected internal error"),
        ):
            with pytest.raises(IndexingError) as exc_info:
                await indexer.index_project()

        assert "Indexing failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_index_project_does_not_double_wrap_indexing_error(self, tmp_path):
        """IndexingError is re-raised as-is without double-wrapping."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from mcp_vector_search.core.exceptions import IndexingError
        from mcp_vector_search.core.indexer import SemanticIndexer

        mock_db = MagicMock()
        mock_db.get_stats = AsyncMock(
            return_value=MagicMock(total_chunks=0, total_files=0)
        )

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            file_extensions=[".py"],
        )

        original_error = IndexingError("already typed", context={"phase": "chunk"})

        with patch.object(
            indexer,
            "_index_project_impl",
            side_effect=original_error,
        ):
            with pytest.raises(IndexingError) as exc_info:
                await indexer.index_project()

        # Should be the exact same exception (not wrapped again)
        assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_index_project_exception_includes_context(self, tmp_path):
        """IndexingError context dict contains project_root and phase."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from mcp_vector_search.core.exceptions import IndexingError
        from mcp_vector_search.core.indexer import SemanticIndexer

        mock_db = MagicMock()
        mock_db.get_stats = AsyncMock(
            return_value=MagicMock(total_chunks=0, total_files=0)
        )

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            file_extensions=[".py"],
        )

        with patch.object(
            indexer,
            "_index_project_impl",
            side_effect=ValueError("boom"),
        ):
            with pytest.raises(IndexingError) as exc_info:
                await indexer.index_project(phase="chunk")

        ctx = exc_info.value.context
        assert "project_root" in ctx
        assert "phase" in ctx
        assert ctx["phase"] == "chunk"


# ---------------------------------------------------------------------------
# Search engine wrapping tests
# ---------------------------------------------------------------------------


class TestSearchExceptionWrapping:
    """Verify SemanticSearchEngine.search raises SearchError on failure."""

    @pytest.mark.asyncio
    async def test_search_raises_search_error_on_unexpected_failure(self, tmp_path):
        """Unexpected exceptions in search() are wrapped in SearchError."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_vector_search.core.exceptions import SearchError
        from mcp_vector_search.core.search import SemanticSearchEngine

        mock_db = MagicMock()
        mock_db.get_stats = AsyncMock(
            return_value=MagicMock(total_chunks=10, total_files=2)
        )

        engine = SemanticSearchEngine(database=mock_db, project_root=tmp_path)

        # Patch the internal health-check to avoid side effects, then make
        # query processor raise an unexpected error.
        engine._perform_health_check = AsyncMock()
        engine._perform_auto_reindex_check = AsyncMock()
        engine._query_processor.preprocess_query = MagicMock(
            side_effect=RuntimeError("unexpected processing failure")
        )

        with pytest.raises(SearchError) as exc_info:
            await engine.search("test query")

        assert "Search failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_search_error_is_mvs_error(self, tmp_path):
        """SearchError raised by search() is also catchable as MVSError."""
        from unittest.mock import AsyncMock, MagicMock

        from mcp_vector_search.core.exceptions import MVSError
        from mcp_vector_search.core.search import SemanticSearchEngine

        mock_db = MagicMock()
        mock_db.get_stats = AsyncMock(
            return_value=MagicMock(total_chunks=10, total_files=2)
        )

        engine = SemanticSearchEngine(database=mock_db, project_root=tmp_path)
        engine._perform_health_check = AsyncMock()
        engine._perform_auto_reindex_check = AsyncMock()
        engine._query_processor.preprocess_query = MagicMock(
            side_effect=ValueError("bad query")
        )

        with pytest.raises(MVSError):
            await engine.search("test query")

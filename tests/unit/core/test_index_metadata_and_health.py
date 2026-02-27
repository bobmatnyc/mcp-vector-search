"""Unit tests for index metadata storage (#114) and health_check() (#119)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search import HealthStatus as PublicHealthStatus
from mcp_vector_search.core.index_metadata import IndexMetadata
from mcp_vector_search.core.indexer import HealthStatus, SemanticIndexer

# ---------------------------------------------------------------------------
# IndexMetadata tests (#114)
# ---------------------------------------------------------------------------


class TestIndexMetadataEmbeddingModel:
    """Tests for embedding model storage in IndexMetadata."""

    def test_save_with_embedding_model(self, tmp_path: Path) -> None:
        """save() persists embedding_model and embedding_dimensions."""
        meta = IndexMetadata(tmp_path)
        meta.save(
            {},
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimensions=384,
        )

        raw = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )
        assert raw["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert raw["embedding_dimensions"] == 384

    def test_save_preserves_existing_model_when_not_supplied(
        self, tmp_path: Path
    ) -> None:
        """Subsequent save() without embedding args keeps prior values."""
        meta = IndexMetadata(tmp_path)
        meta.save(
            {},
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimensions=384,
        )
        # Save again without model args — should keep original values
        meta.save({"some/file.py": 1234567890.0})

        raw = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )
        assert raw["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert raw["embedding_dimensions"] == 384

    def test_save_allows_overriding_model(self, tmp_path: Path) -> None:
        """save() with a new model name replaces the old value."""
        meta = IndexMetadata(tmp_path)
        meta.save({}, embedding_model="model-v1", embedding_dimensions=384)
        meta.save({}, embedding_model="model-v2", embedding_dimensions=768)

        result = meta.get_index_metadata()
        assert result["embedding_model"] == "model-v2"
        assert result["embedding_dimensions"] == 768

    def test_save_records_created_at_once(self, tmp_path: Path) -> None:
        """created_at is set on first write and never overwritten."""
        meta = IndexMetadata(tmp_path)
        meta.save({})
        first_created = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )["created_at"]

        # Second write should keep the same created_at
        meta.save({})
        second_created = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )["created_at"]

        assert first_created == second_created

    def test_save_updates_updated_at(self, tmp_path: Path) -> None:
        """updated_at is written on every save()."""
        import time

        meta = IndexMetadata(tmp_path)
        meta.save({})
        raw1 = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )

        time.sleep(0.01)  # ensure timestamp difference
        meta.save({})
        raw2 = json.loads(
            (tmp_path / ".mcp-vector-search" / "index_metadata.json").read_text()
        )

        assert raw2["updated_at"] >= raw1["updated_at"]


class TestGetIndexMetadata:
    """Tests for IndexMetadata.get_index_metadata()."""

    def test_returns_none_values_when_no_file(self, tmp_path: Path) -> None:
        """get_index_metadata() returns dict with None values if no file exists."""
        meta = IndexMetadata(tmp_path)
        result = meta.get_index_metadata()

        assert result["embedding_model"] is None
        assert result["embedding_dimensions"] is None
        assert result["index_version"] is None
        assert result["created_at"] is None
        assert result["updated_at"] is None

    def test_returns_saved_values(self, tmp_path: Path) -> None:
        """get_index_metadata() returns what was last saved."""
        meta = IndexMetadata(tmp_path)
        meta.save(
            {},
            embedding_model="nomic-ai/CodeRankEmbed",
            embedding_dimensions=768,
        )

        result = meta.get_index_metadata()
        assert result["embedding_model"] == "nomic-ai/CodeRankEmbed"
        assert result["embedding_dimensions"] == 768
        assert result["index_version"] is not None  # current version
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

    def test_returns_dict_with_required_keys(self, tmp_path: Path) -> None:
        """get_index_metadata() always returns all expected keys."""
        meta = IndexMetadata(tmp_path)
        result = meta.get_index_metadata()

        expected_keys = {
            "embedding_model",
            "embedding_dimensions",
            "index_version",
            "created_at",
            "updated_at",
        }
        assert expected_keys == set(result.keys())

    def test_handles_corrupted_file(self, tmp_path: Path) -> None:
        """get_index_metadata() returns None values when JSON is corrupt."""
        mcp_dir = tmp_path / ".mcp-vector-search"
        mcp_dir.mkdir(parents=True, exist_ok=True)
        (mcp_dir / "index_metadata.json").write_text("not valid json")

        meta = IndexMetadata(tmp_path)
        result = meta.get_index_metadata()
        assert result["embedding_model"] is None


# ---------------------------------------------------------------------------
# HealthStatus dataclass tests
# ---------------------------------------------------------------------------


class TestHealthStatus:
    """Tests for the HealthStatus dataclass."""

    def test_default_values(self) -> None:
        """HealthStatus defaults to safe unknown state."""
        h = HealthStatus()
        assert h.db_connected is False
        assert h.tables_exist == []
        assert h.model_loaded is False
        assert h.index_valid is False
        assert h.status == "unknown"
        assert h.details == {}

    def test_is_exported_from_package(self) -> None:
        """HealthStatus is accessible from top-level mcp_vector_search package."""
        assert PublicHealthStatus is HealthStatus


# ---------------------------------------------------------------------------
# SemanticIndexer.health_check() tests (#119)
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for SemanticIndexer.health_check()."""

    def _make_indexer(self, tmp_path: Path) -> SemanticIndexer:
        """Create a minimal SemanticIndexer with a mock database."""
        mock_db = MagicMock()
        mock_db.embedding_function = MagicMock()
        mock_db.embedding_function.model_name = "all-MiniLM-L6-v2"

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            file_extensions=[".py"],
        )
        return indexer

    @pytest.mark.asyncio
    async def test_healthy_when_tables_exist(self, tmp_path: Path) -> None:
        """health_check() returns 'healthy' when DB and both tables exist."""
        indexer = self._make_indexer(tmp_path)

        # Create the lance directory structure
        lance_path = tmp_path / ".mcp-vector-search" / "lance"
        lance_path.mkdir(parents=True, exist_ok=True)

        # Patch lancedb at the module level (imported locally inside health_check)
        fake_db = MagicMock()
        fake_db.list_tables.return_value = ["chunks", "vectors"]

        with patch("lancedb.connect", return_value=fake_db):
            health = await indexer.health_check()

        assert health.db_connected is True
        assert "chunks" in health.tables_exist
        assert "vectors" in health.tables_exist
        assert health.model_loaded is True
        assert health.index_valid is True
        assert health.status == "healthy"

    @pytest.mark.asyncio
    async def test_degraded_when_only_chunks_table(self, tmp_path: Path) -> None:
        """health_check() returns 'degraded' when only the chunks table exists."""
        indexer = self._make_indexer(tmp_path)

        lance_path = tmp_path / ".mcp-vector-search" / "lance"
        lance_path.mkdir(parents=True, exist_ok=True)

        fake_db = MagicMock()
        fake_db.list_tables.return_value = ["chunks"]

        with patch("lancedb.connect", return_value=fake_db):
            health = await indexer.health_check()

        assert health.db_connected is True
        assert health.index_valid is False
        assert health.status == "degraded"

    @pytest.mark.asyncio
    async def test_unhealthy_when_lance_dir_missing(self, tmp_path: Path) -> None:
        """health_check() returns 'unhealthy' when lance directory doesn't exist."""
        indexer = self._make_indexer(tmp_path)
        # Don't create lance dir — it won't exist

        health = await indexer.health_check()

        assert health.db_connected is False
        assert health.index_valid is False
        assert health.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_details_contains_db_path(self, tmp_path: Path) -> None:
        """health_check() details dict contains db_path key."""
        indexer = self._make_indexer(tmp_path)
        health = await indexer.health_check()

        assert "db_path" in health.details

    @pytest.mark.asyncio
    async def test_details_reports_table_status(self, tmp_path: Path) -> None:
        """health_check() details reports individual table presence."""
        indexer = self._make_indexer(tmp_path)

        lance_path = tmp_path / ".mcp-vector-search" / "lance"
        lance_path.mkdir(parents=True, exist_ok=True)

        fake_db = MagicMock()
        fake_db.list_tables.return_value = ["chunks"]

        with patch("lancedb.connect", return_value=fake_db):
            health = await indexer.health_check()

        assert health.details.get("chunks_table") == "present"
        assert health.details.get("vectors_table") == "missing"

    @pytest.mark.asyncio
    async def test_does_not_raise_on_db_error(self, tmp_path: Path) -> None:
        """health_check() returns unhealthy status instead of raising."""
        indexer = self._make_indexer(tmp_path)

        lance_path = tmp_path / ".mcp-vector-search" / "lance"
        lance_path.mkdir(parents=True, exist_ok=True)

        with patch("lancedb.connect", side_effect=RuntimeError("DB unavailable")):
            health = await indexer.health_check()

        # Should not raise; returns unhealthy
        assert health.status in ("unhealthy", "degraded")


# ---------------------------------------------------------------------------
# SemanticIndexer.get_index_metadata() delegation tests
# ---------------------------------------------------------------------------


class TestSemanticIndexerGetIndexMetadata:
    """Tests for SemanticIndexer.get_index_metadata() delegation."""

    def _make_indexer(self, tmp_path: Path) -> SemanticIndexer:
        mock_db = MagicMock()
        mock_db.embedding_function = MagicMock()
        mock_db.embedding_function.model_name = "all-MiniLM-L6-v2"
        return SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            file_extensions=[".py"],
        )

    def test_returns_metadata_dict(self, tmp_path: Path) -> None:
        """get_index_metadata() returns a dict with required keys."""
        indexer = self._make_indexer(tmp_path)
        result = indexer.get_index_metadata()

        assert isinstance(result, dict)
        assert "embedding_model" in result
        assert "index_version" in result

    def test_reflects_saved_model(self, tmp_path: Path) -> None:
        """get_index_metadata() reflects what was written to disk."""
        indexer = self._make_indexer(tmp_path)
        # Write metadata manually
        indexer.metadata.save(
            {},
            embedding_model="test-model",
            embedding_dimensions=512,
        )

        result = indexer.get_index_metadata()
        assert result["embedding_model"] == "test-model"
        assert result["embedding_dimensions"] == 512

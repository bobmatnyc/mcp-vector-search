"""Tests for migration system."""

import json
from datetime import datetime
from pathlib import Path

import lancedb
import pyarrow as pa

from mcp_vector_search.migrations import (
    Migration,
    MigrationContext,
    MigrationRegistry,
    MigrationResult,
    MigrationRunner,
    MigrationStatus,
)
from mcp_vector_search.migrations.v2_3_0_two_phase import TwoPhaseArchitectureMigration


class TestMigration(Migration):
    """Test migration for unit tests."""

    version = "1.0.0"
    name = "test_migration"
    description = "Test migration"

    def __init__(self, should_run: bool = True, should_fail: bool = False):
        self.should_run = should_run
        self.should_fail = should_fail
        self.executed = False

    def check_needed(self, context: MigrationContext) -> bool:
        return self.should_run

    def execute(self, context: MigrationContext) -> MigrationResult:
        self.executed = True

        if self.should_fail:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message="Test migration failed",
            )

        return MigrationResult(
            migration_id=self.migration_id,
            version=self.version,
            name=self.name,
            status=MigrationStatus.SUCCESS,
            message="Test migration completed",
        )


class TestMigrationRegistry:
    """Tests for MigrationRegistry."""

    def test_create_registry(self, tmp_path: Path):
        """Should create registry file on initialization."""
        registry = MigrationRegistry(tmp_path)
        assert registry.registry_file.exists()

        data = json.loads(registry.registry_file.read_text())
        assert "migrations" in data
        assert data["migrations"] == []

    def test_record_and_retrieve_migration(self, tmp_path: Path):
        """Should record and retrieve migration results."""
        registry = MigrationRegistry(tmp_path)

        result = MigrationResult(
            migration_id="1.0.0_test",
            version="1.0.0",
            name="test",
            status=MigrationStatus.SUCCESS,
            message="Success",
            executed_at=datetime.now(),
        )

        registry.record_migration(result)

        # Retrieve all migrations
        migrations = registry.get_executed_migrations()
        assert len(migrations) == 1
        assert migrations[0].migration_id == "1.0.0_test"
        assert migrations[0].status == MigrationStatus.SUCCESS

    def test_has_migration_run(self, tmp_path: Path):
        """Should check if migration has run successfully."""
        registry = MigrationRegistry(tmp_path)

        # Record successful migration
        result = MigrationResult(
            migration_id="1.0.0_test",
            version="1.0.0",
            name="test",
            status=MigrationStatus.SUCCESS,
            message="Success",
            executed_at=datetime.now(),
        )
        registry.record_migration(result)

        assert registry.has_migration_run("1.0.0_test") is True
        assert registry.has_migration_run("2.0.0_other") is False

    def test_get_last_version(self, tmp_path: Path):
        """Should get the last successfully executed version."""
        registry = MigrationRegistry(tmp_path)

        # Record multiple migrations
        for version in ["1.0.0", "1.1.0", "1.2.0"]:
            result = MigrationResult(
                migration_id=f"{version}_test",
                version=version,
                name="test",
                status=MigrationStatus.SUCCESS,
                message="Success",
                executed_at=datetime.now(),
            )
            registry.record_migration(result)

        assert registry.get_last_version() == "1.2.0"

    def test_record_replaces_previous_attempt(self, tmp_path: Path):
        """Should replace previous migration attempt."""
        registry = MigrationRegistry(tmp_path)

        # Record failed migration
        result1 = MigrationResult(
            migration_id="1.0.0_test",
            version="1.0.0",
            name="test",
            status=MigrationStatus.FAILED,
            message="Failed",
            executed_at=datetime.now(),
        )
        registry.record_migration(result1)

        # Record successful migration
        result2 = MigrationResult(
            migration_id="1.0.0_test",
            version="1.0.0",
            name="test",
            status=MigrationStatus.SUCCESS,
            message="Success",
            executed_at=datetime.now(),
        )
        registry.record_migration(result2)

        # Should only have one entry (successful)
        migrations = registry.get_executed_migrations()
        assert len(migrations) == 1
        assert migrations[0].status == MigrationStatus.SUCCESS


class TestMigrationRunner:
    """Tests for MigrationRunner."""

    def test_get_pending_migrations(self, tmp_path: Path):
        """Should identify pending migrations."""
        runner = MigrationRunner(tmp_path)

        migration1 = TestMigration(should_run=True)
        migration2 = TestMigration(should_run=False)
        migration2.version = "1.1.0"
        migration2.name = "test2"

        runner.register_migrations([migration1, migration2])

        pending = runner.get_pending_migrations()
        assert len(pending) == 1
        assert pending[0].migration_id == migration1.migration_id

    def test_run_migration_success(self, tmp_path: Path):
        """Should successfully run a migration."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=True)

        result = runner.run_migration(migration)

        assert result.status == MigrationStatus.SUCCESS
        assert migration.executed is True
        assert runner.registry.has_migration_run(migration.migration_id)

    def test_run_migration_failure(self, tmp_path: Path):
        """Should handle migration failure."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=True, should_fail=True)

        result = runner.run_migration(migration)

        assert result.status == MigrationStatus.FAILED
        assert migration.executed is True

    def test_run_migration_not_needed(self, tmp_path: Path):
        """Should skip migration if not needed."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=False)

        result = runner.run_migration(migration)

        assert result.status == MigrationStatus.SKIPPED
        assert migration.executed is False

    def test_run_migration_already_executed(self, tmp_path: Path):
        """Should skip migration if already executed successfully."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=True)

        # Run first time
        result1 = runner.run_migration(migration)
        assert result1.status == MigrationStatus.SUCCESS

        # Reset execution flag
        migration.executed = False

        # Run second time (should skip)
        result2 = runner.run_migration(migration)
        assert result2.status == MigrationStatus.SKIPPED
        assert migration.executed is False

    def test_run_pending_migrations(self, tmp_path: Path):
        """Should run all pending migrations in order."""
        runner = MigrationRunner(tmp_path)

        migration1 = TestMigration(should_run=True)
        migration2 = TestMigration(should_run=True)
        migration2.version = "1.1.0"
        migration2.name = "test2"

        runner.register_migrations([migration1, migration2])

        results = runner.run_pending_migrations()

        assert len(results) == 2
        assert all(r.status == MigrationStatus.SUCCESS for r in results)

    def test_dry_run(self, tmp_path: Path):
        """Should not execute migrations in dry run mode."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=True)

        result = runner.run_migration(migration, dry_run=True)

        assert result.status == MigrationStatus.PENDING
        assert migration.executed is False
        assert not runner.registry.has_migration_run(migration.migration_id)

    def test_force_rerun(self, tmp_path: Path):
        """Should rerun migration when forced."""
        runner = MigrationRunner(tmp_path)
        migration = TestMigration(should_run=True)

        # Run first time
        result1 = runner.run_migration(migration)
        assert result1.status == MigrationStatus.SUCCESS

        # Reset execution flag
        migration.executed = False

        # Force rerun
        result2 = runner.run_migration(migration, force=True)
        assert result2.status == MigrationStatus.SUCCESS
        assert migration.executed is True

    def test_list_migrations(self, tmp_path: Path):
        """Should list all migrations with status."""
        runner = MigrationRunner(tmp_path)

        migration1 = TestMigration(should_run=True)
        migration2 = TestMigration(should_run=True)
        migration2.version = "1.1.0"
        migration2.name = "test2"

        runner.register_migrations([migration1, migration2])

        # Run first migration
        runner.run_migration(migration1)

        # List all migrations
        migrations = runner.list_migrations()

        assert len(migrations) == 2
        assert migrations[0]["status"] == "success"  # migration1
        assert migrations[1]["status"] == "not_run"  # migration2


class TestMigrationContext:
    """Tests for MigrationContext."""

    def test_get_config_value(self, tmp_path: Path):
        """Should retrieve config values with defaults."""
        context = MigrationContext(
            project_root=tmp_path,
            index_path=tmp_path / ".mcp-vector-search",
            config={"key1": "value1"},
        )

        assert context.get_config_value("key1") == "value1"
        assert context.get_config_value("key2", "default") == "default"


class TestMigrationResult:
    """Tests for MigrationResult."""

    def test_to_dict_and_from_dict(self):
        """Should serialize and deserialize correctly."""
        result = MigrationResult(
            migration_id="1.0.0_test",
            version="1.0.0",
            name="test",
            status=MigrationStatus.SUCCESS,
            message="Success",
            executed_at=datetime.now(),
            duration_seconds=1.5,
            metadata={"key": "value"},
        )

        # Serialize
        data = result.to_dict()
        assert data["migration_id"] == "1.0.0_test"
        assert data["status"] == "success"

        # Deserialize
        restored = MigrationResult.from_dict(data)
        assert restored.migration_id == result.migration_id
        assert restored.status == result.status
        assert restored.metadata == result.metadata


class TestTwoPhaseArchitectureMigration:
    """Tests for two-phase architecture migration."""

    def test_check_needed_no_old_table(self, tmp_path: Path):
        """Should not need migration if no old table exists."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        # Create fresh LanceDB (no tables)
        lancedb.connect(str(db_path))

        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
        )

        assert not migration.check_needed(context)

    def test_check_needed_new_tables_exist(self, tmp_path: Path):
        """Should not need migration if new tables already exist."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        # Create LanceDB with new tables
        db = lancedb.connect(str(db_path))

        # Create chunks table
        chunks_data = [
            {
                "chunk_id": "test1",
                "file_path": "test.py",
                "file_hash": "abc123",
                "content": "def test(): pass",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "start_char": 0,
                "end_char": 20,
                "chunk_type": "function",
                "name": "test",
                "parent_name": "",
                "hierarchy_path": "test",
                "docstring": "",
                "signature": "",
                "complexity": 1,
                "token_count": 5,
                "embedding_status": "pending",
                "embedding_batch_id": 0,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "error_message": "",
            }
        ]
        from mcp_vector_search.core.chunks_backend import CHUNKS_SCHEMA

        db.create_table("chunks", chunks_data, schema=CHUNKS_SCHEMA)

        # Create vectors table
        vectors_data = [
            {
                "chunk_id": "test1",
                "vector": [0.1] * 384,
                "file_path": "test.py",
                "content": "def test(): pass",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "function",
                "name": "test",
                "hierarchy_path": "test",
                "embedded_at": "2024-01-01T00:00:00",
                "model_version": "all-MiniLM-L6-v2",
            }
        ]
        from mcp_vector_search.core.vectors_backend import VECTORS_SCHEMA

        db.create_table("vectors", vectors_data, schema=VECTORS_SCHEMA)

        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
        )

        assert not migration.check_needed(context)

    def test_check_needed_old_table_exists(self, tmp_path: Path):
        """Should need migration if old table exists without new tables."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        # Create LanceDB with old code_chunks table
        db = lancedb.connect(str(db_path))

        # Create old schema table
        old_data = [
            {
                "chunk_id": "test1",
                "content": "def test(): pass",
                "vector": [0.1] * 384,
                "file_path": "test.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "function",
                "function_name": "test",
                "class_name": "",
                "docstring": "",
                "complexity_score": 1,
            }
        ]

        # Create schema for old table
        old_schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 384)),
                pa.field("file_path", pa.string()),
                pa.field("language", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("chunk_type", pa.string()),
                pa.field("function_name", pa.string()),
                pa.field("class_name", pa.string()),
                pa.field("docstring", pa.string()),
                pa.field("complexity_score", pa.int32()),
            ]
        )

        db.create_table("code_chunks", old_data, schema=old_schema)

        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
        )

        assert migration.check_needed(context)

    def test_execute_migration(self, tmp_path: Path):
        """Should successfully migrate from old to new schema."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        # Create LanceDB with old code_search table
        db = lancedb.connect(str(db_path))

        # Create old schema table with test data
        old_data = [
            {
                "chunk_id": "test1",
                "content": "def foo(): pass",
                "vector": [0.1] * 384,
                "file_path": "test.py",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "start_char": 0,
                "end_char": 20,
                "chunk_type": "function",
                "function_name": "foo",
                "class_name": "",
                "docstring": "Test function",
                "complexity_score": 1,
                "hierarchy_path": "foo",
            },
            {
                "chunk_id": "test2",
                "content": "class Bar: pass",
                "vector": [0.2] * 384,
                "file_path": "test.py",
                "language": "python",
                "start_line": 3,
                "end_line": 3,
                "start_char": 0,
                "end_char": 15,
                "chunk_type": "class",
                "function_name": "",
                "class_name": "Bar",
                "docstring": "",
                "complexity_score": 1,
                "hierarchy_path": "Bar",
            },
        ]

        # Create schema for old table
        old_schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 384)),
                pa.field("file_path", pa.string()),
                pa.field("language", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("start_char", pa.int32()),
                pa.field("end_char", pa.int32()),
                pa.field("chunk_type", pa.string()),
                pa.field("function_name", pa.string()),
                pa.field("class_name", pa.string()),
                pa.field("docstring", pa.string()),
                pa.field("complexity_score", pa.int32()),
                pa.field("hierarchy_path", pa.string()),
            ]
        )

        db.create_table("code_search", old_data, schema=old_schema)

        # Run migration
        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
        )

        result = migration.execute(context)

        # Verify migration succeeded
        assert result.status == MigrationStatus.SUCCESS
        assert result.metadata["chunks_migrated"] == 2
        assert result.metadata["vectors_migrated"] == 2

        # Verify new tables exist
        tables_response = db.list_tables()
        if hasattr(tables_response, "tables"):
            table_names = tables_response.tables
        else:
            table_names = tables_response

        assert "chunks" in table_names
        assert "vectors" in table_names

        # Verify data migrated correctly
        chunks_table = db.open_table("chunks")
        chunks_df = chunks_table.to_pandas()
        assert len(chunks_df) == 2
        assert "test1" in chunks_df["chunk_id"].values
        assert "test2" in chunks_df["chunk_id"].values
        assert (
            chunks_df[chunks_df["chunk_id"] == "test1"]["embedding_status"].iloc[0]
            == "complete"
        )

        vectors_table = db.open_table("vectors")
        vectors_df = vectors_table.to_pandas()
        assert len(vectors_df) == 2
        assert "test1" in vectors_df["chunk_id"].values
        assert (
            len(vectors_df[vectors_df["chunk_id"] == "test1"]["vector"].iloc[0]) == 384
        )

    def test_execute_dry_run(self, tmp_path: Path):
        """Should return success without making changes in dry run mode."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
            dry_run=True,
        )

        result = migration.execute(context)

        assert result.status == MigrationStatus.SUCCESS
        assert "DRY RUN" in result.message

    def test_rollback(self, tmp_path: Path):
        """Should drop new tables on rollback."""
        db_path = tmp_path / "lance"
        db_path.mkdir(parents=True)

        db = lancedb.connect(str(db_path))

        # Create new tables
        chunks_data = [
            {
                "chunk_id": "test",
                "file_path": "test.py",
                "file_hash": "",
                "content": "",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "start_char": 0,
                "end_char": 0,
                "chunk_type": "code",
                "name": "",
                "parent_name": "",
                "hierarchy_path": "",
                "docstring": "",
                "signature": "",
                "complexity": 0,
                "token_count": 0,
                "embedding_status": "pending",
                "embedding_batch_id": 0,
                "created_at": "",
                "updated_at": "",
                "error_message": "",
            }
        ]
        from mcp_vector_search.core.chunks_backend import CHUNKS_SCHEMA

        db.create_table("chunks", chunks_data, schema=CHUNKS_SCHEMA)

        vectors_data = [
            {
                "chunk_id": "test",
                "vector": [0.1] * 384,
                "file_path": "test.py",
                "content": "",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "code",
                "name": "",
                "hierarchy_path": "",
                "embedded_at": "",
                "model_version": "test",
            }
        ]
        from mcp_vector_search.core.vectors_backend import VECTORS_SCHEMA

        db.create_table("vectors", vectors_data, schema=VECTORS_SCHEMA)

        # Run rollback
        migration = TwoPhaseArchitectureMigration()
        context = MigrationContext(
            project_root=tmp_path,
            index_path=db_path,
            config={},
        )

        success = migration.rollback(context)

        assert success

        # Verify tables were dropped
        tables_response = db.list_tables()
        if hasattr(tables_response, "tables"):
            table_names = tables_response.tables
        else:
            table_names = tables_response

        assert "chunks" not in table_names
        assert "vectors" not in table_names

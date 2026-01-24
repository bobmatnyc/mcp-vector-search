"""Tests for migration system."""

import json
from datetime import datetime
from pathlib import Path

from mcp_vector_search.migrations import (
    Migration,
    MigrationContext,
    MigrationRegistry,
    MigrationResult,
    MigrationRunner,
    MigrationStatus,
)


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

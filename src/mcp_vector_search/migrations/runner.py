"""Migration runner for executing pending migrations."""

import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from packaging import version

from ..core.config_utils import ConfigManager
from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus
from .registry import MigrationRegistry


class MigrationRunner:
    """Execute and manage migrations."""

    def __init__(self, project_root: Path):
        """Initialize migration runner.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.index_path = project_root / ".mcp-vector-search"
        self.registry = MigrationRegistry(self.index_path)
        self._interrupted = False

        # Register interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum: int, frame) -> None:
        """Handle SIGINT gracefully during migration."""
        logger.warning("Migration interrupted by user (Ctrl+C)")
        self._interrupted = True

    def register_migrations(self, migrations: list[Migration]) -> None:
        """Register available migrations.

        Args:
            migrations: List of migration instances
        """
        self._migrations = sorted(migrations, key=lambda m: version.parse(m.version))
        logger.debug(f"Registered {len(self._migrations)} migrations")

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations that need to run.

        Returns:
            List of pending migrations sorted by version
        """
        if not hasattr(self, "_migrations"):
            return []

        pending = []
        last_version = self.registry.get_last_version()

        for migration in self._migrations:
            # Skip if already executed successfully
            if self.registry.has_migration_run(migration.migration_id):
                continue

            # Skip if version is older than last executed version
            if last_version and version.parse(migration.version) <= version.parse(
                last_version
            ):
                continue

            # Check if migration is needed
            context = self._create_context(dry_run=True)
            if migration.check_needed(context):
                pending.append(migration)

        return pending

    def run_pending_migrations(self, dry_run: bool = False) -> list[MigrationResult]:
        """Run all pending migrations.

        Args:
            dry_run: If True, only check what would be done

        Returns:
            List of migration results
        """
        pending = self.get_pending_migrations()
        if not pending:
            logger.info("No pending migrations to run")
            return []

        logger.info(
            f"Found {len(pending)} pending migration(s) "
            f"{'(DRY RUN)' if dry_run else ''}"
        )

        results = []
        for migration in pending:
            if self._interrupted:
                logger.warning("Migration run interrupted, stopping...")
                break

            result = self.run_migration(migration, dry_run=dry_run)
            results.append(result)

            # Stop on failure unless it's a dry run
            if result.status == MigrationStatus.FAILED and not dry_run:
                logger.error(
                    f"Migration {migration.migration_id} failed, stopping execution"
                )
                break

        return results

    def run_migration(
        self, migration: Migration, dry_run: bool = False, force: bool = False
    ) -> MigrationResult:
        """Run a specific migration.

        Args:
            migration: Migration to execute
            dry_run: If True, only check what would be done
            force: If True, run even if already executed

        Returns:
            Migration result
        """
        logger.info(f"Running migration: {migration.migration_id}")

        # Check if already run (unless forced)
        if not force and self.registry.has_migration_run(migration.migration_id):
            result = MigrationResult(
                migration_id=migration.migration_id,
                version=migration.version,
                name=migration.name,
                status=MigrationStatus.SKIPPED,
                message="Migration already executed successfully",
                executed_at=datetime.now(),
            )
            logger.info(f"Skipping {migration.migration_id} (already executed)")
            return result

        # Create context
        context = self._create_context(dry_run=dry_run)

        # Check if migration is needed
        if not migration.check_needed(context):
            result = MigrationResult(
                migration_id=migration.migration_id,
                version=migration.version,
                name=migration.name,
                status=MigrationStatus.SKIPPED,
                message="Migration not needed",
                executed_at=datetime.now(),
            )
            logger.info(f"Skipping {migration.migration_id} (not needed)")
            self.registry.record_migration(result)
            return result

        # Execute migration
        start_time = time.time()
        try:
            if dry_run:
                result = MigrationResult(
                    migration_id=migration.migration_id,
                    version=migration.version,
                    name=migration.name,
                    status=MigrationStatus.PENDING,
                    message="DRY RUN: Would execute this migration",
                    executed_at=datetime.now(),
                )
            else:
                result = migration.execute(context)
                result.executed_at = datetime.now()
                result.duration_seconds = time.time() - start_time

                # Record result
                self.registry.record_migration(result)

                if result.status == MigrationStatus.SUCCESS:
                    logger.info(
                        f"✓ Migration {migration.migration_id} completed "
                        f"in {result.duration_seconds:.2f}s"
                    )
                else:
                    logger.error(
                        f"✗ Migration {migration.migration_id} failed: {result.message}"
                    )

            return result

        except Exception as e:
            duration = time.time() - start_time
            result = MigrationResult(
                migration_id=migration.migration_id,
                version=migration.version,
                name=migration.name,
                status=MigrationStatus.FAILED,
                message=f"Exception during execution: {e}",
                executed_at=datetime.now(),
                duration_seconds=duration,
            )
            self.registry.record_migration(result)
            logger.exception(f"Migration {migration.migration_id} failed")
            return result

    def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration.

        Args:
            migration_id: ID of migration to rollback

        Returns:
            True if rollback successful
        """
        logger.warning(f"Rolling back migration: {migration_id}")

        # Find the migration
        if not hasattr(self, "_migrations"):
            logger.error("No migrations registered")
            return False

        migration = next(
            (m for m in self._migrations if m.migration_id == migration_id), None
        )
        if not migration:
            logger.error(f"Migration {migration_id} not found")
            return False

        # Execute rollback
        context = self._create_context()
        try:
            success = migration.rollback(context)
            if success:
                logger.info(f"✓ Rollback of {migration_id} successful")
            else:
                logger.warning(f"Rollback of {migration_id} not supported or failed")
            return success
        except Exception as e:
            logger.exception(f"Rollback of {migration_id} failed: {e}")
            return False

    def _create_context(self, dry_run: bool = False) -> MigrationContext:
        """Create migration context with project configuration."""
        # Load current config
        try:
            config_manager = ConfigManager(self.index_path)
            config_dict = config_manager.load_config()
        except Exception as e:
            logger.warning(f"Failed to load project config: {e}")
            config_dict = {}

        return MigrationContext(
            project_root=self.project_root,
            index_path=self.index_path,
            config=config_dict,
            dry_run=dry_run,
        )

    def list_migrations(self) -> list[dict[str, Any]]:
        """List all migrations with their status.

        Returns:
            List of migration info dicts
        """
        if not hasattr(self, "_migrations"):
            return []

        executed = {r.migration_id: r for r in self.registry.get_executed_migrations()}

        result = []
        for migration in self._migrations:
            executed_result = executed.get(migration.migration_id)
            result.append(
                {
                    "migration_id": migration.migration_id,
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "status": (
                        executed_result.status.value if executed_result else "not_run"
                    ),
                    "executed_at": (
                        executed_result.executed_at.isoformat()
                        if executed_result and executed_result.executed_at
                        else None
                    ),
                }
            )

        return result

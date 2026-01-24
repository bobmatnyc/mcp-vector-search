"""Migration v1.2.2: Upgrade to CodeXEmbed-400M embedding model.

This migration handles the upgrade from the legacy all-MiniLM-L6-v2 model
(384 dimensions) to the new CodeXEmbed-400M model (768 dimensions).

The migration:
1. Checks if current index uses the old model (384 dimensions)
2. Backs up index metadata
3. Clears old embeddings
4. Updates configuration to use new model
5. Triggers re-indexing if necessary
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..config.defaults import DEFAULT_EMBEDDING_MODELS, get_model_dimensions
from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus


class CodeXEmbedMigration(Migration):
    """Migrate from legacy embedding model to CodeXEmbed-400M."""

    version = "1.2.2"
    name = "codexembed_upgrade"
    description = "Upgrade to CodeXEmbed-400M embedding model (384→768 dimensions)"

    def check_needed(self, context: MigrationContext) -> bool:
        """Check if migration is needed.

        Migration is needed if:
        1. Index exists with old model (384 dimensions)
        2. Config specifies old model explicitly
        3. No embeddings exist yet (fresh install - skip migration)

        Returns:
            True if migration should run
        """
        # Check if index directory exists
        chroma_path = context.index_path / "chroma"
        if not chroma_path.exists():
            logger.debug("No ChromaDB index found, migration not needed")
            return False

        # Check config for current model
        current_model = context.get_config_value(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Check if using legacy model
        try:
            current_dims = get_model_dimensions(current_model)
            target_dims = get_model_dimensions(DEFAULT_EMBEDDING_MODELS["code"])

            if current_dims == target_dims:
                logger.debug(
                    f"Model already at target dimensions ({target_dims}), "
                    "migration not needed"
                )
                return False

            # Migration needed if dimensions differ
            logger.info(
                f"Found legacy model with {current_dims} dimensions, "
                f"migration to {target_dims} dimensions needed"
            )
            return True

        except ValueError as e:
            logger.warning(f"Could not determine model dimensions: {e}")
            # Default to needing migration if we can't determine
            return True

    def execute(self, context: MigrationContext) -> MigrationResult:
        """Execute the migration.

        Steps:
        1. Backup existing index metadata
        2. Clear old embeddings (delete chroma directory)
        3. Update config to new model
        4. Return success with metadata about what was done

        Returns:
            Migration result with status and metadata
        """
        if context.dry_run:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="DRY RUN: Would upgrade to CodeXEmbed-400M",
            )

        try:
            metadata = {}

            # Step 1: Backup metadata
            backup_path = self._backup_metadata(context)
            if backup_path:
                metadata["backup_path"] = str(backup_path)
                logger.info(f"Backed up metadata to {backup_path}")

            # Step 2: Clear old embeddings
            cleared_count = self._clear_old_embeddings(context)
            metadata["chunks_cleared"] = cleared_count
            logger.info(f"Cleared {cleared_count} chunks with old embeddings")

            # Step 3: Update config
            self._update_config(context)
            metadata["new_model"] = DEFAULT_EMBEDDING_MODELS["code"]
            logger.info(f"Updated config to use {metadata['new_model']}")

            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message=(
                    "Successfully upgraded to CodeXEmbed-400M. "
                    "Re-indexing required on next index operation."
                ),
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Migration failed: {e}")
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message=f"Migration failed: {e}",
            )

    def _backup_metadata(self, context: MigrationContext) -> Path | None:
        """Backup index metadata before clearing.

        Args:
            context: Migration context

        Returns:
            Path to backup file if successful
        """
        try:
            # Create backups directory
            backups_dir = context.index_path / "backups"
            backups_dir.mkdir(exist_ok=True)

            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backups_dir / f"pre_codexembed_migration_{timestamp}.json"

            # Collect metadata to backup
            backup_data = {
                "migration_version": self.version,
                "backup_time": datetime.now().isoformat(),
                "config": context.config,
            }

            # Write backup
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2)

            return backup_file

        except Exception as e:
            logger.warning(f"Failed to backup metadata: {e}")
            return None

    def _clear_old_embeddings(self, context: MigrationContext) -> int:
        """Clear old embedding data.

        Args:
            context: Migration context

        Returns:
            Number of chunks cleared (approximate)
        """
        chroma_path = context.index_path / "chroma"
        count = 0

        if chroma_path.exists():
            # Try to count existing chunks before deletion
            try:
                # Look for parquet files which store embeddings
                parquet_files = list(chroma_path.rglob("*.parquet"))
                count = len(parquet_files)
            except Exception:
                count = 0

            # Remove ChromaDB directory
            try:
                shutil.rmtree(chroma_path)
                logger.info("Removed old ChromaDB index directory")
            except Exception as e:
                logger.warning(f"Failed to remove old index: {e}")

        return count

    def _update_config(self, context: MigrationContext) -> None:
        """Update configuration to use new model.

        Args:
            context: Migration context
        """
        config_file = context.index_path / "config.json"

        # Load existing config or create new
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        # Update embedding model
        config["embedding_model"] = DEFAULT_EMBEDDING_MODELS["code"]
        config["last_migration"] = self.version
        config["migration_date"] = datetime.now().isoformat()

        # Write updated config
        context.index_path.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def rollback(self, context: MigrationContext) -> bool:
        """Rollback to legacy model.

        This migration doesn't support automatic rollback since
        the old embeddings have been deleted. Users must manually
        reconfigure to the old model if needed.

        Returns:
            False (rollback not supported)
        """
        logger.warning(
            "Rollback not supported for CodeXEmbed migration. "
            "Old embeddings have been deleted. "
            "To revert, manually set embedding_model to "
            "'sentence-transformers/all-MiniLM-L6-v2' in config and re-index."
        )
        return False

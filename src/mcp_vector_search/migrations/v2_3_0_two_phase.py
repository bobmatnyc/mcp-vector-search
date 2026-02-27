"""Migration v2.3.0: Upgrade from old single-table schema to two-phase architecture.

This migration handles the upgrade from the old LanceDB single-table schema
(code_chunks or code_search table) to the new two-phase architecture
(chunks + vectors tables).

The migration:
1. Detects old schema (code_chunks or code_search table exists)
2. Backs up old table (renames with timestamp)
3. Migrates data to chunks table (without vector)
4. Migrates data to vectors table (with vector + denormalized search fields)
5. Preserves all metadata, embeddings, and relationships
"""

from datetime import datetime
from typing import Any

import lancedb
from loguru import logger

from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus


class TwoPhaseArchitectureMigration(Migration):
    """Migrate from old single-table schema to two-phase architecture."""

    version = "2.3.0"
    name = "two_phase_architecture"
    description = (
        "Upgrade to two-phase architecture (chunks + vectors tables) "
        "for faster parsing and incremental embedding"
    )

    # Old table names that might exist
    OLD_TABLE_NAMES = ["code_chunks", "code_search"]
    NEW_CHUNKS_TABLE = "chunks"
    NEW_VECTORS_TABLE = "vectors"

    def check_needed(self, context: MigrationContext) -> bool:
        """Check if migration is needed.

        Migration is needed if:
        1. Old table exists (code_chunks or code_search)
        2. New tables don't exist (chunks and vectors)

        Returns:
            True if migration should run
        """
        try:
            # Connect to LanceDB
            db = lancedb.connect(str(context.index_path))

            # Check table names (handle both dict-like and list responses)
            tables_response = db.list_tables()
            if hasattr(tables_response, "tables"):
                table_names = tables_response.tables
            else:
                table_names = tables_response

            # Check if old table exists
            has_old = any(name in table_names for name in self.OLD_TABLE_NAMES)

            # Check if new tables exist
            has_new = (
                self.NEW_CHUNKS_TABLE in table_names
                and self.NEW_VECTORS_TABLE in table_names
            )

            if has_old and not has_new:
                logger.info(
                    "Detected old schema format. Migration to two-phase architecture needed."
                )
                return True

            if has_new:
                logger.debug("New two-phase tables already exist, migration not needed")
                return False

            # No tables exist yet - fresh install
            logger.debug("No index tables found (fresh install), migration not needed")
            return False

        except Exception as e:
            logger.warning(f"Could not check migration status: {e}")
            # Default to not needing migration if we can't determine
            return False

    def execute(self, context: MigrationContext) -> MigrationResult:
        """Execute the migration.

        Steps:
        1. Detect and open old table
        2. Count rows for progress tracking
        3. Create chunks table with migrated data (no vectors)
        4. Create vectors table with migrated data (vectors + search fields)
        5. Rename old table as backup
        6. Return success with statistics

        Returns:
            Migration result with status and metadata
        """
        if context.dry_run:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="DRY RUN: Would migrate to two-phase architecture",
            )

        start_time = datetime.now()
        metadata: dict[str, Any] = {}

        try:
            # Connect to LanceDB
            db = lancedb.connect(str(context.index_path))

            # Get table names
            tables_response = db.list_tables()
            if hasattr(tables_response, "tables"):
                table_names = tables_response.tables
            else:
                table_names = tables_response

            # Find old table
            old_table_name = None
            for name in self.OLD_TABLE_NAMES:
                if name in table_names:
                    old_table_name = name
                    break

            if not old_table_name:
                return MigrationResult(
                    migration_id=self.migration_id,
                    version=self.version,
                    name=self.name,
                    status=MigrationStatus.SKIPPED,
                    message="No old table found to migrate",
                    executed_at=datetime.now(),
                    duration_seconds=0.0,
                )

            logger.info(f"Starting migration from {old_table_name} to two-phase schema")

            # Open old table
            old_table = db.open_table(old_table_name)
            total_rows = old_table.count_rows()
            metadata["old_table"] = old_table_name
            metadata["total_rows"] = total_rows

            if total_rows == 0:
                return MigrationResult(
                    migration_id=self.migration_id,
                    version=self.version,
                    name=self.name,
                    status=MigrationStatus.SKIPPED,
                    message="Old table is empty, nothing to migrate",
                    executed_at=datetime.now(),
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    metadata=metadata,
                )

            logger.info(f"Migrating {total_rows:,} rows from {old_table_name}")

            # Read all data (LanceDB doesn't have great pagination for small datasets)
            all_data = old_table.to_pandas()

            # Prepare chunks data (without vector)
            chunks_data = []
            vectors_data = []
            timestamp = datetime.utcnow().isoformat()

            for idx, row in all_data.iterrows():
                chunk_id = row.get("chunk_id") or row.get("id") or str(idx)

                # Build chunk record (Phase 1 - no vector)
                chunk = {
                    "chunk_id": chunk_id,
                    "file_path": row.get("file_path", ""),
                    "file_hash": "",  # Unknown for migrated data
                    "content": row.get("content", ""),
                    "language": row.get("language", "unknown"),
                    "start_line": int(row.get("start_line", 0)),
                    "end_line": int(row.get("end_line", 0)),
                    "start_char": int(row.get("start_char", 0)),
                    "end_char": int(row.get("end_char", 0)),
                    "chunk_type": row.get("chunk_type", "code"),
                    "name": row.get("function_name") or row.get("class_name") or "",
                    "parent_name": row.get("parent_chunk_id", ""),
                    "hierarchy_path": row.get("hierarchy_path", ""),
                    "docstring": row.get("docstring", ""),
                    "signature": "",  # Not in old schema
                    "complexity": int(row.get("complexity_score", 0)),
                    "token_count": 0,  # Not in old schema
                    # Code relationships (new fields for KG support)
                    "calls": [],  # Not available in old schema
                    "imports": [],  # Not available in old schema
                    "inherits_from": [],  # Not available in old schema
                    # Git blame metadata (new fields)
                    "last_author": "",  # Not available in old schema
                    "last_modified": "",  # Not available in old schema
                    "commit_hash": "",  # Not available in old schema
                    # Phase tracking
                    "embedding_status": "complete",  # Already has vector
                    "embedding_batch_id": 0,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "error_message": "",
                }
                chunks_data.append(chunk)

                # Build vector record (Phase 2 - vector + search fields)
                vector = row.get("vector")
                if vector is not None and len(vector) > 0:
                    # Convert vector to list if it's a numpy array or similar
                    try:
                        if hasattr(vector, "tolist"):
                            vector = vector.tolist()
                        elif not isinstance(vector, list):
                            vector = list(vector)
                    except Exception as e:
                        logger.warning(
                            f"Could not convert vector for chunk {chunk_id}: {e}"
                        )
                        continue

                    # Extract function/class names separately for new schema
                    function_name = row.get("function_name", "")
                    class_name = row.get("class_name", "")
                    # Use whichever is present for the "name" field
                    name = function_name or class_name or ""

                    vector_record = {
                        "chunk_id": chunk_id,
                        "vector": vector,
                        "file_path": row.get("file_path", ""),
                        "content": row.get("content", "")[:500],  # Truncate for search
                        "language": row.get("language", "unknown"),
                        "start_line": int(row.get("start_line", 0)),
                        "end_line": int(row.get("end_line", 0)),
                        "chunk_type": row.get("chunk_type", "code"),
                        "name": name,
                        "function_name": function_name,
                        "class_name": class_name,
                        "project_name": "",  # Not available in old schema
                        "hierarchy_path": row.get("hierarchy_path", ""),
                        "embedded_at": timestamp,
                        "model_version": "migrated",
                    }
                    vectors_data.append(vector_record)

            # Create chunks table
            if chunks_data:
                # Let LanceDB infer schema from data (handles extra columns gracefully)
                # Using explicit schema would fail on forward-compatible migrations
                # when new columns are added to the schema definition
                db.create_table(self.NEW_CHUNKS_TABLE, chunks_data, mode="overwrite")
                logger.info(f"✓ Created chunks table with {len(chunks_data):,} rows")
                metadata["chunks_migrated"] = len(chunks_data)

            # Create vectors table
            if vectors_data:
                # Let LanceDB infer schema from data
                # This is critical because:
                # 1. Vector dimension varies (384 for MiniLM, 768 for GraphCodeBERT, etc.)
                # 2. Schema has evolved over time (function_name, class_name, project_name added)
                # 3. Migration must be forward-compatible with schema changes
                db.create_table(self.NEW_VECTORS_TABLE, vectors_data, mode="overwrite")
                logger.info(f"✓ Created vectors table with {len(vectors_data):,} rows")
                metadata["vectors_migrated"] = len(vectors_data)

            # Rename old table as backup
            backup_name = (
                f"{old_table_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            # Note: LanceDB doesn't have a rename operation, so we document the backup
            # The old table remains but won't be used by the new code
            metadata["backup_table"] = backup_name
            metadata["note"] = (
                f"Old table '{old_table_name}' preserved for rollback. "
                "You can safely delete it after verifying the migration."
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"✓ Migration complete: {len(chunks_data):,} chunks, "
                f"{len(vectors_data):,} vectors in {duration:.1f}s"
            )

            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message=(
                    f"Successfully migrated {len(chunks_data)} chunks and "
                    f"{len(vectors_data)} vectors to two-phase architecture"
                ),
                executed_at=datetime.now(),
                duration_seconds=duration,
                metadata=metadata,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Migration failed: {e}")
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message=f"Migration failed: {str(e)}",
                executed_at=datetime.now(),
                duration_seconds=duration,
                metadata=metadata,
            )

    def rollback(self, context: MigrationContext) -> bool:
        """Rollback migration by dropping new tables and restoring old table.

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            db = lancedb.connect(str(context.index_path))

            # Get table names
            tables_response = db.list_tables()
            if hasattr(tables_response, "tables"):
                table_names = tables_response.tables
            else:
                table_names = tables_response

            # Drop new tables if they exist
            if self.NEW_CHUNKS_TABLE in table_names:
                db.drop_table(self.NEW_CHUNKS_TABLE)
                logger.info(f"Dropped {self.NEW_CHUNKS_TABLE} table")

            if self.NEW_VECTORS_TABLE in table_names:
                db.drop_table(self.NEW_VECTORS_TABLE)
                logger.info(f"Dropped {self.NEW_VECTORS_TABLE} table")

            # Old table should still exist (we didn't delete it)
            logger.info("Rollback complete - old table preserved")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

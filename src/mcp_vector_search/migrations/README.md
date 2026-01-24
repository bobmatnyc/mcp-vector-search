# Migration System

Version-based migration system for mcp-vector-search database upgrades.

## Overview

The migration system automatically handles database schema changes, index rebuilds, and configuration updates when upgrading between versions. Migrations run automatically on:

- MCP server startup
- First CLI command after upgrade (optional)

## Architecture

### Components

1. **Migration** - Abstract base class for version-specific migrations
2. **MigrationRunner** - Executes and manages migration lifecycle
3. **MigrationRegistry** - Tracks executed migrations in `.mcp-vector-search/migrations.json`
4. **MigrationContext** - Provides project info and configuration to migrations

### Migration Lifecycle

```
1. Check if needed (migration.check_needed())
2. Execute migration (migration.execute())
3. Record result (registry.record_migration())
4. Optional rollback (migration.rollback())
```

## Usage

### CLI Commands

```bash
# List all migrations and their status
mcp-vector-search migrate list

# Run pending migrations
mcp-vector-search migrate

# Dry run (show what would be done)
mcp-vector-search migrate migrate --dry-run

# Run specific migration
mcp-vector-search migrate migrate --version 1.2.2

# Force re-run a migration
mcp-vector-search migrate migrate --force --version 1.2.2

# Show migration system status
mcp-vector-search migrate status
```

### Automatic Execution

Migrations run automatically on MCP server startup:

```python
# In mcp/server.py
async def initialize(self) -> None:
    # Run pending migrations first
    await self._run_migrations()

    # Continue with normal initialization...
```

Failed migrations log warnings but don't block startup to ensure availability.

## Creating Migrations

### 1. Define Migration Class

Create a new file in `src/mcp_vector_search/migrations/` following the naming pattern `v{VERSION}_{NAME}.py`:

```python
# migrations/v1_2_2_codexembed.py
from ..config.defaults import DEFAULT_EMBEDDING_MODELS
from .migration import Migration, MigrationContext, MigrationResult, MigrationStatus


class CodeXEmbedMigration(Migration):
    """Migrate from legacy embedding model to CodeXEmbed-400M."""

    version = "1.2.2"
    name = "codexembed_upgrade"
    description = "Upgrade to CodeXEmbed-400M embedding model"

    def check_needed(self, context: MigrationContext) -> bool:
        """Check if migration should run."""
        # Check current model dimensions
        current_model = context.get_config_value("embedding_model")
        # Return True if migration needed
        return current_model != DEFAULT_EMBEDDING_MODELS["code"]

    def execute(self, context: MigrationContext) -> MigrationResult:
        """Execute the migration."""
        if context.dry_run:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="DRY RUN: Would upgrade to CodeXEmbed-400M"
            )

        try:
            # 1. Backup metadata
            backup_path = self._backup_metadata(context)

            # 2. Clear old embeddings
            self._clear_old_embeddings(context)

            # 3. Update config
            self._update_config(context)

            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.SUCCESS,
                message="Successfully upgraded to CodeXEmbed-400M",
                metadata={"backup_path": str(backup_path)}
            )
        except Exception as e:
            return MigrationResult(
                migration_id=self.migration_id,
                version=self.version,
                name=self.name,
                status=MigrationStatus.FAILED,
                message=f"Migration failed: {e}"
            )

    def rollback(self, context: MigrationContext) -> bool:
        """Optional: Rollback the migration."""
        # Implement rollback logic if supported
        return False
```

### 2. Register Migration

Add the migration to the runner in two places:

**CLI Commands** (`cli/commands/migrate.py`):
```python
from ...migrations.v1_2_2_codexembed import CodeXEmbedMigration

def _get_runner(project_root: Path | None = None) -> MigrationRunner:
    runner = MigrationRunner(project_root or Path.cwd())
    runner.register_migrations([
        CodeXEmbedMigration(),
        # Add new migrations here
    ])
    return runner
```

**MCP Server** (`mcp/server.py`):
```python
async def _run_migrations(self) -> None:
    from ..migrations import MigrationRunner
    from ..migrations.v1_2_2_codexembed import CodeXEmbedMigration

    runner = MigrationRunner(self.project_root)
    runner.register_migrations([
        CodeXEmbedMigration(),
        # Add new migrations here
    ])
    # ... execute migrations
```

### 3. Test Migration

Create tests in `tests/unit/test_migrations.py`:

```python
def test_codexembed_migration(tmp_path: Path):
    """Test CodeXEmbed migration."""
    runner = MigrationRunner(tmp_path)
    migration = CodeXEmbedMigration()

    # Test check_needed
    context = MigrationContext(
        project_root=tmp_path,
        index_path=tmp_path / ".mcp-vector-search",
        config={"embedding_model": "old-model"}
    )
    assert migration.check_needed(context) is True

    # Test execute
    result = runner.run_migration(migration)
    assert result.status == MigrationStatus.SUCCESS
```

## Migration Best Practices

### Safety Features

1. **Idempotent Operations** - Safe to run multiple times
   ```python
   def check_needed(self, context):
       # Only run if actually needed
       return not already_migrated(context)
   ```

2. **Automatic Backups** - Backup before destructive operations
   ```python
   backup_path = self._backup_metadata(context)
   metadata["backup_path"] = str(backup_path)
   ```

3. **Progress Logging** - Log actions for debugging
   ```python
   logger.info(f"Cleared {count} chunks with old embeddings")
   ```

4. **Interrupt Handling** - Save state on Ctrl+C
   ```python
   signal.signal(signal.SIGINT, self._handle_interrupt)
   ```

5. **Dry Run Support** - Preview changes without executing
   ```python
   if context.dry_run:
       return "DRY RUN: Would do X"
   ```

### Error Handling

- Migrations return status (SUCCESS/FAILED/SKIPPED)
- Failed migrations don't block startup (warning only)
- Registry tracks all attempts (success and failure)
- Metadata includes error details for troubleshooting

### Versioning

- Use semantic versioning (1.2.2)
- Migrations run in version order
- Skip migrations older than last executed version
- Each migration has unique ID (`{version}_{name}`)

## Registry Storage

Migrations are tracked in `.mcp-vector-search/migrations.json`:

```json
{
  "migrations": [
    {
      "migration_id": "1.2.2_codexembed_upgrade",
      "version": "1.2.2",
      "name": "codexembed_upgrade",
      "status": "success",
      "message": "Successfully upgraded to CodeXEmbed-400M",
      "executed_at": "2024-01-15T10:30:00",
      "duration_seconds": 2.5,
      "metadata": {
        "backup_path": ".mcp-vector-search/backups/pre_codexembed_20240115_103000.json",
        "chunks_cleared": 1234
      }
    }
  ],
  "last_updated": "2024-01-15T10:30:00"
}
```

## Example Migrations

### 1. CodeXEmbed Upgrade (v1.2.2)

Migrates from all-MiniLM-L6-v2 (384 dims) to CodeXEmbed-400M (768 dims):

- Checks current model dimensions
- Backs up metadata
- Clears old embeddings
- Updates config to new model
- Triggers re-index on next index operation

### 2. Schema Update (template)

```python
class SchemaUpdateMigration(Migration):
    version = "2.0.0"
    name = "schema_update"

    def check_needed(self, context):
        # Check if schema needs update
        return not has_new_schema(context)

    def execute(self, context):
        # 1. Backup existing data
        # 2. Update schema
        # 3. Migrate data
        # 4. Validate migration
        pass
```

## Troubleshooting

### Migration Failed

1. Check migration status:
   ```bash
   mcp-vector-search migrate status
   ```

2. View migration history:
   ```bash
   cat .mcp-vector-search/migrations.json
   ```

3. Force re-run if needed:
   ```bash
   mcp-vector-search migrate migrate --force --version 1.2.2
   ```

### Rollback Not Supported

Most migrations delete old data and can't be automatically rolled back:

1. Restore from backup (if available)
2. Manually configure old settings
3. Re-index from scratch

### Clear Migration History

⚠️ Use with caution - this will re-run all migrations:

```python
from mcp_vector_search.migrations import MigrationRegistry
from pathlib import Path

registry = MigrationRegistry(Path(".mcp-vector-search"))
registry.clear_history()
```

## Future Enhancements

- [ ] SQLite storage for better query performance
- [ ] Migration dependencies (requires/blocks)
- [ ] Automatic backup rotation
- [ ] Migration dry-run validation
- [ ] Progress bars for long migrations
- [ ] Migration templates generator
- [ ] Automatic rollback support

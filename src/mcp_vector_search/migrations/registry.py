"""Migration registry for tracking executed migrations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .migration import MigrationResult, MigrationStatus


class MigrationRegistry:
    """Track which migrations have been executed.

    Stores migration history in .mcp-vector-search/migrations.json
    """

    def __init__(self, index_path: Path):
        """Initialize registry.

        Args:
            index_path: Path to .mcp-vector-search directory
        """
        self.index_path = index_path
        self.registry_file = index_path / "migrations.json"
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Create registry file if it doesn't exist."""
        if not self.registry_file.exists():
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._write_registry({"migrations": [], "last_updated": None})

    def _read_registry(self) -> dict[str, Any]:
        """Read registry data from file."""
        try:
            with open(self.registry_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to read migration registry: {e}")
            return {"migrations": [], "last_updated": None}

    def _write_registry(self, data: dict[str, Any]) -> None:
        """Write registry data to file."""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_executed_migrations(self) -> list[MigrationResult]:
        """Get list of all executed migrations."""
        data = self._read_registry()
        return [MigrationResult.from_dict(m) for m in data.get("migrations", [])]

    def get_last_version(self) -> str | None:
        """Get the last successfully executed migration version."""
        migrations = self.get_executed_migrations()
        successful = [m for m in migrations if m.status == MigrationStatus.SUCCESS]
        if not successful:
            return None

        # Sort by version (semantic versioning)
        sorted_migrations = sorted(
            successful, key=lambda m: self._version_tuple(m.version)
        )
        return sorted_migrations[-1].version if sorted_migrations else None

    def has_migration_run(self, migration_id: str) -> bool:
        """Check if a migration has been executed successfully."""
        migrations = self.get_executed_migrations()
        return any(
            m.migration_id == migration_id and m.status == MigrationStatus.SUCCESS
            for m in migrations
        )

    def record_migration(self, result: MigrationResult) -> None:
        """Record migration execution result."""
        data = self._read_registry()
        migrations = data.get("migrations", [])

        # Remove any previous attempts of this migration
        migrations = [
            m for m in migrations if m.get("migration_id") != result.migration_id
        ]

        # Add new result
        migrations.append(result.to_dict())

        data["migrations"] = migrations
        self._write_registry(data)

        logger.debug(
            f"Recorded migration {result.migration_id} with status {result.status}"
        )

    def get_migration_result(self, migration_id: str) -> MigrationResult | None:
        """Get the result of a specific migration."""
        migrations = self.get_executed_migrations()
        for m in migrations:
            if m.migration_id == migration_id:
                return m
        return None

    def clear_history(self) -> None:
        """Clear all migration history (use with caution!)."""
        logger.warning("Clearing migration history")
        self._write_registry({"migrations": [], "last_updated": None})

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, ...]:
        """Convert semantic version to tuple for sorting."""
        try:
            return tuple(int(x) for x in version.split("."))
        except (ValueError, AttributeError):
            return (0, 0, 0)
